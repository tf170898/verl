from enum import Enum
from typing import Optional

import numpy as np
from gymnasium.utils import seeding

from ..core import Game, Rank, Suit
from .deck import BlackjackDeck as Deck, BlackjackCard as Card



class BlackjackPlayer:
    class Status(Enum):
        ALIVE = 'alive'
        BUST = 'bust'
        STAND = 'stand'

    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = []
        self.status = self.Status.ALIVE
        self.score = 0

    def __str__(self):
        return f"Player {self.player_id}: Status={self.status}, Score={self.score}"

    def is_active(self) -> bool:
        return self.status == self.Status.ALIVE


class BlackjackDealer(BlackjackPlayer):
    pass


class BlackjackAction(Enum):
    HIT = 'hit'
    STAND = 'stand'


class BlackjackGame(Game):
    THRESHOLD = 21
    DEALER_THRESHOLD = 17

    class State(Enum): # pyright: ignore[reportIncompatibleVariableOverride]
        INITIALIZED = 1
        WAITING_ACTION = 2
        COMPLETED = 3

    def __init__(self, n_player: int = 1, allow_step_back: bool = False, seed: Optional[int] = None):
        """Initialize the class Blackjack Game."""
        super().__init__()

        self.n_player = n_player
        self.allow_step_back = allow_step_back
        self.rng = np.random.RandomState()
        self.seed(seed)

    def seed(self, seed: Optional[int] = None) -> int:
        """Seed the RNG for deterministic play and return the effective seed."""
        if seed is None:
            self.seed_value = 0
        else:
            self.seed_value = int(seed) % (2**32)
        self.rng = np.random.RandomState(self.seed_value)
        return self.seed_value

    def reset(self):
        self.init_game()

    def init_game(self) -> tuple[dict, int]:
        ''' Initialilze the game

        Returns:
            state (dict): the first state of the game
            player_id (int): current player's id
        '''
        # -1: lose because of bust, -2: lose because of score lower than dealer
        # 0: not decided yet, 1: win by dealer bust, 2: win by score higher than dealer
        self.winner = np.zeros(self.n_player + 1, dtype=np.int8)
        self.state = self.State.INITIALIZED

        self.history = []

        self.deck = Deck(self.rng)
        self.players = [BlackjackPlayer(i + 1) for i in range(self.n_player)]
        self.dealer = BlackjackDealer(0)

        self.deck.shuffle()
        self._init_round()

        self.state = self.State.WAITING_ACTION
        self.player_id = 1

    def run(self):
        self.init_game()
        while not self.is_over():
            self.step()

        self.judge_round()

        print("Final dealer hand: ", [str(card) for card in self.dealer.hand], " score: ", self.dealer.score)
        print("Final players' hands and scores:")
        for player in self.players:
            print(f"Player {player.player_id}: {[str(card) for card in player.hand]}, score: {player.score}")

        winner_ids = np.where(self.winner > 0)[0]
        if len(winner_ids) == 0:
            print("Game over! Dealer wins.")
        else:
            print(f"Game over! Winner IDs: {winner_ids}")

    def step(self, action: Optional[BlackjackAction]=None) -> BlackjackPlayer:
        print(f"Player {self.player_id}'s turn.")
        print(f"Current dealer hand: {[str(card) for card in self.dealer.hand]}, score: {self.dealer.score}")
        print("Current players' hands and scores:")
        for player in self.players:
            print(f"Player {player.player_id}: {[str(card) for card in player.hand]}, score: {player.score}")
        player = self.get_current_player()

        self.step_player(player, action)
        if isinstance(player, BlackjackDealer):
            if any(p.status == p.Status.STAND for p in self.players):
                self.step_player(self.dealer)
            self.state = self.State.COMPLETED

        return self.get_current_player()

    def step_player(self, player: BlackjackPlayer, action: Optional[BlackjackAction]=None):
        match player:
            case BlackjackDealer():
                while player.status == BlackjackPlayer.Status.ALIVE:
                    if player.score < self.DEALER_THRESHOLD:
                        self.deck.deal_card(player)
                        player.status, player.score = self._judge_player(player)
                    else:
                        player.status = BlackjackPlayer.Status.STAND
            case BlackjackPlayer():
                action = self._ask_action(player, action)

                if player.status == BlackjackPlayer.Status.ALIVE:
                    if action == BlackjackAction.HIT:
                        self.deck.deal_card(player)
                    elif action == BlackjackAction.STAND:
                        player.status = BlackjackPlayer.Status.STAND
                        self.player_id = (self.player_id + 1) % (self.n_player + 1)
        player.status, player.score = self._judge_player(player)

    def _ask_action(self, player: BlackjackPlayer, action: Optional[BlackjackAction]=None) -> BlackjackAction:
        if action is not None:
            return action

        msg = f"[Message to Player {player}]: Choosing action from "
        mapping = {}
        for key, action in enumerate(BlackjackAction):
            mapping[key] = action
            msg += f"({key}: {action.value} ) "
        print(msg)
        while True:
            choice = input(f"Please input your action choice (0-{len(BlackjackAction)-1}): ")
            try:
                choice = int(choice)
                if choice in mapping:
                    return mapping[choice]
                else:
                    print(f"Invalid choice. Please choose a number between 0 and {len(BlackjackAction)-1}.")
            except ValueError:
                print("Invalid input. Please enter a number.")


    def _init_round(self):
        for _ in range(2):
            for player in self.players:
                self.deck.deal_card(player)
            self.deck.deal_card(self.dealer)

        self.current_player_id = 1

        self.judge_round()

    def _rank2score(self, rank: Rank) -> int:
        if rank in [Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN]:
            return 10
        elif rank == Rank.ACE:
            return 11
        else:
            return int(rank.value)

    def _judge_score(self, cards: list) -> int:
        score = 0
        n_aces = 0
        for card in cards:
            card_score = self._rank2score(card.rank)
            score += card_score
            if card.rank == Rank.ACE:
                n_aces += 1

        while score > self.THRESHOLD and n_aces > 0:
            score -= 10
            n_aces -= 1

        return score


    def _judge_player(self, player: BlackjackPlayer) -> tuple[BlackjackPlayer.Status, int]:
        score = self._judge_score(player.hand)
        if player.status == BlackjackPlayer.Status.STAND:
            return BlackjackPlayer.Status.STAND, score
        if score <= self.THRESHOLD:
            return BlackjackPlayer.Status.ALIVE, score
        else:
            return BlackjackPlayer.Status.BUST, score


    def get_current_player(self) -> BlackjackPlayer:
        if self.player_id == 0:
            return self.dealer
        return self.players[self.player_id - 1]

    def judge_round(self):
        # Refresh scores before adjudication
        for i, player in enumerate(self.players):
            match (player.status, self.dealer.status):
                case (BlackjackPlayer.Status.BUST, _):
                    self.winner[i + 1] = -1
                case (_, BlackjackPlayer.Status.BUST):
                    self.winner[i + 1] = 1
                case _:
                    if self.dealer.score < self.DEALER_THRESHOLD:
                        return # Dealer hasn't finished yet
                    if player.score > self.dealer.score:
                        self.winner[i + 1] = 2
                    elif player.score < self.dealer.score:
                        self.winner[i + 1] = -1

    def is_over(self) -> bool:
        return self.state == self.State.COMPLETED


class ToyTextBlackjackGame(BlackjackGame):
    """Blackjack implementation that mirrors Gymnasium's Blackjack-v1 behavior."""

    HIT = 0
    STAND = 1
    _TOY_TEXT_DECK = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)

    def __init__(self, n_player: int = 1, allow_step_back: bool = False, seed: Optional[int] = None):
        super().__init__(n_player=n_player, allow_step_back=allow_step_back, seed=seed)
        self._np_random, _ = seeding.np_random(seed)
        self._terminated = False
        self._last_observation: tuple[int, int, int] = (0, 1, 0)
        self._player_hand_values: list[int] = []
        self._dealer_hand_values: list[int] = []

    def seed(self, seed: Optional[int] = None) -> int:
        result = super().seed(seed)
        self._np_random, _ = seeding.np_random(self.seed_value)
        return result

    def reset_round(self, seed: Optional[int] = None) -> tuple[tuple[int, int, int], dict]:
        self.seed(seed)
        self._terminated = False
        self.winner = np.zeros(self.n_player + 1, dtype=np.int8)
        self.state = self.State.INITIALIZED
        self.history = []
        self.players = [BlackjackPlayer(i + 1) for i in range(self.n_player)]
        self.dealer = BlackjackDealer(0)
        self.player_id = 1
        self._dealer_hand_values = self._draw_hand_values()
        self._player_hand_values = self._draw_hand_values()

        for idx, player in enumerate(self.players):
            values = self._player_hand_values if idx == 0 else self._draw_hand_values()
            if idx == 0:
                self._player_hand_values = values
            player.hand = [self._value_to_card(v) for v in values]
            player.status, player.score = self._judge_player(player)
        self.dealer.hand = [self._value_to_card(v) for v in self._dealer_hand_values]
        self.dealer.status, self.dealer.score = self._judge_player(self.dealer)

        self.state = self.State.WAITING_ACTION
        observation = self._get_observation()
        self._consume_render_rng_noise(observation[1])
        self._last_observation = observation
        self._sync_game_state(reset_winner=True)
        return observation, {}

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool, bool, dict]:
        if self._terminated:
            return self._last_observation, 0.0, True, False, {}

        if action not in (self.HIT, self.STAND):
            raise ValueError(f"Invalid action {action}. Expected 0 (hit) or 1 (stand).")

        reward = 0.0
        if action == self.HIT:
            self._player_hand_values.append(self._draw_card())
            if self._is_bust(self._player_hand_values):
                self._terminated = True
                reward = -1.0
        else:
            self._terminated = True
            while self._sum_hand(self._dealer_hand_values) < 17:
                self._dealer_hand_values.append(self._draw_card())
            reward = self._cmp(self._score(self._player_hand_values), self._score(self._dealer_hand_values))

        observation = self._get_observation()
        self._last_observation = observation
        self._update_winner_from_reward(reward)
        self._sync_game_state()
        return observation, reward, self._terminated, False, {}

    # Helper methods --------------------------------------------------------------------

    def _draw_card(self) -> int:
        return int(self._np_random.choice(self._TOY_TEXT_DECK))

    def _draw_hand_values(self) -> list[int]:
        return [self._draw_card(), self._draw_card()]

    def _give_card(self, player: BlackjackPlayer, value: int) -> None:
        if isinstance(player, BlackjackDealer):
            self._dealer_hand_values.append(value)
        else:
            self._player_hand_values.append(value)
        player.hand.append(self._value_to_card(value))
        player.status, player.score = self._judge_player(player)

    def _value_to_card(self, value: int) -> Card:
        rank_map = {
            1: Rank.ACE,
            2: Rank.TWO,
            3: Rank.THREE,
            4: Rank.FOUR,
            5: Rank.FIVE,
            6: Rank.SIX,
            7: Rank.SEVEN,
            8: Rank.EIGHT,
            9: Rank.NINE,
            10: Rank.TEN,
        }
        return Card(Suit.SPADES, rank_map[value])

    def _sum_hand(self, hand: list[int]) -> int:
        return self._hand_sum_and_usable_ace(hand)[0]

    def _usable_ace(self, hand: list[int]) -> int:
        return self._hand_sum_and_usable_ace(hand)[1]

    def _is_bust(self, hand: list[int]) -> bool:
        return self._sum_hand(hand) > 21

    def _score(self, hand: list[int]) -> int:
        return 0 if self._is_bust(hand) else self._sum_hand(hand)

    def _cmp(self, a: int, b: int) -> float:
        return float(a > b) - float(a < b)

    @staticmethod
    def _hand_sum_and_usable_ace(hand: list[int]) -> tuple[int, int]:
        total = sum(hand)
        if 1 in hand and total + 10 <= 21:
            return total + 10, 1
        return total, 0

    def _get_observation(self) -> tuple[int, int, int]:
        if not self._player_hand_values or not self._dealer_hand_values:
            return (0, 1, 0)
        player_sum, usable = self._hand_sum_and_usable_ace(self._player_hand_values)
        return (player_sum, self._dealer_hand_values[0], usable)

    def _consume_render_rng_noise(self, dealer_card_value: int) -> None:
        suits = ["C", "D", "H", "S"]
        self._np_random.choice(suits)
        if dealer_card_value == 10:
            self._np_random.choice(["J", "Q", "K"])

    def _sync_game_state(self, reset_winner: bool = False) -> None:
        if not self.players:
            return
        if reset_winner:
            self.winner = self.winner * 0
        player = self.players[0]
        dealer = self.dealer
        player.hand = [self._value_to_card(v) for v in self._player_hand_values]
        dealer.hand = [self._value_to_card(v) for v in self._dealer_hand_values]
        player.status, player.score = self._judge_player(player)
        dealer.status, dealer.score = self._judge_player(dealer)
        self.state = self.State.COMPLETED if self._terminated else self.State.WAITING_ACTION

    def _update_winner_from_reward(self, reward: float) -> None:
        if reward > 0:
            self.winner[1] = 1
        elif reward < 0:
            self.winner[1] = -1
        else:
            self.winner[1] = 0
