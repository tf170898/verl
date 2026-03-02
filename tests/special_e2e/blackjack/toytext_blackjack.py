"""Self-contained toy-text blackjack env for e2e smoke tests.

This mirrors the simplified blackjack dynamics used by redstone's toy-text env,
without importing redstone or gymnasium.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


OBSERVATION_PROMPT_TEMPLATE = (
    "Your hand sums to {player_sum}, the dealer shows {dealer_card}. Usable ace: {usable_ace}."
)

SYSTEM_PROMPT = (
    "You are playing blackjack. Respond ONLY with 'hit' or 'stand'. "
    "Do not add any additional information."
)


@dataclass(frozen=True)
class BlackjackObservation:
    player_sum: int
    dealer_card: int
    usable_ace: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (int(self.player_sum), int(self.dealer_card), int(self.usable_ace))


def format_observation_prompt(observation: tuple[int, int, int] | BlackjackObservation) -> str:
    if isinstance(observation, BlackjackObservation):
        obs = observation
    else:
        obs = BlackjackObservation(
            player_sum=int(observation[0]),
            dealer_card=int(observation[1]),
            usable_ace=int(observation[2]),
        )
    return OBSERVATION_PROMPT_TEMPLATE.format(
        player_sum=obs.player_sum,
        dealer_card=obs.dealer_card,
        usable_ace=bool(obs.usable_ace),
    )


def build_user_prompt(observation: tuple[int, int, int] | BlackjackObservation) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"{format_observation_prompt(observation)}\n"
        "Reply with one word: hit or stand."
    )


class ToyTextBlackjackEnv:
    """Deterministic toy-text blackjack environment."""

    HIT = 0
    STAND = 1
    _TOY_TEXT_DECK = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self._player_hand: list[int] = []
        self._dealer_hand: list[int] = []
        self._terminated = False
        self._last_observation = BlackjackObservation(0, 1, 0).as_tuple()

    def reset(self, seed: Optional[int] = None) -> tuple[int, int, int]:
        seed_value = 0 if seed is None else int(seed) % (2**32)
        self._rng = np.random.default_rng(seed_value)
        self._terminated = False
        self._player_hand = self._draw_hand_values()
        self._dealer_hand = self._draw_hand_values()
        obs = self._get_observation()
        self._last_observation = obs
        return obs

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool]:
        if self._terminated:
            return self._last_observation, 0.0, True

        if action not in (self.HIT, self.STAND):
            raise ValueError(f"Invalid action {action}. Expected 0 (hit) or 1 (stand).")

        reward = 0.0
        if action == self.HIT:
            self._player_hand.append(self._draw_card())
            if self._is_bust(self._player_hand):
                self._terminated = True
                reward = -1.0
        else:
            self._terminated = True
            while self._sum_hand(self._dealer_hand) < 17:
                self._dealer_hand.append(self._draw_card())
            reward = self._cmp(self._score(self._player_hand), self._score(self._dealer_hand))

        obs = self._get_observation()
        self._last_observation = obs
        return obs, reward, self._terminated

    def _draw_card(self) -> int:
        return int(self._rng.choice(self._TOY_TEXT_DECK))

    def _draw_hand_values(self) -> list[int]:
        return [self._draw_card(), self._draw_card()]

    @staticmethod
    def _hand_sum_and_usable_ace(hand: list[int]) -> tuple[int, int]:
        total = int(sum(hand))
        if 1 in hand and total + 10 <= 21:
            return total + 10, 1
        return total, 0

    def _sum_hand(self, hand: list[int]) -> int:
        total, _ = self._hand_sum_and_usable_ace(hand)
        return total

    def _is_bust(self, hand: list[int]) -> bool:
        return self._sum_hand(hand) > 21

    def _score(self, hand: list[int]) -> int:
        return 0 if self._is_bust(hand) else self._sum_hand(hand)

    @staticmethod
    def _cmp(a: int, b: int) -> float:
        return float(a > b) - float(a < b)

    def _get_observation(self) -> tuple[int, int, int]:
        if not self._player_hand or not self._dealer_hand:
            return BlackjackObservation(0, 1, 0).as_tuple()
        player_sum, usable_ace = self._hand_sum_and_usable_ace(self._player_hand)
        obs = BlackjackObservation(
            player_sum=player_sum,
            dealer_card=int(self._dealer_hand[0]),
            usable_ace=usable_ace,
        )
        return obs.as_tuple()
