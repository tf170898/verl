import numpy as np

from ..core import Deck, Card, Suit


class BlackjackCard(Card):
    ALL_VALID_SUITS = [
        Suit.HEARTS,
        Suit.DIAMONDS,
        Suit.CLUBS,
        Suit.SPADES
    ]


class BlackjackDeck(Deck[BlackjackCard]):

    def __init__(self, rng):
        ''' Initialize a Blackjack dealer class
        '''
        super().__init__(BlackjackCard)
        self.rng = rng
        self.shuffle()

        self.hand = []
        self.status = 'alive'
        self.score = 0

    def deal_card(self, player):
        ''' Distribute one card to the player

        Args:
            player_id (int): the target player's id
        '''
        card = self.remained_cards.pop()
        player.hand.append(card)
