"""
Gymnasium environment for single-deck Blackjack.

Game rules match the existing Monte Carlo simulator and ML dataset exactly:
- Single 52-card deck, reshuffled each hand
- Actions: hit, stand, double down (legal only on 2-card hands)
- Dealer stands on all 17s
- Reward: +1 win, 0 push, -1 loss, doubled for double down
- Observation: 16-feature vector identical to ml/generate_dataset.py
"""

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SUITS = ["spades", "hearts", "diamonds", "clubs"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

ACTION_HIT = 0
ACTION_STAND = 1
ACTION_DOUBLE = 2

# Observation vector index order matches FEATURE_COLUMNS in generate_dataset.py
OBS_PLAYER_TOTAL = 0
OBS_IS_SOFT = 1
OBS_CARD_COUNT = 2
OBS_DEALER_UP = 3
OBS_CAN_DOUBLE = 4
OBS_REMAINING = 5
# indices 6-15: count_ace, count_2, ..., count_10


def _card_value(rank: str) -> int:
    if rank == "A":
        return 11
    if rank in {"J", "Q", "K"}:
        return 10
    return int(rank)


def _hand_value(hand: list) -> int:
    total = sum(_card_value(c["rank"]) for c in hand)
    aces = sum(1 for c in hand if c["rank"] == "A")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def _is_soft(hand: list) -> bool:
    hard = sum(1 if c["rank"] == "A" else _card_value(c["rank"]) for c in hand)
    return any(c["rank"] == "A" for c in hand) and hard + 10 <= 21


def _count_ranks(deck: list) -> list:
    """Return 10-element list: [ace, 2, 3, 4, 5, 6, 7, 8, 9, 10+face]."""
    counts = [0] * 10
    rank_idx = {"A": 0, "2": 1, "3": 2, "4": 3, "5": 4,
                "6": 5, "7": 6, "8": 7, "9": 8}
    for c in deck:
        r = c["rank"]
        if r in rank_idx:
            counts[rank_idx[r]] += 1
        else:  # 10, J, Q, K
            counts[9] += 1
    return counts


def _create_deck() -> list:
    return [{"rank": r, "suit": s} for s in SUITS for r in RANKS]


class BlackjackEnv(gym.Env):
    """
    Single-deck Blackjack as a Gymnasium environment.

    Observation space: Box(16,) with the same 16 features used by the
    supervised ML model (player_total, is_soft_hand, player_card_count,
    dealer_upcard_value, can_double_down, remaining_cards, count_ace..count_10).

    Action space: Discrete(3) — hit=0, stand=1, doubleDown=2.
    Use action_masks() to retrieve a boolean mask; illegal actions are never
    sampled when using MaskablePPO.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()

        self.observation_space = spaces.Box(
            low=np.array([4, 0, 2, 2, 0, 0] + [0] * 10, dtype=np.float32),
            high=np.array([21, 1, 10, 11, 1, 52] + [4] * 10, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._rng = random.Random(seed)
        self._player: list = []
        self._dealer: list = []
        self._deck: list = []
        self._doubled: bool = False

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = random.Random(seed)

        self._deck = _create_deck()
        self._rng.shuffle(self._deck)
        self._doubled = False

        self._player = [self._deck.pop(), self._deck.pop()]
        self._dealer = [self._deck.pop(), self._deck.pop()]

        return self._obs(), {}

    def step(self, action: int):
        assert not self._is_terminal_state(), "step() called on finished episode"

        reward = 0.0
        terminated = False

        if action == ACTION_HIT:
            self._player.append(self._deck.pop())
            if _hand_value(self._player) > 21:
                reward = -1.0
                terminated = True
            elif _hand_value(self._player) == 21:
                self._play_dealer()
                reward = self._resolve()
                terminated = True

        elif action == ACTION_STAND:
            self._play_dealer()
            reward = self._resolve()
            terminated = True

        elif action == ACTION_DOUBLE:
            self._doubled = True
            self._player.append(self._deck.pop())
            if _hand_value(self._player) > 21:
                reward = -2.0
            else:
                self._play_dealer()
                reward = self._resolve() * 2
            terminated = True

        return self._obs(), reward, terminated, False, {}

    def action_masks(self) -> np.ndarray:
        can_double = len(self._player) == 2
        return np.array([True, True, can_double], dtype=bool)

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        rank_counts = _count_ranks(self._deck)
        obs = np.array(
            [
                _hand_value(self._player),
                int(_is_soft(self._player)),
                len(self._player),
                min(_card_value(self._dealer[0]["rank"]), 10),
                int(len(self._player) == 2),
                len(self._deck),
            ]
            + rank_counts,
            dtype=np.float32,
        )
        return obs

    def _play_dealer(self):
        while _hand_value(self._dealer) < 17:
            self._dealer.append(self._deck.pop())

    def _resolve(self) -> float:
        p = _hand_value(self._player)
        d = _hand_value(self._dealer)
        if d > 21 or p > d:
            return 1.0
        if p < d:
            return -1.0
        return 0.0

    def _is_terminal_state(self) -> bool:
        return _hand_value(self._player) > 21
