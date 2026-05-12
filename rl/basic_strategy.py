"""
Hard-coded basic strategy for single-deck Blackjack.

Rules assumed: dealer stands on all 17s, double down on any 2 cards,
no split (out of scope for this project), no surrender.

Returns: "hit", "stand", or "doubleDown"

Sources: single-deck basic strategy charts are well-established in the
literature. This chart is the canonical no-hole-card, S17, no-split variant
consistent with the game rules implemented in env.py.
"""

# Hard totals: (player_total, dealer_upcard) -> action
# dealer_upcard uses the same encoding as the env: Ace=11, face=10
_HARD = {
    # 8 or less: always hit
    **{(t, d): "hit" for t in range(4, 9) for d in range(2, 12)},
    # 9: double vs 2-6, else hit
    (9, 2): "hit",
    (9, 3): "doubleDown",
    (9, 4): "doubleDown",
    (9, 5): "doubleDown",
    (9, 6): "doubleDown",
    (9, 7): "hit",
    (9, 8): "hit",
    (9, 9): "hit",
    (9, 10): "hit",
    (9, 11): "hit",
    # 10: double vs 2-9, else hit
    (10, 2): "doubleDown",
    (10, 3): "doubleDown",
    (10, 4): "doubleDown",
    (10, 5): "doubleDown",
    (10, 6): "doubleDown",
    (10, 7): "doubleDown",
    (10, 8): "doubleDown",
    (10, 9): "doubleDown",
    (10, 10): "hit",
    (10, 11): "hit",
    # 11: double vs 2-10, hit vs Ace
    (11, 2): "doubleDown",
    (11, 3): "doubleDown",
    (11, 4): "doubleDown",
    (11, 5): "doubleDown",
    (11, 6): "doubleDown",
    (11, 7): "doubleDown",
    (11, 8): "doubleDown",
    (11, 9): "doubleDown",
    (11, 10): "doubleDown",
    (11, 11): "hit",
    # 12: stand vs 4-6, else hit
    (12, 2): "hit",
    (12, 3): "hit",
    (12, 4): "stand",
    (12, 5): "stand",
    (12, 6): "stand",
    (12, 7): "hit",
    (12, 8): "hit",
    (12, 9): "hit",
    (12, 10): "hit",
    (12, 11): "hit",
    # 13-16: stand vs 2-6, else hit
    **{(t, d): "stand" if d <= 6 else "hit" for t in range(13, 17) for d in range(2, 12)},
    # 17-21: always stand
    **{(t, d): "stand" for t in range(17, 22) for d in range(2, 12)},
}

# Soft totals (hand contains an Ace counted as 11):
# key is the non-ace total (so soft 18 = Ace + 7, key is 7... actually
# easier to key on the full hand value)
# Soft 13-17: various doubles/hits; Soft 18+: stand or double
_SOFT = {
    # Soft 13 (A+2): double vs 5-6, else hit
    (13, 2): "hit",
    (13, 3): "hit",
    (13, 4): "hit",
    (13, 5): "doubleDown",
    (13, 6): "doubleDown",
    (13, 7): "hit",
    (13, 8): "hit",
    (13, 9): "hit",
    (13, 10): "hit",
    (13, 11): "hit",
    # Soft 14 (A+3): double vs 5-6, else hit
    (14, 2): "hit",
    (14, 3): "hit",
    (14, 4): "hit",
    (14, 5): "doubleDown",
    (14, 6): "doubleDown",
    (14, 7): "hit",
    (14, 8): "hit",
    (14, 9): "hit",
    (14, 10): "hit",
    (14, 11): "hit",
    # Soft 15 (A+4): double vs 4-6, else hit
    (15, 2): "hit",
    (15, 3): "hit",
    (15, 4): "doubleDown",
    (15, 5): "doubleDown",
    (15, 6): "doubleDown",
    (15, 7): "hit",
    (15, 8): "hit",
    (15, 9): "hit",
    (15, 10): "hit",
    (15, 11): "hit",
    # Soft 16 (A+5): double vs 4-6, else hit
    (16, 2): "hit",
    (16, 3): "hit",
    (16, 4): "doubleDown",
    (16, 5): "doubleDown",
    (16, 6): "doubleDown",
    (16, 7): "hit",
    (16, 8): "hit",
    (16, 9): "hit",
    (16, 10): "hit",
    (16, 11): "hit",
    # Soft 17 (A+6): double vs 2-6, else hit
    (17, 2): "doubleDown",
    (17, 3): "doubleDown",
    (17, 4): "doubleDown",
    (17, 5): "doubleDown",
    (17, 6): "doubleDown",
    (17, 7): "hit",
    (17, 8): "hit",
    (17, 9): "hit",
    (17, 10): "hit",
    (17, 11): "hit",
    # Soft 18 (A+7): double vs 2-6, stand vs 7-8, hit vs 9-A
    (18, 2): "doubleDown",
    (18, 3): "doubleDown",
    (18, 4): "doubleDown",
    (18, 5): "doubleDown",
    (18, 6): "doubleDown",
    (18, 7): "stand",
    (18, 8): "stand",
    (18, 9): "hit",
    (18, 10): "hit",
    (18, 11): "hit",
    # Soft 19+ always stand
    **{(t, d): "stand" for t in range(19, 22) for d in range(2, 12)},
}


def basic_strategy_action(
    player_total: int,
    is_soft: bool,
    dealer_upcard: int,
    can_double: bool,
) -> str:
    """
    Return the basic-strategy action for the given state.

    If the chart calls for doubleDown but can_double is False (3+ card hand),
    fall back to hit (standard casino rule: no double after hit).
    """
    table = _SOFT if is_soft else _HARD
    action = table.get((player_total, dealer_upcard), "stand")

    if action == "doubleDown" and not can_double:
        action = "hit"

    return action
