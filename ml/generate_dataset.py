import argparse
import csv
import random
from pathlib import Path


SUITS = ["spades", "hearts", "diamonds", "clubs"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
RANK_COUNT_COLUMNS = [
    "count_ace",
    "count_2",
    "count_3",
    "count_4",
    "count_5",
    "count_6",
    "count_7",
    "count_8",
    "count_9",
    "count_10",
]
FEATURE_COLUMNS = [
    "player_total",
    "is_soft_hand",
    "player_card_count",
    "dealer_upcard_value",
    "can_double_down",
    "remaining_cards",
    *RANK_COUNT_COLUMNS,
]
LABEL_COLUMN = "recommended_action"
OUTPUT_COLUMNS = [*FEATURE_COLUMNS, LABEL_COLUMN]


def create_deck():
    return [{"rank": rank, "suit": suit} for suit in SUITS for rank in RANKS]


def card_value(rank):
    if rank == "A":
        return 11
    if rank in {"J", "Q", "K"}:
        return 10
    return int(rank)


def hand_value(hand):
    total = sum(card_value(card["rank"]) for card in hand)
    aces = sum(1 for card in hand if card["rank"] == "A")

    while total > 21 and aces:
        total -= 10
        aces -= 1

    return total


def is_soft_hand(hand):
    hard_total = sum(1 if card["rank"] == "A" else card_value(card["rank"]) for card in hand)
    return any(card["rank"] == "A" for card in hand) and hard_total + 10 <= 21


def is_bust(hand):
    return hand_value(hand) > 21


def draw_card(deck):
    return deck.pop()


def deal_random_state(rng):
    deck = create_deck()
    rng.shuffle(deck)

    player_hand = [draw_card(deck), draw_card(deck)]
    dealer_hand = [draw_card(deck), draw_card(deck)]

    # Add occasional extra player cards so the dataset includes post-hit decisions.
    while hand_value(player_hand) < 17 and len(player_hand) < 5 and rng.random() < 0.35:
        player_hand.append(draw_card(deck))
        if is_bust(player_hand):
            return deal_random_state(rng)

    return {
        "player_hand": player_hand,
        "dealer_hand": dealer_hand,
        "remaining_deck": deck,
        "can_double_down": len(player_hand) == 2,
    }


def legal_actions(state):
    if is_bust(state["player_hand"]):
        return []

    actions = ["hit", "stand"]
    if state["can_double_down"]:
        actions.append("doubleDown")
    return actions


def label_state_with_monte_carlo(state, simulations, rng):
    action_results = {
        action: simulate_action(action, state, simulations, rng)
        for action in legal_actions(state)
    }
    return max(action_results, key=lambda action: action_results[action]["expected_value"])


def simulate_action(action, state, simulations, rng):
    wins = 0
    losses = 0

    for _ in range(simulations):
        outcome = simulate_round(action, state, rng)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1

    win_rate = wins / simulations
    loss_rate = losses / simulations
    multiplier = 2 if action == "doubleDown" else 1

    return {
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "expected_value": multiplier * (win_rate - loss_rate),
    }


def simulate_round(action, state, rng):
    player_hand = clone_cards(state["player_hand"])
    dealer_hand = clone_cards(state["dealer_hand"])
    deck = clone_cards(state["remaining_deck"])
    rng.shuffle(deck)

    if action == "hit":
        player_hand.append(draw_card(deck))
        complete_player_turn(player_hand, deck)
    elif action == "doubleDown":
        player_hand.append(draw_card(deck))

    if is_bust(player_hand):
        return "loss"

    play_dealer_turn(dealer_hand, deck)
    return resolve_round(player_hand, dealer_hand)


def complete_player_turn(player_hand, deck):
    while not is_bust(player_hand) and hand_value(player_hand) < 17:
        player_hand.append(draw_card(deck))


def play_dealer_turn(dealer_hand, deck):
    while hand_value(dealer_hand) < 17:
        dealer_hand.append(draw_card(deck))


def resolve_round(player_hand, dealer_hand):
    player_score = hand_value(player_hand)
    dealer_score = hand_value(dealer_hand)

    if dealer_score > 21:
        return "win"
    if player_score > dealer_score:
        return "win"
    if player_score < dealer_score:
        return "loss"
    return "push"


def clone_cards(cards):
    return [{"rank": card["rank"], "suit": card["suit"]} for card in cards]


def state_to_row(state, recommended_action):
    player_hand = state["player_hand"]
    dealer_upcard = state["dealer_hand"][0]
    remaining_deck = state["remaining_deck"]
    rank_counts = count_remaining_ranks(remaining_deck)

    return {
        "player_total": hand_value(player_hand),
        "is_soft_hand": int(is_soft_hand(player_hand)),
        "player_card_count": len(player_hand),
        "dealer_upcard_value": min(card_value(dealer_upcard["rank"]), 10),
        "can_double_down": int(state["can_double_down"]),
        "remaining_cards": len(remaining_deck),
        **rank_counts,
        "recommended_action": recommended_action,
    }


def count_remaining_ranks(deck):
    counts = {column: 0 for column in RANK_COUNT_COLUMNS}

    for card in deck:
        rank = card["rank"]
        if rank == "A":
            counts["count_ace"] += 1
        elif rank in {"10", "J", "Q", "K"}:
            counts["count_10"] += 1
        else:
            counts[f"count_{rank}"] += 1

    return counts


def write_dataset(output_path, rows, simulations, seed):
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row_index in range(1, rows + 1):
            state = deal_random_state(rng)
            recommended_action = label_state_with_monte_carlo(state, simulations, rng)
            writer.writerow(state_to_row(state, recommended_action))

            if row_index % 500 == 0:
                print(f"Generated {row_index}/{rows} rows")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Monte Carlo-labeled Blackjack decision states.")
    parser.add_argument("--rows", type=int, default=1000, help="Number of dataset rows to generate.")
    parser.add_argument("--simulations", type=int, default=500, help="Monte Carlo simulations per legal action.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/data/blackjack_states.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    write_dataset(args.output, args.rows, args.simulations, args.seed)
    print(f"Dataset written to {args.output}")


if __name__ == "__main__":
    main()
