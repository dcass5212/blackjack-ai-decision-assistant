import argparse
from pathlib import Path


SUITS = ["spades", "hearts", "diamonds", "clubs"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
RANK_ALIASES = {
    "1": "A",
    "ACE": "A",
    "T": "10",
    "JACK": "J",
    "QUEEN": "Q",
    "KING": "K",
}
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
SUIT_ALIASES = {
    "S": "spades",
    "SPADE": "spades",
    "SPADES": "spades",
    "H": "hearts",
    "HEART": "hearts",
    "HEARTS": "hearts",
    "D": "diamonds",
    "DIAMOND": "diamonds",
    "DIAMONDS": "diamonds",
    "C": "clubs",
    "CLUB": "clubs",
    "CLUBS": "clubs",
}
SAMPLE_HANDS = {
    "hard16": {
        "label": "Hard 16 vs dealer 10",
        "player_cards": ["10S", "6H"],
        "dealer_upcard": "10C",
    },
    "soft18": {
        "label": "Soft 18 vs dealer 9",
        "player_cards": ["AH", "7C"],
        "dealer_upcard": "9S",
    },
    "double11": {
        "label": "11 vs dealer 6",
        "player_cards": ["5C", "6S"],
        "dealer_upcard": "6H",
    },
    "stand20": {
        "label": "20 vs dealer 10",
        "player_cards": ["KH", "QC"],
        "dealer_upcard": "10H",
    },
}


def create_deck():
    return [{"rank": rank, "suit": suit} for suit in SUITS for rank in RANKS]


def normalize_rank(value):
    rank = value.strip().upper()
    rank = RANK_ALIASES.get(rank, rank)

    if rank not in RANKS:
        raise ValueError(f"Unknown card rank: {value}")

    return rank


def parse_card(value):
    token = value.strip().upper()

    if not token:
        raise ValueError("Card value cannot be empty")

    if " " in token:
        rank_text, suit_text = token.split(None, 1)
        return {"rank": normalize_rank(rank_text), "suit": parse_suit(suit_text)}

    if len(token) >= 2 and token[-1] in SUIT_ALIASES:
        return {"rank": normalize_rank(token[:-1]), "suit": parse_suit(token[-1])}

    return {"rank": normalize_rank(token), "suit": None}


def parse_suit(value):
    suit = SUIT_ALIASES.get(value.strip().upper())

    if not suit:
        raise ValueError(f"Unknown card suit: {value}")

    return suit


def parse_cards(value):
    return [parse_card(card_text) for card_text in value.split(",") if card_text.strip()]


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


def remove_visible_card(deck, visible_card):
    for index, deck_card in enumerate(deck):
        if visible_card["suit"] and deck_card["suit"] != visible_card["suit"]:
            continue
        if deck_card["rank"] == visible_card["rank"]:
            deck.pop(index)
            return

    card_name = visible_card["rank"]
    if visible_card["suit"]:
        card_name = f"{card_name} of {visible_card['suit']}"
    raise ValueError(f"Card is not available in the deck: {card_name}")


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


def build_feature_row(player_cards, dealer_upcard, can_double_down):
    remaining_deck = create_deck()

    for visible_card in [*player_cards, dealer_upcard]:
        remove_visible_card(remaining_deck, visible_card)

    rank_counts = count_remaining_ranks(remaining_deck)
    row = {
        "player_total": hand_value(player_cards),
        "is_soft_hand": int(is_soft_hand(player_cards)),
        "player_card_count": len(player_cards),
        "dealer_upcard_value": min(card_value(dealer_upcard["rank"]), 10),
        "can_double_down": int(can_double_down),
        "remaining_cards": len(remaining_deck),
        **rank_counts,
    }

    return [row[column] for column in FEATURE_COLUMNS], row


def require_inference_dependencies():
    try:
        from joblib import load
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing dependency: joblib. Install ML dependencies with: "
            "py -m pip install -r ml/requirements.txt"
        ) from error

    return {"load": load}


def load_model_bundle(model_path):
    deps = require_inference_dependencies()

    if not model_path.exists():
        raise SystemExit(f"Model artifact not found: {model_path}")

    bundle = deps["load"](model_path)
    model_columns = bundle.get("feature_columns")

    if model_columns != FEATURE_COLUMNS:
        raise SystemExit(
            "Model feature columns do not match this inference script. "
            "Retrain the model with ml/train_model.py."
        )

    return bundle


def print_prediction(bundle, feature_vector, feature_row, label):
    model = bundle["model"]
    prediction = model.predict([feature_vector])[0]

    print("Blackjack policy model prediction")
    print("=================================")
    if label:
        print(f"Scenario: {label}")
    print(f"Player total: {feature_row['player_total']}")
    print(f"Soft hand: {bool(feature_row['is_soft_hand'])}")
    print(f"Dealer upcard value: {feature_row['dealer_upcard_value']}")
    print(f"Can double down: {bool(feature_row['can_double_down'])}")
    print(f"Recommended action: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([feature_vector])[0]
        classes = list(model.classes_)
        print()
        print("Model probabilities")
        print("===================")
        for action, probability in sorted(zip(classes, probabilities), key=lambda item: item[0]):
            print(f"{action}: {probability:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a Blackjack action with the trained supervised ML policy model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ml/models/blackjack_policy_model.joblib"),
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--sample",
        choices=sorted(SAMPLE_HANDS),
        help="Use a preset demo hand.",
    )
    parser.add_argument(
        "--player-cards",
        help="Comma-separated player cards, such as '10S,6H' or 'A,7'.",
    )
    parser.add_argument(
        "--dealer-upcard",
        help="Dealer upcard, such as '10C', '9S', or '6'.",
    )
    parser.add_argument(
        "--no-double",
        action="store_true",
        help="Mark double down as unavailable.",
    )
    return parser.parse_args()


def get_input_state(args):
    if args.sample:
        sample = SAMPLE_HANDS[args.sample]
        return (
            parse_cards(",".join(sample["player_cards"])),
            parse_card(sample["dealer_upcard"]),
            len(sample["player_cards"]) == 2 and not args.no_double,
            sample["label"],
        )

    if not args.player_cards or not args.dealer_upcard:
        raise SystemExit("Provide --sample or both --player-cards and --dealer-upcard.")

    player_cards = parse_cards(args.player_cards)
    return (
        player_cards,
        parse_card(args.dealer_upcard),
        len(player_cards) == 2 and not args.no_double,
        None,
    )


def main():
    args = parse_args()
    player_cards, dealer_upcard, can_double_down, label = get_input_state(args)
    feature_vector, feature_row = build_feature_row(player_cards, dealer_upcard, can_double_down)
    bundle = load_model_bundle(args.model)
    print_prediction(bundle, feature_vector, feature_row, label)


if __name__ == "__main__":
    main()
