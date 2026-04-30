import argparse
import csv
from pathlib import Path


FEATURE_COLUMNS = [
    "player_total",
    "is_soft_hand",
    "player_card_count",
    "dealer_upcard_value",
    "can_double_down",
    "remaining_cards",
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
LABEL_COLUMN = "recommended_action"


def load_dataset(path):
    features = []
    labels = []

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            features.append([float(row[column]) for column in FEATURE_COLUMNS])
            labels.append(row[LABEL_COLUMN])

    return features, labels


def require_ml_dependencies():
    try:
        from joblib import dump
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
    except ModuleNotFoundError as error:
        missing = error.name
        raise SystemExit(
            f"Missing dependency: {missing}. Install ML dependencies with: "
            f"py -m pip install -r ml/requirements.txt"
        ) from error

    return {
        "dump": dump,
        "RandomForestClassifier": RandomForestClassifier,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "train_test_split": train_test_split,
        "DecisionTreeClassifier": DecisionTreeClassifier,
    }


def train_and_evaluate(data_path, model_path, test_size, seed):
    deps = require_ml_dependencies()
    features, labels = load_dataset(data_path)

    if not features:
        raise SystemExit(f"No rows found in {data_path}")

    train_test_split = deps["train_test_split"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    models = {
        "decision_tree": deps["DecisionTreeClassifier"](max_depth=8, random_state=seed),
        "random_forest": deps["RandomForestClassifier"](
            n_estimators=100,
            max_depth=12,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = deps["accuracy_score"](y_test, predictions)
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "predictions": predictions,
        }

    best_name = max(results, key=lambda name: results[name]["accuracy"])
    best_result = results[best_name]
    labels_sorted = sorted(set(labels))

    print("Model results")
    print("=============")
    for name, result in results.items():
        print(f"{name}: accuracy={result['accuracy']:.4f}")

    print()
    print(f"Best model: {best_name}")
    print()
    print("Classification report")
    print("=====================")
    print(deps["classification_report"](y_test, best_result["predictions"], labels=labels_sorted))

    print("Confusion matrix")
    print("================")
    print("labels:", labels_sorted)
    print(deps["confusion_matrix"](y_test, best_result["predictions"], labels=labels_sorted))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    deps["dump"](
        {
            "model_name": best_name,
            "model": best_result["model"],
            "feature_columns": FEATURE_COLUMNS,
            "labels": labels_sorted,
            "accuracy": best_result["accuracy"],
        },
        model_path,
    )
    print()
    print(f"Saved best model to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Blackjack policy model from Monte Carlo labels.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ml/data/blackjack_states.csv"),
        help="Input CSV dataset path.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("ml/models/blackjack_policy_model.joblib"),
        help="Output model artifact path.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows used for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_and_evaluate(args.data, args.model_output, args.test_size, args.seed)


if __name__ == "__main__":
    main()
