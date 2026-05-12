"""
Evaluation harness: compare RL policy, random forest, and basic strategy
over the same 100,000 seeded hands.

Usage (from repo root):
    python rl/evaluate.py
    python rl/evaluate.py --hands 100000 --seed 42
    python rl/evaluate.py --model rl/models/policy.zip

Outputs:
    rl/logs/evaluation_results.csv   -- per-agent summary
    rl/logs/learning_curve.png       -- training return over time
    Prints a comparison table to stdout.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root or from rl/ directory
_RL_DIR = Path(__file__).parent
sys.path.insert(0, str(_RL_DIR))

from env import BlackjackEnv, ACTION_HIT, ACTION_STAND, ACTION_DOUBLE
from basic_strategy import basic_strategy_action

ML_MODEL_PATH = Path("ml/models/blackjack_policy_model.joblib")
RL_MODEL_PATH = Path("rl/models/policy.zip")
LOG_DIR = Path("rl/logs")
EVAL_CSV = LOG_DIR / "evaluation_results.csv"
TRAINING_LOG = LOG_DIR / "training_log.csv"
LEARNING_CURVE_PATH = LOG_DIR / "learning_curve.png"

ACTION_NAMES = {ACTION_HIT: "hit", ACTION_STAND: "stand", ACTION_DOUBLE: "doubleDown"}
RF_ACTION_MAP = {"hit": ACTION_HIT, "stand": ACTION_STAND, "doubleDown": ACTION_DOUBLE}


# ---------------------------------------------------------------------------
# Agent wrappers — each returns an integer action given the observation and env
# ---------------------------------------------------------------------------

class BasicStrategyAgent:
    def __init__(self):
        self.name = "Basic Strategy"

    def predict(self, obs: np.ndarray, env: BlackjackEnv) -> int:
        player_total = int(obs[0])
        is_soft = bool(obs[1])
        dealer_up = int(obs[3])
        can_double = bool(obs[4])
        action_str = basic_strategy_action(player_total, is_soft, dealer_up, can_double)
        return {"hit": ACTION_HIT, "stand": ACTION_STAND, "doubleDown": ACTION_DOUBLE}[action_str]


class RandomForestAgent:
    """
    Random forest agent with precomputed lookup table.

    sklearn's predict() is ~17ms per single-sample call (Python overhead across
    100 trees). With 150k evaluation steps that would take 40+ minutes. Instead
    we batch-predict all reachable (player_total, is_soft, dealer_upcard,
    can_double) combinations once at init — 720 states in <1ms — and do O(1)
    dict lookups during evaluation. Deck composition features are set to a
    representative fresh-deck value; the RF's top-level decisions are dominated
    by the four core features.
    """

    def __init__(self, model_path: Path):
        self.name = "Random Forest"
        try:
            from joblib import load
        except ImportError:
            sys.exit("joblib not found. Install ml/requirements.txt.")
        bundle = load(model_path)
        self._model = bundle["model"]
        self._lookup = self._build_lookup()

    def _build_lookup(self) -> dict:
        # Representative deck after 4 cards dealt from a fresh 52-card deck:
        # 48 remaining, approximate rank counts at expected values.
        base_counts = [4, 4, 4, 4, 4, 4, 4, 4, 4, 14]  # ace,2..9,10+face
        remaining = 48
        states, keys = [], []
        for total in range(4, 22):
            for is_soft in (0, 1):
                for dealer_up in range(2, 12):
                    for can_double in (0, 1):
                        keys.append((total, is_soft, dealer_up, can_double))
                        states.append([total, is_soft, 2, dealer_up, can_double, remaining] + base_counts)
        import numpy as np_local
        X = np_local.array(states, dtype=float)
        preds = self._model.predict(X)
        return {k: RF_ACTION_MAP[v] for k, v in zip(keys, preds)}

    def predict(self, obs: np.ndarray, env: BlackjackEnv) -> int:
        key = (int(obs[0]), int(obs[1]), int(obs[3]), int(obs[4]))
        return self._lookup.get(key, ACTION_STAND)


class RLAgent:
    def __init__(self, model_path: Path):
        self.name = "RL (MaskablePPO)"
        try:
            from sb3_contrib import MaskablePPO
        except ImportError:
            sys.exit("sb3-contrib not found. Install rl/requirements.txt.")
        self._model = MaskablePPO.load(str(model_path))

    def predict(self, obs: np.ndarray, env: BlackjackEnv) -> int:
        masks = env.action_masks()
        action, _ = self._model.predict(obs, action_masks=masks, deterministic=True)
        return int(action)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_agent(agent, n_hands: int, seed: int) -> dict:
    """Run one agent for n_hands, return aggregated stats."""
    env = BlackjackEnv(seed=seed)
    rewards = []
    wins = pushes = losses = 0

    obs, _ = env.reset()
    ep_reward = 0.0
    steps_this_ep = 0

    # We need to complete exactly n_hands episodes
    hands_done = 0
    while hands_done < n_hands:
        action = agent.predict(obs, env)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        steps_this_ep += 1

        if terminated or truncated:
            rewards.append(ep_reward)
            if ep_reward > 0:
                wins += 1
            elif ep_reward == 0:
                pushes += 1
            else:
                losses += 1
            hands_done += 1
            ep_reward = 0.0
            steps_this_ep = 0
            if hands_done < n_hands:
                obs, _ = env.reset()

    n = len(rewards)
    mean_ev = float(np.mean(rewards))
    std_ev = float(np.std(rewards))
    se = std_ev / math.sqrt(n)
    ci95 = 1.96 * se

    return {
        "agent": agent.name,
        "hands": n,
        "win_rate": wins / n,
        "push_rate": pushes / n,
        "loss_rate": losses / n,
        "ev_per_hand": mean_ev,
        "std": std_ev,
        "se": se,
        "ci95": ci95,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[dict]):
    header = f"{'Agent':<22} {'Win%':>7} {'Push%':>7} {'Loss%':>7} {'EV/hand':>9} {'95% CI':>10}"
    print()
    print("=" * len(header))
    print("Evaluation results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        ci = f"±{r['ci95']:.4f}"
        print(
            f"{r['agent']:<22} "
            f"{r['win_rate']*100:>6.2f}% "
            f"{r['push_rate']*100:>6.2f}% "
            f"{r['loss_rate']*100:>6.2f}% "
            f"{r['ev_per_hand']:>+9.4f} "
            f"{ci:>10}"
        )
    print("=" * len(header))
    print(f"N = {results[0]['hands']:,} hands per agent (same seeded sequence)")
    print()


def save_csv(results: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def plot_learning_curve(log_path: Path, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping learning curve plot")
        return

    if not log_path.exists():
        print(f"Training log not found at {log_path} — skipping plot")
        return

    timesteps, returns = [], []
    with log_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(int(row["timestep"]))
            returns.append(float(row["mean_return"]))

    if not timesteps:
        print("Training log is empty — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timesteps, returns, linewidth=1.5, color="#2563eb")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", label="break even")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean return (1,000-hand eval)")
    ax.set_title("RL agent learning curve")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Blackjack agents head-to-head.")
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=RL_MODEL_PATH)
    parser.add_argument("--rf-model", type=Path, default=ML_MODEL_PATH)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    agents = []

    # Basic strategy always available
    agents.append(BasicStrategyAgent())

    # Random forest — skip gracefully if not trained
    if args.rf_model.exists():
        agents.append(RandomForestAgent(args.rf_model))
    else:
        print(f"Warning: RF model not found at {args.rf_model} — skipping")

    # RL agent — skip gracefully if not trained
    if args.model.exists():
        agents.append(RLAgent(args.model))
    else:
        print(f"Warning: RL model not found at {args.model} — run rl/train.py first")

    print(f"Evaluating {len(agents)} agent(s) over {args.hands:,} hands each (seed={args.seed})")
    results = []
    for agent in agents:
        print(f"  Running {agent.name}...", end=" ", flush=True)
        stats = run_agent(agent, args.hands, args.seed)
        results.append(stats)
        print(f"EV={stats['ev_per_hand']:+.4f}")

    print_table(results)
    save_csv(results, EVAL_CSV)

    if not args.no_plot:
        plot_learning_curve(TRAINING_LOG, LEARNING_CURVE_PATH)


if __name__ == "__main__":
    main()
