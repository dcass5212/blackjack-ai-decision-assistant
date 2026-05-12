"""
Train a MaskablePPO agent on the BlackjackEnv.

Usage (from repo root):
    python rl/train.py                        # default 1M steps (~15 min CPU)
    python rl/train.py --timesteps 500000     # ~7-8 min, usually sufficient
    python rl/train.py --timesteps 2000000    # ~30 min, thorough

Outputs:
    rl/models/policy.zip          -- saved SB3 policy (load with MaskablePPO.load)
    rl/logs/training_log.csv      -- per-rollout metrics

Wall-clock note: SB3's gym wrapper overhead limits this env to ~1100 FPS on CPU.
Convergence on blackjack typically happens well before 500k steps; the default
1M gives a clear flat tail in the learning curve.
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

# Allow running from repo root (python rl/train.py) or from rl/ directory
sys.path.insert(0, str(Path(__file__).parent))

# Suppress gymnasium deprecation warning about action_masks wrapper access;
# sb3_contrib handles this correctly, the warning is cosmetic.
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from env import BlackjackEnv

LOG_DIR = Path("rl/logs")
MODEL_DIR = Path("rl/models")
LOG_CSV = LOG_DIR / "training_log.csv"
MODEL_PATH = MODEL_DIR / "policy"


class TrainingLogger(BaseCallback):
    """
    Log episode statistics to CSV after each rollout buffer collection.
    Captures mean episode return, win/push/loss rates, and policy entropy.
    """

    def __init__(self, log_path: Path, eval_env: BlackjackEnv, eval_freq: int = 10_000):
        super().__init__()
        self._log_path = log_path
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._last_eval_step = 0
        self._writer = None
        self._file = None

    def _on_training_start(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._log_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=["timestep", "mean_return", "win_rate", "push_rate", "loss_rate", "entropy"],
        )
        self._writer.writeheader()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self._eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self):
        n_eval = 1000
        returns, wins, pushes, losses = [], 0, 0, 0

        obs, _ = self._eval_env.reset()
        ep_return = 0.0

        for _ in range(n_eval):
            masks = self._eval_env.action_masks()
            action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, _ = self._eval_env.step(int(action))
            ep_return += reward

            if terminated or truncated:
                returns.append(ep_return)
                if ep_return > 0:
                    wins += 1
                elif ep_return == 0:
                    pushes += 1
                else:
                    losses += 1
                ep_return = 0.0
                obs, _ = self._eval_env.reset()

        n_eps = len(returns) or 1
        mean_return = float(np.mean(returns)) if returns else 0.0

        # Approximate entropy from the rollout buffer's logged values
        entropy = float(self.model.logger.name_to_value.get("train/entropy_loss", 0.0))

        self._writer.writerow(
            {
                "timestep": self.num_timesteps,
                "mean_return": round(mean_return, 4),
                "win_rate": round(wins / n_eps, 4),
                "push_rate": round(pushes / n_eps, 4),
                "loss_rate": round(losses / n_eps, 4),
                "entropy": round(entropy, 4),
            }
        )
        self._file.flush()

        print(
            f"  step={self.num_timesteps:>8,}  "
            f"return={mean_return:+.4f}  "
            f"win={wins/n_eps:.3f}  push={pushes/n_eps:.3f}  loss={losses/n_eps:.3f}"
        )

    def _on_training_end(self):
        if self._file:
            self._file.close()


def train(timesteps: int, seed: int):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training MaskablePPO for {timesteps:,} timesteps (seed={seed})")
    print("=" * 60)

    train_env = BlackjackEnv(seed=seed)
    eval_env = BlackjackEnv(seed=seed + 1)

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=1.0,          # no discounting — each hand is independent
        ent_coef=0.01,      # entropy bonus prevents premature policy collapse
        clip_range=0.2,
        verbose=0,
        seed=seed,
        policy_kwargs={"net_arch": [64, 64]},
    )

    logger_callback = TrainingLogger(LOG_CSV, eval_env, eval_freq=20_000)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=logger_callback)
    elapsed = time.time() - t0

    model.save(str(MODEL_PATH))
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Policy saved to {MODEL_PATH}.zip")
    print(f"Training log saved to {LOG_CSV}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskablePPO Blackjack agent.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.timesteps, args.seed)
