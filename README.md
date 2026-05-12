# Blackjack AI Decision Assistant

## Live Demo
Try it here: https://dcass5212.github.io/blackjack-ai-decision-assistant/

## Overview
An interactive Blackjack game with a three-layer AI system: a Monte Carlo simulator that estimates odds in real time, a supervised random forest trained to imitate the simulator, and a reinforcement learning agent that learned strategy from game outcomes alone. The live demo shows all three working together; the Python codebase lets you reproduce training, evaluation, and inference from scratch.

The project is a portfolio demonstration of applied AI/ML concepts: Monte Carlo simulation, feature engineering, supervised classification, policy gradient RL, and honest head-to-head evaluation.

## Demo Screenshot
![Blackjack AI decision assistant showing odds and recommendation](docs/Screenshot%202026-05-06%20033756.png)

## Scope
Single-player, single-deck Blackjack. Decisions covered: hit, stand, double down.

Out of scope: splitting pairs, surrender, insurance, side bets, multi-deck shoe behavior.

## AI Architecture

| Layer | Method | Where |
|---|---|---|
| Oracle | Monte Carlo simulation (5,000 trials per action) | `blackjack-ai.js` |
| Imitation | Random forest trained on Monte Carlo labels | `ml/` |
| Experience | MaskablePPO agent trained from game outcomes | `rl/` |

**Monte Carlo** — for each legal action the simulator clones the current hand and remaining deck, randomly completes thousands of rounds, and returns estimated win/push/loss rates and expected value. This runs live in the browser on every deal and hit.

**Supervised ML** — 10,000 labeled game states were generated offline with the Monte Carlo simulator and used to train a random forest classifier. The model learns to predict which action the simulator would recommend given the same visible information: player total, soft-hand flag, dealer upcard, double-down eligibility, and remaining deck composition.

**Reinforcement learning** — a MaskablePPO agent trained for 1 million steps against a Python Gymnasium environment. It received only terminal rewards (+1 win, 0 push, −1 loss, ±2 for doubles) and no labels or strategy charts. The trained policy is exported as a static JSON lookup table and runs in the browser alongside the Monte Carlo panel.

## Evaluation Results

Three agents evaluated over 100,000 hands from the same seeded sequence (single deck, dealer stands on all 17s, no splitting):

| Agent | Win% | Push% | Loss% | EV/hand |
|---|---:|---:|---:|---:|
| Basic strategy | 43.09% | 9.05% | 47.86% | −0.0306 |
| Random forest | 42.50% | 8.61% | 48.89% | −0.0543 |
| RL (MaskablePPO) | 45.16% | 9.02% | 45.82% | −0.0065 |

The RL agent outperforms basic strategy because it has access to remaining deck composition and implicitly learned to exploit it — correct behavior for single-deck play, not overfit. See [`rl/RESULTS.md`](rl/RESULTS.md) for the full analysis and learning curve.

## Key Features
- Interactive Blackjack gameplay with betting and balance tracking
- Live Monte Carlo odds panel (win%, push%, loss%, EV) for every legal action
- RL policy recommendation shown alongside Monte Carlo, with agree/differ indicator
- Three-agent evaluation table displayed below the game
- Demo test hands (Hard 16, Soft 18, Double 11, Stand 20) for instant comparison
- Fully static — no build step, no server; deploys directly to GitHub Pages

## Getting Started

**JavaScript demo** — open `index.html` in a browser. No installation required.

**Python environment** (needed for ML and RL workflows):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
```

## Reproducing the ML Workflow

```powershell
py -m pip install -r ml/requirements.txt
py ml/generate_dataset.py --rows 10000 --simulations 500 --seed 42
py ml/train_model.py
```

Expected output: random forest test accuracy ~0.907, saved to `ml/models/blackjack_policy_model.joblib`.

Run inference on a preset hand:

```powershell
py ml/predict_policy.py --sample hard16
py ml/predict_policy.py --player-cards 10S,6H --dealer-upcard 10C
```

## Reproducing the RL Workflow

```powershell
py -m pip install -r rl/requirements.txt
py rl/train.py --timesteps 1000000 --seed 42   # ~15 min on CPU
py rl/evaluate.py --hands 100000 --seed 42     # reproduces the table above
py rl/export_policy.py                         # regenerates rl_policy_lookup.js
```

The trained policy is committed at `rl/models/policy.zip`. Running `rl/evaluate.py` without retraining reproduces the reported numbers exactly.

## Tests

```powershell
npm test
```

Covers hand scoring, blackjack and bust detection, legal actions, dealer draw behavior, Monte Carlo result shape, input-state immutability, and ML feature-column consistency.

## Project Docs
- `docs/ROADMAP.md` — phase-by-phase build history
- `docs/AI_SPEC.md` — Monte Carlo assistant feature specification
- `docs/ML_PLAN.md` — supervised learning design notes
- `rl/RESULTS.md` — evaluation write-up with learning curve and analysis

## Limitations
- The Monte Carlo panel varies slightly between runs because the dealer's hidden card is sampled from unseen cards during simulation.
- After a simulated hit, the player follows a fixed threshold policy (draw until 17+) rather than recursively consulting the oracle.
- The supervised model imitates Monte Carlo labels, not perfect basic strategy; its accuracy is bounded by the quality of those labels.
- The RL policy lookup uses a representative fresh-deck composition for all states; decisions that depend heavily on specific remaining card counts may differ from the full neural network policy.
- Splitting, surrender, insurance, and multi-deck shoe behavior are out of scope for all three agents.

## Technologies
- HTML, CSS, JavaScript
- Monte Carlo simulation
- Python, scikit-learn (supervised learning)
- PyTorch, Stable-Baselines3, Gymnasium (reinforcement learning)
