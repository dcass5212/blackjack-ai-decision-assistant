# Blackjack AI Decision Assistant

## Live Demo
Try it here: https://dcass5212.github.io/blackjack-ai-decision-assistant/

## Overview
This project is a Blackjack decision assistant wrapped in an interactive browser demo. The assistant estimates the probability of winning, losing, or pushing for each available action: hit, stand, and double down.

The goal is to demonstrate applied AI concepts in a practical browser-based project, including probability simulation, expected value calculation, decision support, and explainable recommendations.

## Demo Screenshot
![Blackjack AI decision assistant showing odds and recommendation](docs/Screenshot%202026-05-06%20033756.png)

## Scope
This is a scoped decision-support project, not a full casino Blackjack simulator. It focuses on core single-hand decisions and keeps advanced table rules out of scope so the AI/ML workflow stays clear and reviewable.

Included:
- Hit, stand, and double down decisions
- Single-player browser gameplay
- Single-deck remaining-card simulation
- Dealer upcard-based Monte Carlo advice
- Offline supervised-learning workflow that imitates Monte Carlo labels

Out of scope:
- Splitting pairs
- Surrender
- Insurance
- Side bets
- Multi-deck shoe behavior
- Casino-grade basic strategy validation

## AI Approach
The first version uses Monte Carlo simulation. For each possible action, the program simulates thousands of possible outcomes from the current hand and remaining deck. It then estimates the probability of each outcome and recommends the action with the highest expected value.

## Explainability
The assistant displays the simulated win, push, and loss rates for each legal action, along with a normalized expected value.

Expected value is calculated as:

```txt
EV = winRate - lossRate
```

Double down uses the same idea but doubles the EV impact because the bet is doubled.

The live assistant uses realistic visible information: it starts from the dealer upcard and samples the hidden dealer card from the remaining unseen cards during simulation.

## Key Features
- Interactive Blackjack gameplay
- Betting and balance system
- AI-generated odds for hit, stand, and double down
- Monte Carlo simulation engine
- Expected value calculation
- Explainable action recommendation
- Demo test hands for comparing interesting decisions
- Future-ready structure for supervised ML expansion

## Demo Test Hands
The game includes preset hands that can be loaded from the betting screen. These are useful for portfolio demos because they immediately show how the assistant compares actions in different situations:

- Hard 16 vs dealer 10
- Soft 18 vs dealer 9
- 11 vs dealer 6
- 20 vs dealer 10

## Deployment
The app is static HTML, CSS, and JavaScript, so no build step or local server is required.

Run locally:

```txt
Open index.html in a browser.
```

Deploy with GitHub Pages:
1. Push the repository to GitHub.
2. Open the repository settings.
3. Go to Pages.
4. Set the source to deploy from the default branch and repository root.
5. Use the generated GitHub Pages URL as the live demo link.

`index.html` is the live demo entry point. `blackjackAI.html` is kept as a redirect for older links.

Verify JavaScript behavior:

```powershell
npm test
```

Reproduce the ML workflow:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r ml/requirements.txt
py ml/generate_dataset.py --rows 10000 --simulations 500 --seed 42
py ml/train_model.py
```

Run the trained model on a preset demo hand:

```powershell
py ml/predict_policy.py --sample hard16
```

Run the trained model on custom visible cards:

```powershell
py ml/predict_policy.py --player-cards 10S,6H --dealer-upcard 10C
```

## Project Docs
- `docs/ROADMAP.md` tracks portfolio readiness and future work.
- `docs/AI_SPEC.md` describes the AI assistant design.
- `docs/ML_PLAN.md` outlines the supervised-learning extension.
- `docs/Screenshot 2026-05-06 033756.png` shows the browser demo with the AI odds panel.

## Supervised ML Expansion
The project now includes a supervised-learning workflow that trains a model to approximate the Monte Carlo assistant's recommended action. The ML model is not meant to replace the simulator as a perfect strategy engine; it learns from Monte Carlo-generated labels and gives a clear portfolio example of dataset generation, feature engineering, model training, and evaluation.

Implemented workflow:

- Generate simulated Blackjack decision states with `ml/generate_dataset.py`
- Label each state with the Monte Carlo-recommended action: `hit`, `stand`, or `doubleDown`
- Engineer features such as player total, soft hand, card count, dealer upcard, double-down eligibility, remaining deck size, and remaining rank counts
- Train both a decision tree and random forest classifier with `ml/train_model.py`
- Evaluate models with accuracy, classification report, and confusion matrix
- Save the best model as a reusable `.joblib` artifact
- Run the saved model against demo or custom hands with `ml/predict_policy.py`

This is intentionally scoped as supervised learning, not reinforcement learning or deep learning.

Phase 7 reproducibility:

Use Python 3.10 or newer. On Windows, the commands below use the Python launcher `py`; if your machine does not have it, replace `py` with `python`.

Create and activate a local virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r ml/requirements.txt
```

Recreate the committed 10,000-row dataset and train the model:

```powershell
py ml/generate_dataset.py --rows 10000 --simulations 500 --seed 42
py ml/train_model.py
```

Expected output range:
- The committed dataset contains 10,000 rows.
- With the default train/test split and seed, the random forest should be the best model.
- Expected random forest test accuracy is about 0.90 to 0.91.
- The saved model is written to `ml/models/blackjack_policy_model.joblib`.

## Tests
The JavaScript game and AI logic include a dependency-free Node test suite:

```powershell
npm test
```

Current coverage includes hand scoring, blackjack and bust detection, legal actions, dealer draw behavior, realistic dealer upcard simulation, Monte Carlo result shape, input-state immutability, and ML feature-column consistency.

Current results:

| Dataset | Simulations per action | Best model | Test accuracy |
|---|---:|---|---:|
| 1,000 rows | 500 | Random forest | 0.8800 |
| 10,000 rows | 500 | Random forest | 0.9070 |

For the 10,000-row run, the random forest outperformed the decision tree:

- Decision tree accuracy: 0.8345
- Random forest accuracy: 0.9070
- Test split size: 2,000 rows
- Strongest class: `stand` with 0.95 recall
- Most common confusion: `hit` states predicted as `stand`

The saved model artifact is:

```txt
ml/models/blackjack_policy_model.joblib
```

Inference demo:

```powershell
py ml/predict_policy.py --sample hard16
py ml/predict_policy.py --sample soft18
py ml/predict_policy.py --sample double11
py ml/predict_policy.py --sample stand20
```

## Limitations
- The live Monte Carlo assistant estimates outcomes from the visible dealer upcard, so recommendations vary slightly between runs because the hidden card is sampled during simulation.
- After simulating an initial `hit`, the simulated player follows a simple policy of drawing until 17 or higher.
- The supervised ML model learns to imitate Monte Carlo labels; it is not trained against perfect Blackjack strategy or casino-grade basic strategy tables.
- Splitting, surrender, insurance, side bets, and multi-deck shoe behavior are intentionally out of scope.

## Reinforcement Learning Extension

The project now includes a reinforcement learning agent trained from scratch using PPO, compared head-to-head against the Monte Carlo simulator labels and the supervised random forest. The three-stage narrative is:

1. **Monte Carlo** (oracle) — simulates thousands of futures per decision
2. **Supervised ML** (imitation) — random forest trained on Monte Carlo labels
3. **RL agent** (experience) — learns from game outcomes alone, no labels

See [`rl/RESULTS.md`](rl/RESULTS.md) for the full comparison table, learning curve, and analysis.

Quick summary (100,000 hands, same seeded sequence):

| Agent | Win% | Push% | Loss% | EV/hand |
|---|---:|---:|---:|---:|
| Basic strategy | 43.09% | 9.05% | 47.86% | −0.0306 |
| Random forest | 42.50% | 8.61% | 48.89% | −0.0543 |
| RL (MaskablePPO) | 45.16% | 9.02% | 45.82% | −0.0065 |

*(Run `python rl/evaluate.py` to reproduce these numbers from the saved policy.)*

Train the RL agent from scratch:

```powershell
py -m pip install -r rl/requirements.txt
py rl/train.py                   # ~15 min on CPU, saves rl/models/policy.zip
py rl/evaluate.py                # compare all three agents, saves rl/logs/evaluation_results.csv
```

## Limitations
- The live Monte Carlo assistant estimates outcomes from the visible dealer upcard, so recommendations vary slightly between runs because the hidden card is sampled during simulation.
- After simulating an initial `hit`, the simulated player follows a simple policy of drawing until 17 or higher.
- The supervised ML model learns to imitate Monte Carlo labels; it is not trained against perfect Blackjack strategy or casino-grade basic strategy tables.
- Splitting, surrender, insurance, side bets, and multi-deck shoe behavior are intentionally out of scope.

## Technologies
- HTML
- CSS
- JavaScript
- Monte Carlo simulation
- Probability modeling
- Python
- scikit-learn
- Supervised learning
- PyTorch
- Stable-Baselines3 (MaskablePPO)
- Gymnasium
