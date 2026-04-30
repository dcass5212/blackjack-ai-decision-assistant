# Blackjack AI Decision Assistant

## Overview
This project is an interactive Blackjack game enhanced with an AI decision assistant. The assistant estimates the probability of winning, losing, or pushing for each available action: hit, stand, and double down.

The goal is to demonstrate applied AI concepts in a practical browser-based project, including probability simulation, expected value calculation, decision support, and explainable recommendations.

## AI Approach
The first version uses Monte Carlo simulation. For each possible action, the program simulates thousands of possible outcomes from the current hand and remaining deck. It then estimates the probability of each outcome and recommends the action with the highest expected value.

## Explainability
The assistant displays the simulated win, push, and loss rates for each legal action, along with a normalized expected value.

Expected value is calculated as:

```txt
EV = winRate - lossRate
```

Double down uses the same idea but doubles the EV impact because the bet is doubled.

The current implementation uses internal/debug mode for simulation, which means it uses the full dealer hand already stored by the game. A later version can switch to a more realistic mode that only uses the dealer upcard and samples the hidden card during simulation.

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

## Supervised ML Expansion
The project now includes a supervised-learning workflow that trains a model to approximate the Monte Carlo assistant's recommended action. The ML model is not meant to replace the simulator as a perfect strategy engine; it learns from Monte Carlo-generated labels and gives a clear portfolio example of dataset generation, feature engineering, model training, and evaluation.

Implemented workflow:

- Generate simulated Blackjack decision states with `ml/generate_dataset.py`
- Label each state with the Monte Carlo-recommended action: `hit`, `stand`, or `doubleDown`
- Engineer features such as player total, soft hand, card count, dealer upcard, double-down eligibility, remaining deck size, and remaining rank counts
- Train both a decision tree and random forest classifier with `ml/train_model.py`
- Evaluate models with accuracy, classification report, and confusion matrix
- Save the best model as a reusable `.joblib` artifact

This is intentionally scoped as supervised learning, not reinforcement learning or deep learning.

Phase 7 scripts:

```powershell
py ml/generate_dataset.py --rows 1000 --simulations 500
py -m pip install -r ml/requirements.txt
py ml/train_model.py
```

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

## Limitations
- The live Monte Carlo assistant currently uses internal/debug mode, so simulations can access the dealer's full hand instead of only the visible upcard.
- After simulating an initial `hit`, the simulated player follows a simple policy of drawing until 17 or higher.
- The supervised ML model learns to imitate Monte Carlo labels; it is not trained against perfect Blackjack strategy or casino-grade basic strategy tables.
- Insurance, surrender, splitting, side bets, and multi-deck shoe behavior are out of scope.

## Technologies
- HTML
- CSS
- JavaScript
- Monte Carlo simulation
- Probability modeling
- Python
- scikit-learn
- Supervised learning