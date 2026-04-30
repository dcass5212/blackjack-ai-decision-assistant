# Blackjack AI Project Context

## Project Goal
Add an AI/ML-style decision assistant to an existing browser-based Blackjack game. The assistant should estimate the probability of winning, losing, or pushing for available actions:

- Hit
- Stand
- Double Down

The feature should be portfolio-ready for AI/ML roles by showing simulation, probability modeling, clean software design, and explainable recommendations.

## Current App Summary
The current project is a single-page HTML/CSS/JavaScript Blackjack game with:

- Standard 52-card deck generation and shuffling
- Player and dealer hands
- Betting system
- Balance tracking
- Win/loss tracking
- Dealer behavior: dealer hits while score is below 17
- Player actions: hit, stand, double down
- Round outcomes: blackjack, win, lose, push, bust, dealer bust

## Recommended AI Feature
Use Monte Carlo simulation first. This is the best starting point because it is realistic, explainable, portfolio-friendly, and does not require a large training dataset.

For each possible action, simulate thousands of possible round outcomes from the current game state using the remaining deck. Then calculate:

- Win probability
- Loss probability
- Push probability
- Expected value
- Recommended action

## Why Monte Carlo First?
Monte Carlo simulation is a strong portfolio choice because it demonstrates:

- Probability modeling
- Game-state simulation
- Decision-making under uncertainty
- Expected value calculation
- Clean separation between game UI and AI logic
- Explainable AI output

Later, the project can be expanded into true machine learning by generating simulated data and training a supervised model to imitate the Monte Carlo decision policy. This should stay intentionally scoped for an entry-level AI/ML portfolio: dataset generation, feature engineering, model training, evaluation, and clear discussion of limitations.

## Portfolio Positioning
Suggested portfolio title:

**Blackjack AI Decision Assistant**

Suggested resume bullet:

Built a Blackjack AI decision assistant using Monte Carlo simulation to estimate win probability and expected value for hit, stand, and double-down actions based on live game state and remaining deck composition.

## Core Technical Goals
1. Separate Blackjack game logic from UI code.
2. Create reusable game-state functions.
3. Add a Monte Carlo simulation engine.
4. Display odds for each legal action.
5. Recommend the best action based on expected value.
6. Add documentation explaining the AI approach.
7. Generate a dataset for supervised learning.
8. Train and evaluate a simple model that predicts the Monte Carlo-recommended action.

## Constraints and Assumptions
- The game uses one standard 52-card deck.
- Dealer hits until reaching 17 or higher.
- Dealer's hidden card is known internally to the program, but the AI assistant should support either:
  - Realistic mode: only use the visible dealer upcard.
  - Internal/debug mode: use the full dealer hand.
- Double down should only be shown when legal.
- Insurance, surrender, splitting, and side bets are out of scope for the first version.

## Recommended First Version
Version 1 should implement:

- `blackjack-ai.js`
- A function called `getActionOdds(gameState)`
- A UI panel showing odds for hit, stand, and double down
- A highlighted recommended action

## Recommended ML Extension
Phase 7 should not attempt a full reinforcement-learning agent or production-grade Blackjack solver. The best portfolio version is a compact supervised-learning workflow:

1. Generate many random Blackjack states.
2. Label each state using the existing Monte Carlo engine's best action.
3. Convert each state into numeric ML features.
4. Train simple, explainable classifiers such as a decision tree and random forest.
5. Compare model predictions against Monte Carlo labels using accuracy and a confusion matrix.
6. Document that the model approximates the simulator, not perfect Blackjack strategy.

Recommended deliverables:

- `ml/generate_dataset.py`
- `ml/train_model.py`
- `ml/data/blackjack_states.csv`
- `ml/models/blackjack_policy_model.joblib`
- Evaluation output with accuracy, confusion matrix, and notes on common mistakes.
