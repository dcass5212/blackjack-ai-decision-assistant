# Blackjack AI Roadmap

## Phase 1: Clean Up the Existing Game
Goal: Make the current Blackjack game easier to extend.

Tasks:
- Move JavaScript out of the HTML file into `blackjack.js`.
- Keep styling in either the same HTML file for now or move it to `styles.css`.
- Create clear functions for scoring, deck creation, dealer behavior, and round resolution.
- Avoid duplicating game logic inside the AI code.

Deliverable:
- The original game still works exactly as before, but the logic is easier to reuse.

## Phase 2: Define Game State for AI
Goal: Give the AI a clean snapshot of the current hand.

Create a game-state object like:

```js
const gameState = {
  playerHand,
  dealerUpCard,
  dealerHand,
  remainingDeck,
  currentBet: bet,
  balance,
  canDoubleDown: balance >= bet && playerHand.length === 2
};
```

Deliverable:
- A reusable function such as `getCurrentGameState()`.

## Phase 3: Monte Carlo Simulation Engine
Goal: Estimate outcomes for hit, stand, and double down.

Tasks:
- Clone the current game state.
- Randomly complete the rest of the round many times.
- Count wins, losses, pushes, and busts.
- Calculate probabilities and expected value.

Suggested number of simulations:
- Development: 1,000 per action
- Portfolio/demo mode: 5,000 to 10,000 per action

Deliverable:
- `blackjack-ai.js` with functions that return odds for each action.

## Phase 4: AI Odds UI
Goal: Show the AI assistant in the game interface.

Display something like:

| Action | Win | Push | Lose | EV |
|---|---:|---:|---:|---:|
| Hit | 42.1% | 8.3% | 49.6% | -0.075 |
| Stand | 38.4% | 9.1% | 52.5% | -0.141 |
| Double | 45.0% | 7.0% | 48.0% | -0.060 |

Also show:

> Recommended action: Hit

Deliverable:
- A visible AI panel under the player controls.

## Phase 5: Explainability
Goal: Make the feature understandable to recruiters and hiring managers.

Add explanations such as:
- Why the recommendation was chosen
- How many simulations were run
- Whether the AI used full deck tracking or only visible cards
- How expected value was calculated

Deliverable:
- A small explanation panel and README section.

## Phase 6: Portfolio Polish
Goal: Make the project impressive and easy to review.

Tasks:
- Add screenshots or a short demo GIF.
- Add a README with project overview, AI method, and future ML expansion.
- Add clean comments to important AI functions.
- Add test hands that demonstrate interesting decisions.

Deliverable:
- GitHub-ready project.

## Phase 7: Supervised ML Policy Model
Goal: Turn the simulation project into a stronger entry-level AI/ML portfolio piece without overbuilding it.

Approach:
1. Generate a dataset of simulated Blackjack game states.
2. Label each state with the best Monte Carlo action.
3. Engineer compact features such as player total, soft hand, dealer upcard, can double, and remaining deck composition.
4. Train supervised classifiers to predict the Monte Carlo action.
5. Compare model decisions against Monte Carlo labels.
6. Summarize accuracy, confusion matrix, and model limitations.

Possible models:
- Decision tree
- Random forest
- Gradient boosting

Avoid for this phase:
- Deep learning
- Reinforcement learning
- Full casino simulation
- Complex deployment infrastructure

Deliverables:
- `ml/generate_dataset.py`
- `ml/train_model.py`
- Generated CSV dataset
- Saved model artifact
- Evaluation summary with accuracy and confusion matrix
- README section explaining that the model approximates Monte Carlo labels

## Phase 8: Reinforcement Learning Agent

Goal: Train a policy from scratch using RL, then compare it head-to-head against the Monte Carlo simulator and the supervised ML model. The narrative arc is: Monte Carlo (oracle) → supervised imitation (learns from the oracle) → RL (learns from experience alone).

Design decisions (confirmed before implementation):

- **State vector:** 16 features, identical to the supervised model — player_total, is_soft_hand, player_card_count, dealer_upcard_value, can_double_down, remaining_cards, count_ace through count_10. Full feature parity eliminates feature differences as a confound in the comparison.
- **Action space:** 3 discrete actions (hit=0, stand=1, doubleDown=2). Illegal actions handled by masking (not penalty) — doubleDown is masked out on 3-card hands.
- **Reward:** +1 win, 0 push, −1 loss, ±2 on doubles. No intermediate rewards or shaping.
- **Algorithm:** MaskablePPO (stable-baselines3 + sb3-contrib). PPO chosen over DQN for stability under high per-hand variance and clean action-mask integration. Convergence expected in 500k–2M timesteps on CPU.
- **Environment:** Python port of the game (reusing logic from generate_dataset.py), wrapped in a Gymnasium interface. No JS bridge.
- **Evaluation:** 100,000 hands, same seeded sequence for all three agents. SE on EV ≈ ±0.31%, enough to distinguish correct convergence from a poor policy.

Deliverables:
- `rl/env.py` — Gymnasium-compatible BlackjackEnv
- `rl/train.py` — MaskablePPO training script, logs episode return and win/push/loss rates to CSV
- `rl/basic_strategy.py` — hard-coded single-deck basic strategy chart
- `rl/evaluate.py` — evaluation harness comparing RL, random forest, and basic strategy
- `rl/requirements.txt` — pinned RL dependencies (separate from ml/requirements.txt)
- `rl/RESULTS.md` — comparison table, learning curve, and honest analysis
- Updated top-level README section pointing to the RL work