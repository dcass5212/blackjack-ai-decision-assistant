# Blackjack AI Feature Specification

## Feature Name
Blackjack AI Decision Assistant

## Why This Matters
As a player, I want to see the estimated odds for hit, stand, and double down so that I can understand which Blackjack action has the best expected outcome.

## Inputs
The AI should use:

- Player hand
- Dealer visible card
- Remaining deck
- Current bet
- Whether double down is legal
- Dealer rules

## Outputs
For each available action, return:

```js
{
  action: "hit",
  simulations: 5000,
  wins: 2120,
  losses: 2450,
  pushes: 430,
  winRate: 0.424,
  lossRate: 0.49,
  pushRate: 0.086,
  expectedValue: -0.066
}
```

The final AI response should also include:

```js
{
  recommendedAction: "hit",
  reason: "Hit has the highest expected value among legal actions.",
  actionResults: []
}
```

## Expected Value Formula
For a normal bet:

```txt
EV = (winRate * profitOnWin) + (pushRate * 0) + (lossRate * lossAmount)
```

For normalized output, use:

```txt
EV = winRate - lossRate
```

For double down, account for the doubled bet:

```txt
EV = 2 * (winRate - lossRate)
```

## Legal Actions
### Hit
Available while the player has not busted.

### Stand
Available while the player has not busted.

### Double Down
Available only when:

- Player has exactly two cards
- Player has enough balance to double the bet

## Dealer Rule
Dealer must hit while below 17.

Initial version assumption:

```txt
Dealer stands on all 17s, including soft 17.
```

## Simulation Behavior
For each action:

1. Clone the current hands and remaining deck.
2. Apply the selected action.
3. Randomly complete the rest of the round.
4. Resolve outcome.
5. Repeat many times.
6. Aggregate outcome probabilities.

## Realistic Dealer Information Mode
For the most realistic AI assistant, the simulation should not use the dealer's hidden card as known information. It should only know:

- Dealer upcard
- Player cards
- Cards already visible
- Remaining unseen cards

The dealer hole card should be randomly sampled during simulation.

## Debug/Internal Mode
For easier early development, the AI may initially use the full dealer hand because the current program already stores it. This is less realistic but easier to implement.

Recommended path:

1. Start with debug/internal mode.
2. Upgrade to realistic mode after the Monte Carlo engine works.

## UI Requirements
Add an AI panel with:

- Action rows
- Win percentage
- Push percentage
- Loss percentage
- Expected value
- Recommended action
- Simulation count

## Acceptance Criteria
- Odds update after the initial deal.
- Odds update after the player hits, if the round continues.
- Double down only appears when legal.
- Recommended action is based on highest expected value.
- AI logic does not mutate the actual game state.
- Results are reasonably stable with at least 5,000 simulations per action.

## Phase 7 ML Extension Specification

The ML extension should train a supervised model to approximate the Monte Carlo assistant's recommended action. The model is not expected to discover perfect Blackjack strategy independently; it learns from labels produced by the simulator.

### Dataset Rows
Each row should represent one Blackjack decision state.

Recommended columns:

```txt
player_total
is_soft_hand
player_card_count
dealer_upcard_value
can_double_down
remaining_cards
count_ace
count_2
count_3
count_4
count_5
count_6
count_7
count_8
count_9
count_10
recommended_action
```

The `recommended_action` label should be one of:

- `hit`
- `stand`
- `doubleDown`

### Model Training
Start with simple supervised classifiers:

- Decision tree
- Random forest

Optional follow-up:

- Gradient boosting

Do not start with a neural network. The goal is a clear, explainable ML workflow.

### Evaluation
Report:

- Train/test split
- Accuracy
- Confusion matrix
- Per-class precision/recall if practical
- Examples where the model disagrees with Monte Carlo

### Acceptance Criteria
- Dataset generation is reproducible.
- Training script can run from a clean checkout after installing dependencies.
- Model predicts legal action labels.
- Evaluation output is saved or printed clearly.
- Documentation explains that the model approximates Monte Carlo recommendations.
