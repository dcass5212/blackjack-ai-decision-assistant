# Blackjack ML Plan

## Goal
Build a scoped supervised-learning extension that helps this project read as an AI/ML portfolio project without becoming over-engineered.

The ML model should learn to approximate the Monte Carlo assistant's recommended action. It is not intended to replace the simulator as a perfect strategy engine.

## Recommended Deliverables
- `ml/generate_dataset.py`
- `ml/train_model.py`
- `ml/data/blackjack_states.csv`
- `ml/models/blackjack_policy_model.joblib`
- Evaluation output showing accuracy and a confusion matrix
- README notes explaining the model, features, and limitations

## Dataset Strategy
Generate random Blackjack decision states, then label each state by running Monte Carlo simulations for legal actions and selecting the highest-EV action.

Each dataset row should include:

- Player hand total
- Whether the player hand is soft
- Player card count
- Dealer upcard value
- Whether double down is legal
- Remaining deck size
- Remaining rank counts
- Monte Carlo-recommended action

Recommended first dataset size:

- Development: 5,000 to 10,000 rows
- Portfolio run: 50,000+ rows if runtime is acceptable

## Feature Set
Start with compact, explainable features:

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
```

Label:

```txt
recommended_action
```

## Models
Start with:

- Decision tree
- Random forest

Optional:

- Gradient boosting

Avoid for this phase:

- Neural networks
- Reinforcement learning
- Complex deployment infrastructure

## Evaluation
Report:

- Train/test split
- Accuracy
- Confusion matrix
- Per-action precision and recall if available
- A short discussion of disagreements between model predictions and Monte Carlo labels

## Success Criteria
- Scripts are easy to run.
- Dataset generation is reproducible.
- Model evaluation is clear enough for a recruiter or hiring manager to understand.
- Documentation is honest that the model imitates the Monte Carlo simulator.
