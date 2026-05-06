# Blackjack ML Workflow

This folder contains the Phase 7 supervised-learning extension.

The model learns to approximate the Monte Carlo assistant's recommended action. It is trained from simulator-generated labels, so it should be described as an imitation model rather than a perfect Blackjack strategy engine.

## Environment

Use Python 3.10 or newer. On Windows, these commands use the Python launcher `py`; if your machine does not have it, replace `py` with `python`.

Create a local virtual environment from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r ml/requirements.txt
```

## Reproduce Current Results

The committed dataset contains 10,000 rows and was generated with 500 Monte Carlo simulations per legal action.

From the project root:

```powershell
py ml/generate_dataset.py --rows 10000 --simulations 500 --seed 42
py ml/train_model.py
```

Expected result range:
- Best model: random forest
- Decision tree accuracy: about 0.83
- Random forest accuracy: about 0.90 to 0.91
- Test split size: 2,000 rows
- Saved artifact: `ml/models/blackjack_policy_model.joblib`

The current committed evaluation report recorded:
- Decision tree accuracy: 0.8345
- Random forest accuracy: 0.9070

## Run Inference

Use the saved model artifact to predict an action for a demo hand:

```powershell
py ml/predict_policy.py --sample hard16
```

Available samples:
- `hard16`
- `soft18`
- `double11`
- `stand20`

Predict from custom visible cards:

```powershell
py ml/predict_policy.py --player-cards 10S,6H --dealer-upcard 10C
```

Cards may include suits, such as `10S`, `AH`, and `7C`. If a suit is omitted, the script removes the first available card of that rank from the deck when building the feature vector.

## Generate A Different Dataset

Small development run:

```powershell
py ml/generate_dataset.py --rows 1000 --simulations 500
```

Larger portfolio run:

```powershell
py ml/generate_dataset.py --rows 50000 --simulations 1000
```

## Train Models

Train and evaluate:

```powershell
py ml/train_model.py
```

The training script fits a decision tree and random forest, reports accuracy and a confusion matrix, then saves the better model to:

```txt
ml/models/blackjack_policy_model.joblib
```
