# Blackjack ML Workflow

This folder contains the Phase 7 supervised-learning extension.

The model learns to approximate the Monte Carlo assistant's recommended action. It is trained from simulator-generated labels, so it should be described as an imitation model rather than a perfect Blackjack strategy engine.

## Generate A Dataset

Small development run:

```powershell
py ml/generate_dataset.py --rows 1000 --simulations 500
```

Larger portfolio run:

```powershell
py ml/generate_dataset.py --rows 50000 --simulations 1000
```

## Train Models

Install dependencies:

```powershell
py -m pip install -r ml/requirements.txt
```

Train and evaluate:

```powershell
py ml/train_model.py
```

The training script fits a decision tree and random forest, reports accuracy and a confusion matrix, then saves the better model to:

```txt
ml/models/blackjack_policy_model.joblib
```
