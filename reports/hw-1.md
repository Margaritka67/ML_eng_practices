# Homework 1 — Project Setup and Baseline Model

## 1. Project overview

The project predicts hourly bike rental demand using the UCI Bike Sharing Dataset.

- Task: regression
- Target: `cnt` (total hourly rentals)
- Dataset: 17,379 rows and 17 columns
- Python: 3.12
- Environment and dependency management: uv

## 2. Data acquisition

The dataset is downloaded using:

```bash
uv run python -m bike_sharing_ml.get_data
```

The raw data is stored in `data/raw/hour.csv` and excluded from Git.

## 3. Model

A Random Forest Regressor was trained.

The following columns were excluded:
- `instant` — technical identifier
- `dteday` — date column (not used directly)
- `casual` and `registered` — excluded to prevent target leakage

The dataset was split chronologically:
- Training set: 13,903 rows
- Test set: 3,476 rows
- Test period starts: 2012-08-07

## 4. Evaluation results

| Metric | Value |
|---|---:|
| MAE | 45.1529 |
| RMSE | 69.9113 |
| R² | 0.8995 |

The model achieved an R² score of 0.8995 on the held-out test period.

## 5. Reproducibility

Install dependencies:

```bash
uv sync
```

Download data:

```bash
uv run python -m bike_sharing_ml.get_data
```

Train the model:

```bash
uv run python -m bike_sharing_ml.train
```

Run tests:

```bash
uv run pytest -v
```

## 6. Code quality

- Ruff linting configured
- Ruff formatting configured
- pre-commit Git hook installed
- 3 tests passed

## 7. Artifacts

- `artifacts/model.pkl` — trained model (kept locally, excluded from Git)
- `artifacts/metrics.json` — evaluation metrics

## 8. Screenshots

### Dataset download
![Dataset download](screenshots/hw-1%20get_data.png)

### Model training and evaluation
![Model training](screenshots/hw-1%20train.png)

### Tests and code quality
![Tests and Ruff](screenshots/hw-1%20tests.png)

