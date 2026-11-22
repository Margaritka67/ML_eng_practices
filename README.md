# Bike Sharing ML

Machine learning project for predicting hourly bike rental demand using the
[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset).

## Requirements

- Python 3.12
- uv

## Setup

Clone the repository and install dependencies:

```bash
uv sync
```

## Download the dataset

```bash
uv run python -m bike_sharing_ml.get_data
```

The dataset is downloaded to `data/raw/`.

## Train the model

```bash
uv run python -m bike_sharing_ml.train
```

The training script:
- loads hourly bike rental data;
- excludes target leakage columns (`casual`, `registered`);
- splits data chronologically into training and test sets;
- trains a Random Forest regression model;
- evaluates MAE, RMSE and R².

Generated artifacts:
- `artifacts/model.pkl` — trained model;
- `artifacts/metrics.json` — evaluation metrics.

## Run tests

```bash
uv run pytest -v
```

## Code quality

```bash
uv run ruff check .
uv run ruff format --check .
```

## Dataset

The Bike Sharing Dataset contains hourly and daily rental records from
the Capital Bikeshare system in Washington, D.C., together with weather
and calendar features.

Source: UCI Machine Learning Repository.

