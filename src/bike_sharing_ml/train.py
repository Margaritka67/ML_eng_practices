"""Train and evaluate a bike-sharing demand prediction model."""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/raw/hour.csv")
ARTIFACTS_DIR = Path("artifacts")
TARGET = "cnt"

# Exclude identifiers and columns that directly reveal the target.
DROP_COLUMNS = ["instant", "dteday", "casual", "registered"]


def train() -> dict:
    """Train a model and save it together with evaluation metrics."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run get_data first."
        )

    data = pd.read_csv(DATA_PATH)
    data = data.sort_values(["dteday", "hr"]).reset_index(drop=True)

    X = data.drop(columns=[TARGET, *DROP_COLUMNS])
    y = data[TARGET]

    # Chronological split: first 80% for training, last 20% for testing.
    split_index = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "model": "RandomForestRegressor",
        "target": TARGET,
        "features": X.columns.tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_start_date": data.iloc[split_index]["dteday"],
        "mae": round(mean_absolute_error(y_test, predictions), 4),
        "rmse": round(mean_squared_error(y_test, predictions) ** 0.5, 4),
        "r2": round(r2_score(y_test, predictions), 4),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the trained model.
    import pickle

    with (ARTIFACTS_DIR / "model.pkl").open("wb") as file:
        pickle.dump(model, file)

    # Save metrics and experiment details.
    with (ARTIFACTS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print("Training completed!")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")

    return metrics


if __name__ == "__main__":
    train()
