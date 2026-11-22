"""Download and validate the UCI Bike Sharing dataset."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

DATA_DIR = Path("data/raw")
ZIP_PATH = DATA_DIR / "bike-sharing-dataset.zip"
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def download_data() -> Path:
    """Download and extract the dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not (DATA_DIR / "hour.csv").exists():
        print("Downloading Bike Sharing dataset...")
        urlretrieve(DATA_URL, ZIP_PATH)

        with ZipFile(ZIP_PATH) as archive:
            archive.extractall(DATA_DIR)

    csv_path = DATA_DIR / "hour.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    data = pd.read_csv(csv_path)

    required_columns = {"dteday", "hr", "cnt"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    print(f"Dataset downloaded successfully: {data.shape[0]} rows")
    print(f"Saved to: {csv_path}")

    return csv_path


if __name__ == "__main__":
    download_data()
