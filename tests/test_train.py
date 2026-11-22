import pandas as pd

from bike_sharing_ml.train import DROP_COLUMNS, TARGET


def test_target_is_cnt():
    assert TARGET == "cnt"


def test_leakage_columns_are_excluded():
    assert "casual" in DROP_COLUMNS
    assert "registered" in DROP_COLUMNS


def test_dataset_has_expected_target():
    data = pd.read_csv("data/raw/hour.csv")

    assert TARGET in data.columns
    assert data[TARGET].notna().all()
    assert (data[TARGET] >= 0).all()
