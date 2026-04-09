import numpy as np
import pandas as pd
import pandera as pa
import pytest

from src.data.loader import clean_train
from src.data.validation import validate_clean, validate_raw


def _make_valid_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": range(1, n + 1),
            "Age": [30, 45, 22, 60, 35],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "Tenure": [12, 24, 6, 48, 36],
            "Usage Frequency": [10, 20, 5, 15, 25],
            "Support Calls": [2, 5, 0, 8, 3],
            "Payment Delay": [5, 10, 0, 20, 15],
            "Subscription Type": ["Basic", "Standard", "Premium", "Basic", "Standard"],
            "Contract Length": ["Monthly", "Quarterly", "Annual", "Monthly", "Annual"],
            "Total Spend": [200.0, 500.0, 150.0, 900.0, 350.0],
            "Last Interaction": [5, 10, 1, 25, 15],
            "Churn": [0, 1, 0, 1, 0],
        }
    )


def test_valid_schema_passes():
    df = _make_valid_df()
    result = validate_raw(df, dataset_name="test")
    assert result is not None
    assert len(result) == 5


def test_empty_row_is_dropped_by_clean_train():
    df = _make_valid_df()
    empty_row = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
    df_with_empty = pd.concat([df, empty_row], ignore_index=True)
    assert len(df_with_empty) == 6
    cleaned = clean_train(df_with_empty)
    assert len(cleaned) == 5


def test_out_of_range_age_fails():
    df = _make_valid_df()
    df.loc[0, "Age"] = 100
    with pytest.raises(pa.errors.SchemaError):
        validate_raw(df, dataset_name="test")


def test_invalid_gender_fails():
    df = _make_valid_df()
    df.loc[0, "Gender"] = "Unknown"
    with pytest.raises(pa.errors.SchemaError):
        validate_raw(df, dataset_name="test")


def test_invalid_subscription_type_fails():
    df = _make_valid_df()
    df.loc[0, "Subscription Type"] = "Gold"
    with pytest.raises(pa.errors.SchemaError):
        validate_raw(df, dataset_name="test")
