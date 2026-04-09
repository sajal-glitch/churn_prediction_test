from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validation import validate_clean, validate_raw

NUMERIC_INT_COLS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Last Interaction",
]
NUMERIC_FLOAT_COLS = ["Total Spend"]
TARGET_COL = "Churn"
ID_COL = "CustomerID"


def load_raw(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both CSVs and run raw schema validation on each."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    validate_raw(train_df, dataset_name="training")
    validate_raw(test_df, dataset_name="testing")

    return train_df, test_df


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the all-null row, cast dtypes, and run post-cleaning validation."""
    df = df.dropna(how="all").reset_index(drop=True)

    for col in NUMERIC_INT_COLS:
        df[col] = df[col].astype("float64")

    df[NUMERIC_FLOAT_COLS[0]] = df[NUMERIC_FLOAT_COLS[0]].astype("float64")
    df[TARGET_COL] = df[TARGET_COL].astype("float64")

    validate_clean(df)
    return df


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop CustomerID and separate the Churn target column."""
    X = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def split_train_val(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/validation split."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )
    return X_train, X_val, y_train, y_val
