import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import build_preprocessor, get_feature_names


def _make_X(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Age": rng.integers(18, 65, n).astype(float),
            "Tenure": rng.integers(1, 60, n).astype(float),
            "Usage Frequency": rng.integers(1, 30, n).astype(float),
            "Support Calls": rng.integers(0, 10, n).astype(float),
            "Payment Delay": rng.integers(0, 30, n).astype(float),
            "Total Spend": rng.uniform(100, 1000, n),
            "Last Interaction": rng.integers(1, 30, n).astype(float),
            "Gender": rng.choice(["Male", "Female"], n),
            "Subscription Type": rng.choice(["Basic", "Standard", "Premium"], n),
            "Contract Length": rng.choice(["Monthly", "Quarterly", "Annual"], n),
        }
    )


def test_preprocessor_output_shape():
    X = _make_X(50)
    preprocessor = build_preprocessor()
    X_out = preprocessor.fit_transform(X)
    # 7 numeric + 1 gender + 2 ordinal = 10 columns
    assert X_out.shape == (50, 10)


def test_no_nulls_in_output():
    X = _make_X(50)
    preprocessor = build_preprocessor()
    X_out = preprocessor.fit_transform(X)
    assert not np.isnan(X_out).any()


def test_no_leakage_val_uses_train_stats():
    X_train = _make_X(100)
    X_val = _make_X(20)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # RobustScaler center_ is computed from train only
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    train_center = scaler.center_.copy()

    # Transform val — should not change train stats
    preprocessor.transform(X_val)
    assert np.allclose(scaler.center_, train_center)


def test_subscription_ordinal_order():
    X = pd.DataFrame(
        {
            "Age": [30.0] * 3,
            "Tenure": [12.0] * 3,
            "Usage Frequency": [10.0] * 3,
            "Support Calls": [2.0] * 3,
            "Payment Delay": [5.0] * 3,
            "Total Spend": [300.0] * 3,
            "Last Interaction": [5.0] * 3,
            "Gender": ["Male"] * 3,
            "Subscription Type": ["Basic", "Standard", "Premium"],
            "Contract Length": ["Monthly"] * 3,
        }
    )
    preprocessor = build_preprocessor()
    X_out = preprocessor.fit_transform(X)
    # Subscription Type is column index 8 (7 numeric + 1 gender + 0-indexed ordinal col 0)
    sub_vals = X_out[:, 8]
    assert sub_vals[0] < sub_vals[1] < sub_vals[2]


def test_contract_length_ordinal_order():
    X = pd.DataFrame(
        {
            "Age": [30.0] * 3,
            "Tenure": [12.0] * 3,
            "Usage Frequency": [10.0] * 3,
            "Support Calls": [2.0] * 3,
            "Payment Delay": [5.0] * 3,
            "Total Spend": [300.0] * 3,
            "Last Interaction": [5.0] * 3,
            "Gender": ["Male"] * 3,
            "Subscription Type": ["Basic"] * 3,
            "Contract Length": ["Monthly", "Quarterly", "Annual"],
        }
    )
    preprocessor = build_preprocessor()
    X_out = preprocessor.fit_transform(X)
    # Contract Length is column index 9
    contract_vals = X_out[:, 9]
    assert contract_vals[0] < contract_vals[1] < contract_vals[2]


def test_get_feature_names_length():
    X = _make_X(20)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    names = get_feature_names(preprocessor)
    assert len(names) == 10
