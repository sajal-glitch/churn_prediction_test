import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.preprocessing import build_preprocessor
from src.models.evaluate import compute_confusion_matrix, compute_metrics
from src.models.train import build_full_pipeline


def _make_dataset(n: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
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
    y = pd.Series(rng.integers(0, 2, n), name="Churn")
    return X, y


def test_logistic_regression_pipeline_trains_and_predicts():
    X, y = _make_dataset(1000)
    preprocessor = build_preprocessor()
    estimator = LogisticRegression(
        class_weight="balanced", solver="saga", max_iter=200, random_state=42
    )
    pipeline = build_full_pipeline(preprocessor, estimator)
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)
    assert proba.shape == (1000, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_compute_metrics_returns_all_keys():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4])
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = compute_metrics(y_true, y_proba, y_pred)
    expected_keys = {"roc_auc", "f1", "precision", "recall", "avg_precision"}
    assert expected_keys == set(metrics.keys())


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)


def test_confusion_matrix_sum_equals_n():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.sum() == len(y_true)
