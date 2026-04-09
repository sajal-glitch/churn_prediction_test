from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.features.preprocessing import get_feature_names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics for churn prediction.

    Returns dict with keys: roc_auc, f1, precision, recall, avg_precision.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "avg_precision": average_precision_score(y_true, y_proba),
    }


def compute_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, output_path: str | Path) -> str:
    """Save confusion matrix heatmap to output_path, return path string."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return str(output_path)


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
) -> str:
    """Save ROC curve plot to output_path, return path string."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Model")
    ax.set_title("ROC Curve")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return str(output_path)


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def evaluate_on_holdout(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_dir: str | Path = "outputs/artifacts",
    model_name: str = "model",
) -> dict:
    """Full evaluation on the final holdout test set."""
    artifact_dir = Path(artifact_dir)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = compute_metrics(y_test, y_proba, y_pred)
    cm = compute_confusion_matrix(y_test, y_pred)

    cm_path = plot_confusion_matrix(cm, artifact_dir / f"{model_name}_holdout_cm.png")
    roc_path = plot_roc_curve(y_test, y_proba, artifact_dir / f"{model_name}_holdout_roc.png")

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "artifacts": {"confusion_matrix": cm_path, "roc_curve": roc_path},
    }


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def compute_shap_values(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for the churn (positive) class.

    Uses TreeExplainer for RF/XGBoost, LinearExplainer for LR.
    Returns (shap_values, X_transformed).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_val)

    clf_type = type(clf).__name__
    if clf_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(clf)
        shap_output = explainer.shap_values(X_transformed)
        # RandomForest returns list [class0, class1]; XGBoost returns array
        if isinstance(shap_output, list):
            shap_vals = shap_output[1]
        else:
            shap_vals = shap_output
    else:
        explainer = shap.LinearExplainer(clf, X_transformed)
        shap_output = explainer.shap_values(X_transformed)
        if isinstance(shap_output, list):
            shap_vals = shap_output[1]
        else:
            shap_vals = shap_output

    return shap_vals, X_transformed


def plot_shap_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
) -> str:
    """Save SHAP beeswarm summary plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(output_path)


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
) -> str:
    """Save SHAP mean absolute bar plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(output_path)


def get_top_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    n: int = 10,
) -> list[tuple[str, float]]:
    """Return top-n features by mean absolute SHAP value, descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    return ranked[:n]
