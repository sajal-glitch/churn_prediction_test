"""
scripts/run_evaluate.py — Evaluate a trained churn model on the holdout test set.

Usage:
    python scripts/run_evaluate.py <model_path>

    <model_path>  Path to a joblib-serialised sklearn Pipeline
                  (must expose predict_proba and have named steps
                  'preprocessor' and 'classifier').

Outputs written to reports/evaluate/:
    confusion_matrix.png
    roc_curve.png
    pr_curve.png
    calibration_curve.png
    shap_summary.png
    shap_bar.png

Report written to:
    reports/evaluate.md
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Resolve project root so we can import src regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import clean_train, get_feature_target, load_raw  # noqa: E402
from src.models.evaluate import (  # noqa: E402
    compute_confusion_matrix,
    compute_shap_values,
    get_top_features,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_shap_bar,
    plot_shap_summary,
)
from src.features.preprocessing import get_feature_names  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python scripts/run_evaluate.py <model_path>")
    sys.exit(1)

MODEL_PATH = Path(sys.argv[1])
if not MODEL_PATH.exists():
    print(f"Error: model file not found: {MODEL_PATH}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "customer_churn_dataset-training-master.csv"
TEST_PATH = DATA_DIR / "customer_churn_dataset-testing-master.csv"

REPORT_DIR = ROOT / "reports" / "evaluate"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_MD = ROOT / "reports" / "evaluate.md"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading test data...")
_, test_df = load_raw(TRAIN_PATH, TEST_PATH)
X_test, y_test = get_feature_target(test_df)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading model from {MODEL_PATH} ...")
pipeline = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
print("Running predictions...")
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
metrics = {
    "ROC-AUC": roc_auc_score(y_test, y_proba),
    "F1": f1_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "Avg Precision (PR-AUC)": average_precision_score(y_test, y_proba),
}

# ---------------------------------------------------------------------------
# Plot 1 — Confusion matrix
# ---------------------------------------------------------------------------
cm = compute_confusion_matrix(y_test, y_pred)
cm_path = plot_confusion_matrix(cm, REPORT_DIR / "confusion_matrix.png")
print(f"  Saved: {cm_path}")

# ---------------------------------------------------------------------------
# Plot 2 — ROC curve
# ---------------------------------------------------------------------------
roc_path = plot_roc_curve(y_test, y_proba, REPORT_DIR / "roc_curve.png")
print(f"  Saved: {roc_path}")

# ---------------------------------------------------------------------------
# Plot 3 — Precision-Recall curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(
    y_test, y_proba, ax=ax, name="Model",
    plot_chance_level=True,
)
ax.set_title("Precision-Recall Curve")
plt.tight_layout()
pr_path = REPORT_DIR / "pr_curve.png"
fig.savefig(pr_path, dpi=120)
plt.close(fig)
print(f"  Saved: {pr_path}")

# ---------------------------------------------------------------------------
# Plot 4 — Calibration curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
CalibrationDisplay.from_predictions(
    y_test, y_proba, n_bins=10, ax=ax, name="Model",
)
ax.set_title("Calibration Curve (Reliability Diagram)")
plt.tight_layout()
cal_path = REPORT_DIR / "calibration_curve.png"
fig.savefig(cal_path, dpi=120)
plt.close(fig)
print(f"  Saved: {cal_path}")

# ---------------------------------------------------------------------------
# Plot 5 — SHAP (sample 5 000 rows for speed)
# ---------------------------------------------------------------------------
print("Computing SHAP values (sample of 5 000 rows)...")
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_test), size=min(5_000, len(X_test)), replace=False)
X_shap = X_test.iloc[sample_idx].reset_index(drop=True)

try:
    shap_values, X_transformed = compute_shap_values(pipeline, X_shap)
    feature_names = get_feature_names(pipeline.named_steps["preprocessor"])

    shap_summary_path = plot_shap_summary(
        shap_values, X_transformed, feature_names, REPORT_DIR / "shap_summary.png"
    )
    shap_bar_path = plot_shap_bar(shap_values, feature_names, REPORT_DIR / "shap_bar.png")
    top_features = get_top_features(shap_values, feature_names, n=10)
    shap_ok = True
    print(f"  Saved: {shap_summary_path}")
    print(f"  Saved: {shap_bar_path}")
except Exception as exc:
    print(f"  Warning: SHAP computation failed — {exc}")
    shap_ok = False
    top_features = []

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
model_name = MODEL_PATH.stem
lines: list[str] = []


def h(level: int, text: str) -> None:
    lines.append(f"\n{'#' * level} {text}\n")


def para(text: str) -> None:
    lines.append(text + "\n")


h(1, f"Model Evaluation Report — `{model_name}`")
para(f"**Model path:** `{MODEL_PATH}`  ")
para(f"**Test set:** `{TEST_PATH}`  ")
para(f"**Test rows:** {len(y_test):,}  ")
para(f"**Churn rate (test):** {y_test.mean():.2%}")

# --- Metrics table ---
h(2, "1. Metrics Summary")
metrics_df = pd.DataFrame(
    {"Value": [f"{v:.4f}" for v in metrics.values()]},
    index=list(metrics.keys()),
)
lines.append(metrics_df.to_markdown(index=True) + "\n")

# --- Confusion matrix ---
h(2, "2. Confusion Matrix")
tn, fp, fn, tp = cm.ravel()
para(
    f"| | Predicted Retained | Predicted Churned |\n"
    f"|---|---|---|\n"
    f"| **Actual Retained** | {tn:,} (TN) | {fp:,} (FP) |\n"
    f"| **Actual Churned** | {fn:,} (FN) | {tp:,} (TP) |"
)
para("\n![Confusion Matrix](evaluate/confusion_matrix.png)")

# --- ROC ---
h(2, "3. ROC Curve")
para(f"ROC-AUC: **{metrics['ROC-AUC']:.4f}**")
para("![ROC Curve](evaluate/roc_curve.png)")

# --- PR ---
h(2, "4. Precision-Recall Curve")
para(f"PR-AUC (Average Precision): **{metrics['Avg Precision (PR-AUC)']:.4f}**")
para("![PR Curve](evaluate/pr_curve.png)")

# --- Calibration ---
h(2, "5. Calibration Curve")
para(
    "A well-calibrated model's curve hugs the diagonal. "
    "Points above the diagonal = underestimates churn probability; "
    "below = overestimates."
)
para("![Calibration Curve](evaluate/calibration_curve.png)")

# --- SHAP ---
h(2, "6. SHAP Feature Importance")
if shap_ok:
    para("Computed on a 5 000-row sample of the test set.")
    para("**Top 10 features by mean |SHAP value|:**")
    shap_df = pd.DataFrame(top_features, columns=["Feature", "Mean |SHAP|"])
    shap_df["Mean |SHAP|"] = shap_df["Mean |SHAP|"].round(4)
    lines.append(shap_df.to_markdown(index=False) + "\n")
    para("\n![SHAP Summary (beeswarm)](evaluate/shap_summary.png)")
    para("![SHAP Bar (mean |SHAP|)](evaluate/shap_bar.png)")
else:
    para("_SHAP computation failed — see console output for details._")

REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"\nEvaluation complete.")
print(f"  Report : {REPORT_MD}")
print(f"  Plots  : {REPORT_DIR}/")
