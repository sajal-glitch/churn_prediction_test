"""
scripts/run_eda.py — Standardized EDA for data/raw churn CSVs.

Outputs:
  reports/eda.md          — human-readable Markdown report
  reports/missingness.png — seaborn missingness heatmap
  reports/distributions/  — one histogram per numeric column
  reports/correlations.png — correlation heatmap with Churn target
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw")
TRAIN_PATH = DATA_DIR / "customer_churn_dataset-training-master.csv"
TEST_PATH = DATA_DIR / "customer_churn_dataset-testing-master.csv"
REPORT_DIR = Path("reports")
DIST_DIR = REPORT_DIR / "distributions"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
DIST_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Churn"
NUMERIC_COLS = [
    "Age", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Total Spend", "Last Interaction",
]
CATEGORICAL_COLS = ["Gender", "Subscription Type", "Contract Length"]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

lines: list[str] = []


def h(level: int, text: str) -> None:
    lines.append(f"\n{'#' * level} {text}\n")


def para(text: str) -> None:
    lines.append(text + "\n")


def table(df: pd.DataFrame) -> None:
    lines.append(df.to_markdown(index=True) + "\n")


# ---------------------------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------------------------
h(1, "EDA Report — Churn Prediction Dataset")
para(f"Generated from `{TRAIN_PATH}` (train) and `{TEST_PATH}` (test).")

h(2, "1. Dataset Overview")

overview = pd.DataFrame(
    {
        "Rows": [train_df.shape[0], test_df.shape[0]],
        "Columns": [train_df.shape[1], test_df.shape[1]],
    },
    index=["Training", "Testing"],
)
table(overview)

h(3, "Column dtypes (training)")
dtype_df = pd.DataFrame({"dtype": train_df.dtypes.astype(str)})
table(dtype_df)

# ---------------------------------------------------------------------------
# 2. Target balance
# ---------------------------------------------------------------------------
h(2, "2. Target Balance")

for name, df in [("Training", train_df), ("Testing", test_df)]:
    counts = df[TARGET].value_counts().sort_index()
    pcts = (counts / len(df) * 100).round(2)
    bal = pd.DataFrame(
        {"Count": counts, "Percentage (%)": pcts},
        index=["Retained (0)", "Churned (1)"],
    )
    h(3, name)
    table(bal)
    ratio = counts.get(1, 0) / max(counts.get(0, 1), 1)
    para(f"Churn rate: **{pcts.iloc[1]:.2f}%** | Imbalance ratio (churn/retained): **{ratio:.3f}**")

# ---------------------------------------------------------------------------
# 3. Missingness
# ---------------------------------------------------------------------------
h(2, "3. Missingness")

for name, df in [("Training", train_df), ("Testing", test_df)]:
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(4)
    miss_df = pd.DataFrame({"Null Count": null_counts, "Null %": null_pct})
    miss_df = miss_df[miss_df["Null Count"] > 0]
    h(3, name)
    if miss_df.empty:
        para("No missing values detected.")
    else:
        table(miss_df)

# Missingness heatmap (training only — has the empty row)
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(
    train_df.isnull(),
    yticklabels=False,
    cbar=False,
    cmap="viridis",
    ax=ax,
)
ax.set_title("Missingness Heatmap — Training Data")
ax.set_xlabel("Column")
plt.tight_layout()
miss_path = REPORT_DIR / "missingness.png"
fig.savefig(miss_path, dpi=120)
plt.close(fig)
para(f"![Missingness Heatmap](missingness.png)")

# ---------------------------------------------------------------------------
# 4. Numeric distributions
# ---------------------------------------------------------------------------
h(2, "4. Numeric Column Distributions")

stats = train_df[NUMERIC_COLS].describe().T.round(3)
stats["skewness"] = train_df[NUMERIC_COLS].skew().round(3)
table(stats)

# One histogram per numeric column, split by churn
for col in NUMERIC_COLS:
    fig, ax = plt.subplots(figsize=(7, 4))
    for churn_val, label, color in [(0, "Retained", "steelblue"), (1, "Churned", "salmon")]:
        subset = train_df[train_df[TARGET] == churn_val][col].dropna()
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, density=True)
    ax.set_title(f"{col} — Distribution by Churn")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plot_path = DIST_DIR / f"{col.replace(' ', '_')}.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    para(f"![{col}](distributions/{col.replace(' ', '_')}.png)")

# ---------------------------------------------------------------------------
# 5. Categorical cardinality & frequency
# ---------------------------------------------------------------------------
h(2, "5. Categorical Cardinality & Value Counts")

for col in CATEGORICAL_COLS:
    h(3, col)
    vc = train_df[col].value_counts(dropna=False)
    pct = (vc / len(train_df) * 100).round(2)
    cat_df = pd.DataFrame({"Count": vc, "Percentage (%)": pct})
    para(f"Cardinality: **{train_df[col].nunique()}** unique values")
    table(cat_df)

    # Churn rate per category
    churn_by_cat = (
        train_df.groupby(col)[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Churn Rate", "count": "N"})
        .sort_values("Churn Rate", ascending=False)
        .round(4)
    )
    para("Churn rate by category:")
    table(churn_by_cat)

# ---------------------------------------------------------------------------
# 6. Correlation with target
# ---------------------------------------------------------------------------
h(2, "6. Correlation with Target (Churn)")

# Point-biserial correlation for numerics (same as Pearson for binary target)
corr_series = train_df[NUMERIC_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(
    key=abs, ascending=False
).round(4)
corr_df = pd.DataFrame({"Pearson r with Churn": corr_series})
table(corr_df)

para(
    "> Interpretation: positive r = higher values associated with more churn; "
    "negative r = higher values associated with retention."
)

# Full numeric correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = train_df[NUMERIC_COLS + [TARGET]].corr().round(3)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax,
    linewidths=0.5,
)
ax.set_title("Correlation Heatmap (Numeric Features + Churn)")
plt.tight_layout()
corr_path = REPORT_DIR / "correlations.png"
fig.savefig(corr_path, dpi=120)
plt.close(fig)
para(f"![Correlation Heatmap](correlations.png)")

# ---------------------------------------------------------------------------
# 7. Churn rate by numeric bins
# ---------------------------------------------------------------------------
h(2, "7. Churn Rate by Numeric Quartile")

for col in NUMERIC_COLS:
    binned = pd.qcut(train_df[col].dropna(), q=4, duplicates="drop")
    churn_by_bin = (
        train_df.loc[binned.index]
        .groupby(binned)[TARGET]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={col: "Quartile", TARGET: "Churn Rate"})
    )
    churn_by_bin["Quartile"] = churn_by_bin["Quartile"].astype(str)
    h(3, col)
    table(churn_by_bin.set_index("Quartile"))

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
report_path = REPORT_DIR / "eda.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\nEDA complete. Report written to: {report_path}")
print(f"Plots saved to: {REPORT_DIR}/")
