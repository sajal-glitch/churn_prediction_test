# 🔄 Churn Prediction Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow)
![Ruff](https://img.shields.io/badge/Linting-Ruff-purple)

A **production-quality binary churn classification pipeline** that predicts whether a customer will churn (`1`) or be retained (`0`). Built with scikit-learn, XGBoost, MLflow, and pandera — designed for reproducibility, explainability, and zero data leakage.

---

## 📋 Table of Contents

- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Running the Pipeline](#-running-the-pipeline)
- [ML Pipeline Stages](#-ml-pipeline-stages)
- [Testing](#-testing)
- [Key Design Decisions](#-key-design-decisions)
- [Technology Stack](#-technology-stack)

---

## 📊 Dataset

| Split    | Rows    | Features |
|----------|---------|----------|
| Training | 440,832 | 12       |
| Testing  | 64,374  | 12       |

**Feature set:**

| Type | Features |
|------|----------|
| Numeric (7) | `Age`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction` |
| Categorical (3) | `Gender`, `Subscription Type` (Basic/Standard/Premium), `Contract Length` (Monthly/Quarterly/Annual) |

> **Class balance (training):** ~56.7% churned / 43.3% retained — mild imbalance handled via `class_weight='balanced'` and `scale_pos_weight`.

---

## 📁 Project Structure

```
churn_prediction_test/
├── configs/
│   └── model_params.yaml          # Hyperparameter grids for all models
├── data/
│   ├── raw/                        # Source CSVs (train + test)
│   └── processed/                  # churn.db (SQLite)
├── reports/
│   ├── eda.md                      # EDA report
│   ├── correlations.png
│   ├── missingness.png
│   └── distributions/              # Per-feature distribution plots
├── scripts/
│   ├── run_eda.py
│   └── run_evaluate.py
├── src/
│   ├── data/
│   │   ├── loader.py               # Load, clean, split
│   │   └── validation.py           # Pandera schema validation
│   ├── features/
│   │   └── preprocessing.py        # ColumnTransformer builder
│   ├── models/
│   │   ├── train.py                # Pipeline build, CV, tuning (RandomizedSearch + Optuna)
│   │   └── evaluate.py             # Metrics, plots, SHAP
│   └── utils/
│       └── mlflow_helpers.py       # Experiment setup, run logging, model registry
├── tests/
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── run_pipeline.py                  # End-to-end orchestrator
├── requirements.txt
└── CLAUDE.md
```

---

## 🚀 Quick Start

**Requirements:** Python 3.10+

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\activate           # Windows cmd/PowerShell
# source .venv/bin/activate        # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# If requirements.txt is UTF-16LE encoded (Windows):
iconv -f UTF-16LE -t UTF-8 requirements.txt | pip install -r /dev/stdin

# 3. Run the full pipeline
python run_pipeline.py

# 4. Open the MLflow UI to explore results
mlflow ui
# → Visit http://localhost:5000
```

---

## ▶️ Running the Pipeline

```bash
# Full end-to-end run: validate → preprocess → train 3 models → evaluate → log to MLflow
python run_pipeline.py

# View experiment results in the MLflow UI
mlflow ui
# Open http://localhost:5000
```

**Pipeline flow:**

```
Raw CSVs
  └─► Pandera Validation
        └─► Data Loading & Cleaning
              └─► Train/Val Split (stratified 80/20)
                    └─► ColumnTransformer (Preprocessing)
                          └─► 3 Models trained in parallel
                                ├─ Logistic Regression  ──┐
                                ├─ Random Forest         ├─► MLflow Experiments
                                └─ XGBoost (+ Optuna)   ──┘
                                      └─► Best Model Selected
                                            └─► Holdout Evaluation + Model Registry
```

---

## 🧪 ML Pipeline Stages

### Stage 1 — Data Validation (`src/data/validation.py`)

Pandera `SchemaModel` validates **all columns at ingestion** — ranges, allowed categories, and types — before any processing begins.

### Stage 2 — Data Loading & Splitting (`src/data/loader.py`)

- Loads both CSVs and drops the single all-null training row
- Casts types and performs a **stratified 80/20 train/val split**

### Stage 3 — Preprocessing (`src/features/preprocessing.py`)

A single `ColumnTransformer` with three sub-pipelines:

| Transformer | Columns | Steps |
|-------------|---------|-------|
| Numeric (7) | Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction | `SimpleImputer(median)` → `RobustScaler` |
| Gender (1) | Gender | `SimpleImputer(most_frequent)` → `OrdinalEncoder` |
| Ordinal (2) | Subscription Type, Contract Length | `SimpleImputer(most_frequent)` → `OrdinalEncoder` (ordered) |

> `RobustScaler` is used over `StandardScaler` because Support Calls and Payment Delay have floor effects and outliers. Ordinal encoding for Subscription Type and Contract Length preserves real business meaning (Basic < Standard < Premium, Monthly < Quarterly < Annual).

### Stage 4 — Model Training (`src/models/train.py`)

Three candidate models are trained and compared:

| Model | Tuning Strategy | Notes |
|-------|----------------|-------|
| Logistic Regression | `RandomizedSearchCV` over C, penalty, max_iter | Interpretable baseline |
| Random Forest | `RandomizedSearchCV` over n_estimators, max_depth, min_samples | Non-linear, robust to scale |
| XGBoost | `RandomizedSearchCV` + **Optuna TPE** (50 trials) | Best tabular performance; native SHAP |

> All models are wrapped in a scikit-learn `Pipeline` (preprocessor → classifier) to guarantee **no data leakage** — the preprocessor fits only on training folds.

### Stage 5 — Evaluation (`src/models/evaluate.py`)

| Category | Details |
|----------|---------|
| **Primary metric** | ROC-AUC (threshold-independent, robust to mild imbalance) |
| **Secondary metrics** | F1, Precision, Recall, Average Precision |
| **Plots** | Confusion matrix, ROC curve, SHAP summary, SHAP bar chart |
| **SHAP explainer** | `TreeExplainer` for RF/XGBoost · `LinearExplainer` for Logistic Regression |

### Stage 6 — Experiment Tracking (`src/utils/mlflow_helpers.py`)

- One MLflow experiment per model type: `churn_logistic_regression`, `churn_random_forest`, `churn_xgboost`
- `mlflow.sklearn.autolog()` / `mlflow.xgboost.autolog()` for automatic parameter and metric logging
- SHAP plots, confusion matrices, and ROC curves logged as artifacts
- Best model registered in the MLflow Model Registry under `ChurnPredictionModel`

---

## ✅ Testing

```bash
# Run all tests
pytest

# Run a specific test file with verbose output
pytest tests/test_preprocessing.py -v

# Lint and format
ruff check .
ruff format .
```

**Test coverage:**

| Test File | What It Covers |
|-----------|---------------|
| `test_validation.py` | Schema enforcement, invalid value detection, null row handling |
| `test_preprocessing.py` | Output shape, no-leakage guarantee, ordinal ordering |
| `test_models.py` | Pipeline train/predict, metrics dict completeness, confusion matrix shape |

---

## 💡 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No SMOTE** | Class imbalance (~57/43) is mild enough that `class_weight='balanced'` suffices |
| **RobustScaler** | Preferred over StandardScaler for features with floor effects and outliers |
| **Ordinal encoding** | Preserves business-meaningful ordering rather than exploding dimensionality with one-hot encoding |
| **Optuna for XGBoost** | Bayesian TPE search over continuous ranges outperforms discrete RandomizedSearch for XGBoost's many continuous hyperparameters |
| **sklearn Pipeline** | Ensures preprocessing statistics are fit only on training data — no leakage into validation or test sets |

---

## 🛠️ Technology Stack

| Purpose | Library |
|---------|---------|
| Data validation | `pandera` |
| ML pipeline | `scikit-learn` |
| Gradient boosting | `xgboost` |
| Hyperparameter tuning | `optuna` |
| Experiment tracking | `mlflow` |
| Explainability | `shap` |
| Model serialization | `joblib` |
| Testing | `pytest` |
| Linting | `ruff` |
