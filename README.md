# Churn Prediction Model

A production-quality binary churn classification pipeline built with scikit-learn, XGBoost, MLflow, and pandera. Predicts whether a customer will churn (`1`) or be retained (`0`).

## Dataset

| Split    | Rows    | Features |
|----------|---------|----------|
| Training | 440,832 | 12       |
| Testing  | 64,374  | 12       |

**Feature set:** `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type` (Basic/Standard/Premium), `Contract Length` (Monthly/Quarterly/Annual), `Total Spend`, `Last Interaction`

**Class balance (training):** ~56.7% churned / 43.3% retained — mild imbalance handled via `class_weight='balanced'` and `scale_pos_weight`.

## Project Structure

```
Churn_prediction_model/
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

## Setup

**Requirements:** Python 3.10+

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\activate           # Windows cmd/PowerShell
# source .venv/bin/activate        # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# If requirements.txt is UTF-16LE encoded (Windows):
iconv -f UTF-16LE -t UTF-8 requirements.txt | pip install -r /dev/stdin
```

## Running the Pipeline

```bash
# Full end-to-end run: validate → preprocess → train 3 models → evaluate → log to MLflow
python run_pipeline.py

# View experiment results in the MLflow UI
mlflow ui
# Open http://localhost:5000
```

## ML Pipeline Stages

### 1. Data Validation (`src/data/validation.py`)
Pandera `SchemaModel` validates all columns at ingestion — ranges, allowed categories, and types — before any processing.

### 2. Data Loading & Splitting (`src/data/loader.py`)
Loads both CSVs, drops the single all-null training row, casts types, and performs a stratified 80/20 train/val split.

### 3. Preprocessing (`src/features/preprocessing.py`)
A single `ColumnTransformer` with three sub-pipelines:

| Transformer | Columns | Steps |
|-------------|---------|-------|
| Numeric (7) | Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction | `SimpleImputer(median)` → `RobustScaler` |
| Gender (1) | Gender | `SimpleImputer(most_frequent)` → `OrdinalEncoder` |
| Ordinal (2) | Subscription Type, Contract Length | `SimpleImputer(most_frequent)` → `OrdinalEncoder` (ordered) |

`RobustScaler` is used over `StandardScaler` because Support Calls and Payment Delay have floor effects and outliers. Ordinal encoding for Subscription Type and Contract Length preserves real business meaning (Basic < Standard < Premium, Monthly < Quarterly < Annual).

### 4. Models

Three candidate models are trained and compared:

| Model | Tuning | Notes |
|-------|--------|-------|
| Logistic Regression | `RandomizedSearchCV` over C, penalty, max_iter | Interpretable baseline |
| Random Forest | `RandomizedSearchCV` over n_estimators, max_depth, min_samples | Non-linear, robust to scale |
| XGBoost | `RandomizedSearchCV` + **Optuna TPE** (50 trials) | Best tabular performance; native SHAP |

All models are wrapped in a scikit-learn `Pipeline` (preprocessor → classifier) to guarantee **no data leakage** — the preprocessor fits only on training folds.

### 5. Evaluation (`src/models/evaluate.py`)
- **Primary metric:** ROC-AUC (threshold-independent, robust to mild imbalance)
- **Secondary:** F1, Precision, Recall, Average Precision
- **Plots:** Confusion matrix, ROC curve, SHAP summary, SHAP bar chart
- **SHAP:** `TreeExplainer` for RF/XGBoost, `LinearExplainer` for Logistic Regression

### 6. Experiment Tracking (`src/utils/mlflow_helpers.py`)
- One MLflow experiment per model type: `churn_logistic_regression`, `churn_random_forest`, `churn_xgboost`
- `mlflow.sklearn.autolog()` / `mlflow.xgboost.autolog()` for automatic parameter and metric logging
- SHAP plots, confusion matrices, and ROC curves logged as artifacts
- Best model registered in the MLflow Model Registry under `ChurnPredictionModel`

## Testing

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_preprocessing.py -v

# Lint and format
ruff check .
ruff format .
```

**Test coverage:**
- `test_validation.py` — schema enforcement, invalid value detection, null row handling
- `test_preprocessing.py` — output shape, no-leakage guarantee, ordinal ordering
- `test_models.py` — pipeline train/predict, metrics dict completeness, confusion matrix shape

## Key Design Decisions

- **No SMOTE:** Class imbalance (~57/43) is mild enough that `class_weight='balanced'` suffices.
- **RobustScaler:** Preferred over StandardScaler for features with floor effects and outliers.
- **Ordinal encoding for categories:** Preserves business-meaningful ordering rather than exploding dimensionality with one-hot encoding.
- **Optuna for XGBoost:** Bayesian TPE search over continuous ranges outperforms discrete RandomizedSearch for XGBoost's many continuous hyperparameters.
- **sklearn Pipeline:** Ensures preprocessing statistics (scaler means, encoder mappings) are fit only on training data — no leakage into validation or test sets.

## Technology Stack

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
