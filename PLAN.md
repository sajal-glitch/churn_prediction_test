# Churn Prediction Pipeline — Full Plan

## Context

This project has raw CSV data but no Python source code yet. The goal is to build a complete, production-quality churn prediction pipeline from scratch: validated data loading → preprocessing → three candidate models → evaluation → MLflow tracking, all organized to prevent data leakage and be testable.

---

## Dataset Profile (from exploration)

| Dataset | Rows | Columns |
|---|---|---|
| Training | 440,832 | 12 |
| Testing | 64,374 | 12 |

**Features:**
- Numeric (7): `Age` (18–65), `Tenure` (1–60), `Usage Frequency` (1–30), `Support Calls` (0–10), `Payment Delay` (0–30), `Total Spend` (100–1000, float), `Last Interaction` (1–30)
- Categorical (3): `Gender` (Male/Female), `Subscription Type` (Basic/Standard/Premium), `Contract Length` (Monthly/Quarterly/Annual)
- ID (drop): `CustomerID`
- Target: `Churn` (0=retained, 1=churned)

**Data quality issues:**
- 1 fully-empty row in training set (all 12 fields blank) → drop with `dropna(how='all')`
- No partial-null rows exist in either dataset
- `Total Spend` is genuinely mixed int/float precision (not a bug, cast to float64)
- Training class balance: **56.7% churn / 43.3% retained** (mild imbalance; class weights sufficient, no SMOTE needed)
- Test class balance: 47.4% / 52.6% (near-balanced)

---

## Project Structure

```
Churn_prediction_model/
├── configs/
│   └── model_params.yaml          # hyperparameter grids (prefixed classifier__)
├── data/raw/                       # existing CSVs
├── outputs/
│   ├── artifacts/                  # plots: confusion matrices, ROC, SHAP
│   └── models/                     # best_pipeline.joblib
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # load, clean, split
│   │   └── validation.py           # pandera schema + validators
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessing.py        # ColumnTransformer builder
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                # build pipeline, tune, CV
│   │   └── evaluate.py             # metrics, plots, SHAP
│   └── utils/
│       ├── __init__.py
│       └── mlflow_helpers.py       # experiment setup, log_run, register
├── tests/
│   ├── __init__.py
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── run_pipeline.py                 # orchestrator entry point
├── PLAN.md                         # this file
└── CLAUDE.md
```

---

## Stage 1 — Data Validation (`src/data/validation.py`)

Use **pandera `SchemaModel`** (class-based). All numeric columns declared as `pa.Float` with `nullable=True` to tolerate the empty row before cleaning; `coerce=True` at model level.

**Schema constraints per column:**
- `CustomerID`: Float, nullable
- `Age`: Float, ge=18, le=65
- `Gender`: Object, isin(["Male","Female"])
- `Tenure`: Float, ge=1, le=60
- `Usage Frequency`: Float, ge=1, le=30
- `Support Calls`: Float, ge=0, le=10
- `Payment Delay`: Float, ge=0, le=30
- `Subscription Type`: Object, isin(["Basic","Standard","Premium"])
- `Contract Length`: Object, isin(["Monthly","Quarterly","Annual"])
- `Total Spend`: Float, ge=100.0, le=1000.0
- `Last Interaction`: Float, ge=1, le=30
- `Churn`: Float, isin([0.0, 1.0])

**Functions:**
- `validate_raw(df, schema, dataset_name) -> pd.DataFrame` — wraps `schema.validate()`, raises with context on failure
- `validate_clean(df) -> pd.DataFrame` — post-cleaning tighter check (no nulls, Churn in {0,1} as int)

---

## Stage 2 — Data Loading & Splitting (`src/data/loader.py`)

**Functions:**
- `load_raw(train_path, test_path) -> tuple[DataFrame, DataFrame]` — reads both CSVs, calls `validate_raw` on each
- `clean_train(df) -> DataFrame` — drops all-null row, casts numerics to `Int64`/`float64`, calls `validate_clean`
- `get_feature_target(df) -> tuple[DataFrame, Series]` — drops `CustomerID`, separates `Churn`
- `split_train_val(X, y, val_size=0.2, random_state=42) -> tuple[X_tr, X_val, y_tr, y_val]` — stratified split

---

## Stage 3 — Preprocessing (`src/features/preprocessing.py`)

Single `ColumnTransformer` with three transformers, returned by `build_preprocessor()`:

| Transformer | Columns | Steps |
|---|---|---|
| `numeric` | Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction | `SimpleImputer(median)` → `RobustScaler` |
| `gender` | Gender | `SimpleImputer(most_frequent)` → `OrdinalEncoder([['Female','Male']])` |
| `ordinal_cat` | Subscription Type, Contract Length | `SimpleImputer(most_frequent)` → `OrdinalEncoder([['Basic','Standard','Premium'],['Monthly','Quarterly','Annual']])` |

`remainder='drop'` — discards CustomerID.

**Key decisions:**
- `RobustScaler` over `StandardScaler`: Support Calls and Payment Delay have floor effects and potential outliers; RobustScaler (median/IQR) is more robust
- `OrdinalEncoder` for Subscription Type: tier ordering (Basic < Standard < Premium) is real business meaning
- `OrdinalEncoder` for Contract Length: commitment duration (Monthly < Quarterly < Annual) is natural ordinal
- Imputers are defensive — training data has no partial nulls but future inference data might

**Additional function:** `get_feature_names(fitted_preprocessor) -> list[str]` — calls `get_feature_names_out()`, strips prefixes for SHAP

---

## Stage 4 — Three Candidate Models

### Model 1: Logistic Regression (baseline)
- Estimator: `LogisticRegression(class_weight='balanced', solver='saga', random_state=42)`
- Tuning params: `C` [0.01, 0.1, 1.0, 10, 100], `penalty` [l1, l2], `max_iter` [500, 1000]
- Purpose: Fast, interpretable performance floor

### Model 2: Random Forest
- Estimator: `RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)`
- Tuning params: `n_estimators` [100,200,300], `max_depth` [None,10,20,30], `min_samples_split` [2,5,10], `min_samples_leaf` [1,2,4], `max_features` [sqrt,log2], `class_weight` [balanced,balanced_subsample]
- Purpose: Non-linear interactions, robust to feature scale

### Model 3: XGBoost
- Estimator: `XGBClassifier(tree_method='hist', eval_metric='auc', random_state=42)`
- Tuning params: `n_estimators` [100,200,500], `max_depth` [3,5,7,9], `learning_rate` [0.01,0.05,0.1,0.2], `subsample` [0.6,0.8,1.0], `colsample_bytree` [0.6,0.8,1.0], `reg_alpha` [0,0.1,1.0], `reg_lambda` [1.0,5.0,10.0], `scale_pos_weight` [0.763, 1.0]
- Purpose: SOTA on tabular data; native SHAP support via TreeExplainer
- Note: `scale_pos_weight=0.763` = retained_count/churn_count (mild imbalance correction)

All hyperparameter grids stored in `configs/model_params.yaml` with `classifier__` prefix to match sklearn Pipeline naming.

---

## Stage 5 — Training (`src/models/train.py`)

**Functions:**
- `build_full_pipeline(preprocessor, estimator) -> Pipeline` — wraps into `Pipeline([('preprocessor', ...), ('classifier', ...)])`
- `run_cross_validation(pipeline, X_train, y_train, cv=5) -> dict` — `StratifiedKFold`, scores roc_auc/f1/average_precision, returns mean ± std
- `tune_model(pipeline, param_grid, X_train, y_train, n_iter=20, cv=5) -> RandomizedSearchCV` — `RandomizedSearchCV(scoring='roc_auc', refit=True, n_jobs=-1)`, returns fitted search object
- `load_params_from_yaml(model_name) -> dict` — reads `configs/model_params.yaml`

**Anti-leakage guarantee:** The sklearn Pipeline fits the preprocessor only inside `pipeline.fit(X_train, y_train)`. Val and test sets only call `pipeline.predict/predict_proba` which internally calls `transform` using training-fit statistics.

---

## Stage 6 — Evaluation (`src/models/evaluate.py`)

**Metric functions:**
- `compute_metrics(y_true, y_proba, y_pred, threshold=0.5) -> dict` — keys: roc_auc, f1, precision, recall, avg_precision
- `compute_confusion_matrix(y_true, y_pred) -> np.ndarray`
- `plot_confusion_matrix(cm, output_path) -> str` — matplotlib heatmap, saves PNG, returns path
- `plot_roc_curve(y_true, y_proba, output_path) -> str`
- `evaluate_on_holdout(pipeline, X_test, y_test) -> dict` — full evaluation on final holdout set

**SHAP functions:**
- `compute_shap_values(pipeline, X_val) -> tuple[np.ndarray, np.ndarray]` — `TreeExplainer` for RF/XGB, `LinearExplainer` for LR; returns (shap_values for class=1, X_transformed)
- `plot_shap_summary(shap_values, X_transformed, feature_names, output_path) -> str`
- `plot_shap_bar(shap_values, feature_names, output_path) -> str`
- `get_top_features(shap_values, feature_names, n=10) -> list[tuple[str, float]]`

**Primary metric:** ROC-AUC (threshold-independent, robust to mild imbalance). Secondary: F1, Precision, Recall, Average Precision.

---

## Stage 7 — MLflow Tracking (`src/utils/mlflow_helpers.py`)

**One experiment per model type:** `churn_logistic_regression`, `churn_random_forest`, `churn_xgboost`

**Functions:**
- `setup_experiment(model_name) -> str` — sets tracking URI to `mlruns/`, creates/gets experiment, returns experiment ID
- `log_run(run_name, params, metrics, artifacts, model, model_name) -> str` — logs params, metrics, artifacts; optionally registers model; returns run ID
- `register_best_model(run_id, model_name='ChurnPredictionModel') -> None` — registers in Model Registry, transitions to Staging
- `get_best_run(experiment_name, metric='roc_auc') -> Run` — queries MLflow for top run by metric

**Autolog:** `mlflow.sklearn.autolog()` for LR and RF; `mlflow.xgboost.autolog()` for XGBoost. Supplements: SHAP plots, confusion matrix, ROC curve logged as artifacts manually.

---

## Stage 8 — Orchestrator (`run_pipeline.py`)

Entry point `main()` execution order:
1. Load + validate raw data
2. Clean training data (drop empty row, cast types)
3. Get features/target for both train and test
4. Stratified train/val split
5. Build preprocessor
6. For each of 3 models:
   - Setup MLflow experiment
   - Enable autolog
   - Build full pipeline
   - Tune with RandomizedSearchCV (fit on X_train only)
   - Evaluate on X_val: metrics + plots + SHAP
   - Log everything to MLflow
7. Compare 3 models by val ROC-AUC, select winner
8. Evaluate winner on holdout test set
9. Register best model in MLflow Model Registry
10. Serialize best pipeline to `outputs/models/best_pipeline.joblib`

---

## Stage 9 — Tests (`tests/`)

**`test_validation.py`:**
- Valid schema passes without error
- All-null row detected then dropped by `clean_train`
- Age=100 raises `SchemaError`
- Gender='Unknown' raises `SchemaError`
- Subscription Type='Gold' raises `SchemaError`

**`test_preprocessing.py`:**
- Preprocessor output shape = 10 columns (7 numeric + 1 gender + 2 ordinal)
- No leakage: val transform uses train-fit statistics
- Ordinal order: Basic=0 < Standard=1 < Premium=2
- Ordinal order: Monthly=0 < Quarterly=1 < Annual=2

**`test_models.py`:**
- LR pipeline trains on 1000-row sample, `predict_proba` returns (n,2) shape
- `compute_metrics` returns all 5 expected keys
- Confusion matrix shape is (2,2)

---

## Implementation Sequence

1. `configs/model_params.yaml`
2. `src/data/validation.py` + `tests/test_validation.py`
3. `src/data/loader.py`
4. `src/features/preprocessing.py` + `tests/test_preprocessing.py`
5. `src/models/evaluate.py`
6. `src/models/train.py` + `tests/test_models.py`
7. `src/utils/mlflow_helpers.py`
8. `run_pipeline.py`

---

## Verification

```bash
# Activate environment
source .venv/Scripts/activate

# Run all tests
pytest tests/ -v

# Lint
ruff check src/ tests/ run_pipeline.py

# Full pipeline run (logs to mlruns/)
python run_pipeline.py

# View MLflow UI
mlflow ui
# Open http://localhost:5000 — check 3 experiments, compare ROC-AUC across models

# Inspect best model artifact
ls outputs/models/best_pipeline.joblib
ls outputs/artifacts/
```

**Expected outputs:**
- All pytest tests pass
- MLflow UI shows 3 experiments with logged params, metrics, confusion matrices, SHAP plots
- Best model registered in Model Registry under `ChurnPredictionModel`
- `best_pipeline.joblib` loadable and able to predict on new data
