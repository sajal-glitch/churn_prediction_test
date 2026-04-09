# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

The virtual environment is at `.venv/`. Activate before running any Python commands:

```bash
source .venv/Scripts/activate   # Windows Git Bash / MSYS2
# or
.venv\Scripts\activate          # Windows cmd/PowerShell
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Note: `requirements.txt` is UTF-16LE encoded. If `pip` can't read it directly:
```bash
iconv -f UTF-16LE -t UTF-8 requirements.txt | pip install -r /dev/stdin
```

## Common Commands

```bash
# Run tests
pytest

# Run a single test file
pytest tests/test_pipeline.py

# Lint and format
ruff check .
ruff format .

# Start Jupyter for notebook development
jupyter lab

# Launch MLflow experiment tracking UI
mlflow ui

# Serve FastAPI app (when implemented)
uvicorn main:app --reload

# Serve Flask app (when implemented)
flask run
```

## Architecture

This is a binary churn classification system (target: `Churn` column, 1 = churned, 0 = retained).

**Data** (`data/raw/`):
- `customer_churn_dataset-training-master.csv` — ~440K rows, 12 features
- `customer_churn_dataset-testing-master.csv` — ~64K rows, same schema

**Features:** `CustomerID`, `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type` (Basic/Standard/Premium), `Contract Length` (Monthly/Quarterly/Annual), `Total Spend`, `Last Interaction`, `Churn`

**Intended ML pipeline:**
1. Data validation — `pandera` schema validation at ingestion
2. Preprocessing — `scikit-learn` pipelines (encoding, scaling)
3. Model training — `scikit-learn` estimators and/or `xgboost`
4. Explainability — `shap` for feature importance
5. Experiment tracking — `mlflow` (log params, metrics, artifacts)
6. Serving — `FastAPI` or `Flask` REST endpoints

**Key technology choices:**
- `mlflow` for experiment tracking and model registry (not ad-hoc logging)
- `pandera` for data schema validation (validate early at data load)
- `shap` for model interpretability alongside metrics
- `joblib` for model serialization
- `pytest` + `ruff` for testing and linting
