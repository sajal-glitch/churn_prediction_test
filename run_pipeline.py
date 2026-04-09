"""
run_pipeline.py — End-to-end churn prediction pipeline orchestrator.

Usage:
    source .venv/Scripts/activate
    python run_pipeline.py
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna  # noqa: F401 — imported here so the dep is explicit at module level
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data.loader import clean_train, get_feature_target, load_raw, split_train_val
from src.features.preprocessing import build_preprocessor, get_feature_names
from src.models.evaluate import (
    compute_confusion_matrix,
    compute_metrics,
    evaluate_on_holdout,
    get_top_features,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_shap_bar,
    plot_shap_summary,
    compute_shap_values,
)
from src.models.train import (
    build_full_pipeline,
    load_params_from_yaml,
    tune_model,
    tune_model_optuna,
)
from src.utils.mlflow_helpers import (
    log_run,
    make_run_name,
    register_best_model,
    setup_experiment,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw")
TRAIN_PATH = DATA_DIR / "customer_churn_dataset-training-master.csv"
TEST_PATH = DATA_DIR / "customer_churn_dataset-testing-master.csv"
ARTIFACT_DIR = Path("outputs/artifacts")
MODEL_DIR = Path("outputs/models")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        solver="saga",
        random_state=42,
        max_iter=1000,
    ),
    "random_forest": RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "xgboost": XGBClassifier(
        tree_method="hist",
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    ),
}

AUTOLOG = {
    "logistic_regression": mlflow.sklearn.autolog,
    "random_forest": mlflow.sklearn.autolog,
    "xgboost": mlflow.xgboost.autolog,
}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and validate data
    # ------------------------------------------------------------------
    print("Loading data...")
    train_df, test_df = load_raw(TRAIN_PATH, TEST_PATH)

    # ------------------------------------------------------------------
    # 2. Clean training data
    # ------------------------------------------------------------------
    print("Cleaning training data...")
    train_clean = clean_train(train_df)

    # ------------------------------------------------------------------
    # 3. Feature / target separation
    # ------------------------------------------------------------------
    X_full, y_full = get_feature_target(train_clean)
    X_test, y_test = get_feature_target(test_df)

    # ------------------------------------------------------------------
    # 4. Stratified train / validation split
    # ------------------------------------------------------------------
    print("Splitting train/validation...")
    X_train, X_val, y_train, y_val = split_train_val(X_full, y_full)
    print(
        f"  Train: {X_train.shape[0]:,} rows | "
        f"Val: {X_val.shape[0]:,} rows | "
        f"Test: {X_test.shape[0]:,} rows"
    )

    # ------------------------------------------------------------------
    # 5. Train, tune, and evaluate each model
    # ------------------------------------------------------------------
    val_results: dict[str, dict] = {}

    for model_name, estimator in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()}")
        print("=" * 60)

        # Setup MLflow experiment
        setup_experiment(model_name)

        # Enable autolog
        AUTOLOG[model_name](log_models=False, silent=True)

        # Tune — XGBoost uses Optuna TPE; other models use RandomizedSearchCV
        print(f"Tuning {model_name}...")
        if model_name == "xgboost":
            xgb_fixed = {
                "tree_method": "hist",
                "eval_metric": "auc",
                "random_state": 42,
                "verbosity": 0,
            }
            best_pipeline, best_params = tune_model_optuna(
                preprocessor=build_preprocessor(),
                estimator_cls=XGBClassifier,
                estimator_fixed_kwargs=xgb_fixed,
                X_train=X_train,
                y_train=y_train,
                n_trials=50,
                cv=5,
            )
            # Track the number of Optuna trials in MLflow params
            best_params["n_optuna_trials"] = 50
            print(f"  Best Optuna params: {best_params}")
        else:
            pipeline = build_full_pipeline(build_preprocessor(), estimator)
            param_grid = load_params_from_yaml(model_name)
            search = tune_model(pipeline, param_grid, X_train, y_train, n_iter=20, cv=5)
            best_pipeline = search.best_estimator_
            best_params = search.best_params_
            print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
            print(f"  Best params: {best_params}")

        # Evaluate on validation set
        y_proba_val = best_pipeline.predict_proba(X_val)[:, 1]
        y_pred_val = best_pipeline.predict(X_val)
        metrics = compute_metrics(y_val, y_proba_val, y_pred_val)
        cm = compute_confusion_matrix(y_val, y_pred_val)

        print(f"  Val ROC-AUC : {metrics['roc_auc']:.4f}")
        print(f"  Val F1      : {metrics['f1']:.4f}")
        print(f"  Val Precision: {metrics['precision']:.4f}")
        print(f"  Val Recall  : {metrics['recall']:.4f}")

        # Plots
        cm_path = plot_confusion_matrix(
            cm, ARTIFACT_DIR / f"{model_name}_val_cm.png"
        )
        roc_path = plot_roc_curve(
            y_val, y_proba_val, ARTIFACT_DIR / f"{model_name}_val_roc.png"
        )

        # SHAP
        print("  Computing SHAP values...")
        shap_vals, X_transformed = compute_shap_values(best_pipeline, X_val)
        fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
        feature_names = get_feature_names(fitted_preprocessor)

        shap_summary_path = plot_shap_summary(
            shap_vals, X_transformed, feature_names,
            ARTIFACT_DIR / f"{model_name}_shap_summary.png",
        )
        shap_bar_path = plot_shap_bar(
            shap_vals, feature_names,
            ARTIFACT_DIR / f"{model_name}_shap_bar.png",
        )
        top_features = get_top_features(shap_vals, feature_names, n=10)
        print(f"  Top features: {[f for f, _ in top_features[:5]]}")

        # Log to MLflow
        artifact_paths = [cm_path, roc_path, shap_summary_path, shap_bar_path]
        run_id = log_run(
            run_name=make_run_name(model_name),
            params=best_params,
            metrics=metrics,
            artifact_paths=artifact_paths,
            model=best_pipeline,
            model_name=model_name,
            register=False,
        )

        val_results[model_name] = {
            "pipeline": best_pipeline,
            "metrics": metrics,
            "run_id": run_id,
        }

    # ------------------------------------------------------------------
    # 7. Select best model by validation ROC-AUC
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Model comparison (Validation ROC-AUC):")
    for name, result in val_results.items():
        print(f"  {name:25s}: {result['metrics']['roc_auc']:.4f}")

    best_model_name = max(
        val_results, key=lambda n: val_results[n]["metrics"]["roc_auc"]
    )
    best_result = val_results[best_model_name]
    print(f"\nBest model: {best_model_name.upper()}")

    # ------------------------------------------------------------------
    # 8. Evaluate best model on holdout test set
    # ------------------------------------------------------------------
    print("\nEvaluating best model on holdout test set...")
    holdout = evaluate_on_holdout(
        best_result["pipeline"], X_test, y_test,
        artifact_dir=ARTIFACT_DIR, model_name=best_model_name,
    )
    print("  Holdout metrics:")
    for k, v in holdout["metrics"].items():
        print(f"    {k:20s}: {v:.4f}")

    # ------------------------------------------------------------------
    # 9. Register best model in MLflow Model Registry
    # ------------------------------------------------------------------
    print(f"\nRegistering '{best_model_name}' in MLflow Model Registry...")
    register_best_model(best_result["run_id"], model_name="ChurnPredictionModel")

    # ------------------------------------------------------------------
    # 10. Serialize best pipeline
    # ------------------------------------------------------------------
    joblib_path = MODEL_DIR / "best_pipeline.joblib"
    joblib.dump(best_result["pipeline"], joblib_path)
    print(f"Best pipeline saved to: {joblib_path}")
    print("\nDone. Run `mlflow ui` to explore experiment results.")


if __name__ == "__main__":
    main()
