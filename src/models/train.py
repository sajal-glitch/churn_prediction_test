from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline

CONFIGS_PATH = Path(__file__).parents[2] / "configs" / "model_params.yaml"


def build_full_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Wrap preprocessor and estimator into a single sklearn Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def load_params_from_yaml(model_name: str) -> dict:
    """Load hyperparameter grid for a given model from configs/model_params.yaml."""
    with open(CONFIGS_PATH, "r") as f:
        all_params = yaml.safe_load(f)
    if model_name not in all_params:
        raise KeyError(
            f"Model '{model_name}' not found in {CONFIGS_PATH}. "
            f"Available: {list(all_params.keys())}"
        )
    return all_params[model_name]


def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    """Run stratified k-fold CV and return per-fold AND aggregated metrics.

    Returned keys:
      fold_0_roc_auc, fold_1_roc_auc, …   (one per fold, per metric)
      roc_auc_mean, roc_auc_std            (aggregated, per metric)
    Same pattern for f1 and average_precision.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["roc_auc", "f1", "average_precision"]

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}
    for metric in scoring:
        fold_scores = results[f"test_{metric}"]
        for i, score in enumerate(fold_scores):
            summary[f"fold_{i}_{metric}"] = round(float(score), 6)
        summary[f"{metric}_mean"] = round(float(np.mean(fold_scores)), 6)
        summary[f"{metric}_std"] = round(float(np.std(fold_scores)), 6)

    return summary


def tune_model(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 5,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Run RandomizedSearchCV on pipeline, return fitted search object.

    param_grid keys must use 'classifier__' prefix to match the Pipeline step.
    Best estimator is accessible via search.best_estimator_.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv_splitter,
        refit=True,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def tune_model_optuna(
    preprocessor: ColumnTransformer,
    estimator_cls,
    estimator_fixed_kwargs: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
    study_name: str = "xgboost_optuna",
) -> tuple:
    """Tune an estimator with Optuna TPE search, return (best_pipeline, best_params).

    Parameters
    ----------
    preprocessor:
        Unfitted ColumnTransformer — cloned fresh inside every trial.
    estimator_cls:
        Estimator class (e.g. XGBClassifier) — instantiated per trial.
    estimator_fixed_kwargs:
        Keyword arguments forwarded to estimator_cls on every trial (e.g.
        ``{"tree_method": "hist", "eval_metric": "auc", "random_state": 42}``).
    X_train, y_train:
        Training data.
    n_trials:
        Number of Optuna trials.
    cv:
        Number of stratified CV folds used inside each trial.
    random_state:
        Seed for the TPE sampler and CV splitter.
    study_name:
        Name passed to ``optuna.create_study``.

    Returns
    -------
    tuple[Pipeline, dict]
        ``(best_pipeline, best_params)`` where ``best_params`` keys carry the
        ``classifier__`` prefix to match the existing MLflow logging convention.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
        }
        estimator = estimator_cls(**estimator_fixed_kwargs, **trial_params)
        pipeline = build_full_pipeline(preprocessor, estimator)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Refit best params on the full training set
    best_trial_params = study.best_params
    best_estimator = estimator_cls(**estimator_fixed_kwargs, **best_trial_params)
    best_pipeline = build_full_pipeline(preprocessor, best_estimator)
    best_pipeline.fit(X_train, y_train)

    # Return params with classifier__ prefix to match MLflow logging convention
    best_params = {f"classifier__{k}": v for k, v in best_trial_params.items()}

    return best_pipeline, best_params
