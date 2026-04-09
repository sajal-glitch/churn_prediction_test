import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

TRACKING_URI = "mlruns"

_RUN_NAME_PATTERN = re.compile(r"^[a-z0-9_]+__\d{8}_\d{4}__[a-f0-9]{6,}$")


def setup_experiment(model_name: str) -> str:
    """Set MLflow tracking URI and create/get the experiment for model_name.

    Returns the experiment ID.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_name = f"churn_{model_name}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def make_run_name(algo: str) -> str:
    """Build a run name that satisfies rule 1: {algo}__{YYYYmmdd_HHMM}__{git_hash}.

    Falls back to 'nogit' when not inside a git repo.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "nogit"
    return f"{algo}__{timestamp}__{git_hash}"


def _validate_run_name(run_name: str) -> None:
    """Raise ValueError if run_name does not match the required pattern."""
    if not _RUN_NAME_PATTERN.match(run_name):
        raise ValueError(
            f"run_name '{run_name}' does not match required pattern "
            "'<algo>__<YYYYmmdd_HHMM>__<git_hash>'. "
            "Use make_run_name(algo) to generate a compliant name."
        )


def _strip_pipeline_prefix(params: dict) -> dict:
    """Remove sklearn Pipeline step prefixes (e.g. 'classifier__C' → 'C')."""
    return {k.split("__", 1)[-1]: v for k, v in params.items()}


def _log_requirements_snapshot() -> None:
    """Capture pip freeze and log it as an MLflow artifact under environment/."""
    req_txt = subprocess.check_output(["pip", "freeze"]).decode()
    with tempfile.NamedTemporaryFile(
        "w", suffix="_requirements.txt", delete=False
    ) as f:
        f.write(req_txt)
        tmp_path = f.name
    mlflow.log_artifact(tmp_path, artifact_path="environment")


def log_run(
    run_name: str,
    params: dict,
    metrics: dict,
    artifact_paths: list[str],
    model: Pipeline,
    model_name: str,
    local_model_path: str | Path | None = None,
    register: bool = False,
) -> str:
    """Log a compliant MLflow run enforcing all five experiment-logging rules.

    Rules enforced here:
      1. run_name must match '<algo>__<YYYYmmdd_HHMM>__<git_hash>' — raises if not.
      2. params are stripped of Pipeline step prefixes before logging.
      3. metrics are logged as-is; callers must pass per-fold keys (fold_i_*) alongside
         aggregated (*_mean, *_std) — use run_cross_validation() to produce them.
      4. model artifact logged via mlflow.sklearn.log_model; local .joblib also logged
         when local_model_path is provided.
      5. pip freeze snapshot logged automatically.

    Returns the run ID.
    """
    _validate_run_name(run_name)

    with mlflow.start_run(run_name=run_name) as run:
        # rule 2 — strip Pipeline step prefixes
        mlflow.log_params(_strip_pipeline_prefix(params))

        # rule 3 — caller is responsible for providing per-fold keys via metrics dict
        mlflow.log_metrics(metrics)

        # extra artifacts (plots, etc.)
        for path in artifact_paths:
            if Path(path).exists():
                mlflow.log_artifact(path)

        # rule 4a — model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name if register else None,
        )

        # rule 4b — local .joblib artifact
        if local_model_path is not None and Path(local_model_path).exists():
            mlflow.log_artifact(str(local_model_path), artifact_path="model_local")

        # rule 5 — requirements snapshot
        _log_requirements_snapshot()

        run_id = run.info.run_id

    return run_id


def register_best_model(
    run_id: str,
    model_name: str = "ChurnPredictionModel",
) -> None:
    """Register the model from run_id in the MLflow Model Registry and transition to Staging."""
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    print(f"Registered '{model_name}' v{mv.version} → Staging (run {run_id})")


def get_best_run(experiment_name: str, metric: str = "roc_auc") -> mlflow.entities.Run:
    """Return the MLflow Run with the highest value of metric in experiment_name."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")
    return runs[0]
