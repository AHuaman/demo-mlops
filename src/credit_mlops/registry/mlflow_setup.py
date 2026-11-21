"""MLflow connection setup shared by training, the gate and monitoring."""

import mlflow
from mlflow import MlflowClient

from credit_mlops.registry.health import require_reachable
from credit_mlops.settings import Settings


def configure_mlflow(settings: Settings, experiment: str) -> str:
    """Point MLflow at the configured servers and return the experiment id (creating it).

    Raises `MlflowUnavailableError` at once when an http(s) server does not answer.
    """
    for uri in {settings.mlflow_tracking_uri, settings.mlflow_registry_uri or ""}:
        require_reachable(uri)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if settings.mlflow_registry_uri:
        mlflow.set_registry_uri(settings.mlflow_registry_uri)
    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    existing = client.get_experiment_by_name(experiment)
    if existing is not None:
        experiment_id = existing.experiment_id
    else:
        experiment_id = client.create_experiment(
            experiment, artifact_location=settings.mlflow_artifact_root
        )
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id
