"""Load the champion model from the MLflow registry (light imports: the API image uses this)."""

from dataclasses import dataclass

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from credit_mlops.registry.health import unreachable_reason
from credit_mlops.registry.logging import ModelRef
from credit_mlops.settings import Settings

_MISSING_CODES = {"INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST"}


class ModelUnavailableError(RuntimeError):
    """The registry cannot be reached or holds no model under the requested alias."""


@dataclass
class LoadedModel:
    ref: ModelRef
    model: mlflow.pyfunc.PyFuncModel
    threshold: float
    window: dict[str, str]


def alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    """Version string behind `alias`, or None when the alias (or model) does not exist."""
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException as error:
        # MLflow 3.18 reports a missing alias as INVALID_PARAMETER_VALUE "... not found"
        # (SQLite and REST server alike); Unity Catalog as RESOURCE_DOES_NOT_EXIST
        # "... does not exist". Anything else is a real failure, not "absent".
        message = str(error)
        if error.error_code in _MISSING_CODES and (
            "not found" in message or "does not exist" in message
        ):
            return None
        raise


def _check_server_reachable(uri: str) -> None:
    """Fail fast: MLflow's own retry backoff on a dead server takes minutes."""
    reason = unreachable_reason(uri)
    if reason is not None:
        raise ModelUnavailableError(
            f"MLflow server at {uri} is unreachable ({reason}); "
            "start it with: credit-mlops mlflow-server"
        )


def _is_http(uri: str) -> bool:
    return uri.startswith(("http://", "https://"))


def load_champion(settings: Settings, alias: str = "champion") -> LoadedModel:
    uri = settings.mlflow_tracking_uri
    _check_server_reachable(uri)
    mlflow.set_tracking_uri(uri)
    if settings.mlflow_registry_uri:
        mlflow.set_registry_uri(settings.mlflow_registry_uri)
    client = MlflowClient(uri, settings.mlflow_registry_uri)
    name = settings.registered_model_name
    hint = "; start it with: credit-mlops mlflow-server" if _is_http(uri) else ""
    try:
        version = alias_version(client, name, alias)
        if version is None:
            raise ModelUnavailableError(
                f"no @{alias} model for '{name}' in {uri}; train and promote one with: "
                "credit-mlops train && credit-mlops promote"
            )
        # Load by the resolved version so ref.version and the model cannot disagree.
        run_id = client.get_model_version(name, version).run_id
        model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
        params = dict(client.get_run(run_id).data.params)
    except (MlflowException, OSError) as error:
        raise ModelUnavailableError(
            f"cannot load '{name}' @{alias} from {uri}: {error}{hint}"
        ) from error
    if "threshold" not in (model.model_config or {}):
        raise ModelUnavailableError(
            f"model '{name}' version {version} has no 'threshold' in its model_config"
        )
    window_keys = ("train_start", "train_end", "oot_start", "oot_end", "as_of")
    return LoadedModel(
        ref=ModelRef(name=name, version=version, run_id=run_id),
        model=model,
        threshold=float(model.model_config["threshold"]),
        window={k: params[k] for k in window_keys if k in params},
    )
