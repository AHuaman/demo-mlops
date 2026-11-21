"""Light MLflow server reachability probe (no mlflow/xgboost imports: the doctor uses it)."""

import urllib.request

HEALTH_TIMEOUT_SECONDS = 3
MLFLOW_SERVER_HINT = "uv run credit-mlops mlflow-server"


class MlflowUnavailableError(RuntimeError):
    """The MLflow server cannot be reached (raised before MLflow's minutes-long retries)."""


def is_http(uri: str) -> bool:
    return uri.startswith(("http://", "https://"))


def unreachable_reason(uri: str) -> str | None:
    """None when `uri` is not http(s) or answers GET <uri>/health; else the error text."""
    if not is_http(uri):
        return None
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=HEALTH_TIMEOUT_SECONDS):
            pass
    except (OSError, ValueError) as error:  # URLError and timeouts are OSError
        return str(error)
    return None


def require_reachable(uri: str, error_class: type[MlflowUnavailableError] = MlflowUnavailableError):
    """Fail fast: MLflow's own retry backoff on a dead server takes minutes."""
    reason = unreachable_reason(uri)
    if reason is not None:
        raise error_class(
            f"MLflow server at {uri} is unreachable ({reason}); start it with: {MLFLOW_SERVER_HINT}"
        )
