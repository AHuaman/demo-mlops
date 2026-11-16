"""Light MLflow server reachability probe (no mlflow/xgboost imports: the doctor uses it)."""

import urllib.request

HEALTH_TIMEOUT_SECONDS = 3


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
