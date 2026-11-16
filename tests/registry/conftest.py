import mlflow
import pytest
from mlflow.tracking._model_registry import utils as registry_utils


@pytest.fixture(autouse=True)
def restore_mlflow_uris(monkeypatch):
    """configure_mlflow sets global URIs that would outlive the test's tmp_path.

    The registry URI is restored as "unset" when it was unset (it then follows the tracking
    URI); pinning it with get_registry_uri() would make later tests register into a stale DB.
    """
    tracking = mlflow.get_tracking_uri()
    registry = registry_utils._registry_uri
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    yield
    mlflow.set_tracking_uri(tracking)
    registry_utils._registry_uri = registry
