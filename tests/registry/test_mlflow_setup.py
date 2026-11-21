import pytest

from credit_mlops.registry.health import MlflowUnavailableError
from credit_mlops.registry.mlflow_setup import configure_mlflow


def test_configure_mlflow_fails_fast_when_the_server_is_down(small_settings):
    settings = small_settings.model_copy(update={"mlflow_tracking_uri": "http://127.0.0.1:1"})
    with pytest.raises(MlflowUnavailableError, match="uv run credit-mlops mlflow-server") as raised:
        configure_mlflow(settings, "any-experiment")
    assert "http://127.0.0.1:1" in str(raised.value)
