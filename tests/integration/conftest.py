"""Shared production-environment setup for the slow integration tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking._model_registry import utils as registry_utils
from prefect.testing.utilities import prefect_test_harness

from credit_mlops.registry.gate import promote
from credit_mlops.registry.logging import log_and_register
from credit_mlops.settings import DEFAULT_WINDOW, Settings
from credit_mlops.simulation.config import SimulationConfig
from credit_mlops.simulation.generator import simulate, write_lake
from credit_mlops.training.train import train_model


@contextmanager
def _production_environment(tmp: Path, loans_per_month: int) -> Iterator[Settings]:
    """Env-configured lake, MLflow store and initial champion, inside a Prefect test harness.

    The function-scoped autouse restore runs after module fixtures, so the MLflow global
    URIs are restored here; the try/finally starts before the first environment change.
    """
    tracking = mlflow.get_tracking_uri()
    registry = registry_utils._registry_uri
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("MLFLOW_REGISTRY_URI", raising=False)
            mp.setenv("CREDIT_MLOPS_DATA_DIR", str(tmp / "data"))
            mp.setenv("CREDIT_MLOPS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp.as_posix()}/mlflow.db")
            mp.setenv("CREDIT_MLOPS_MLFLOW_ARTIFACT_ROOT", (tmp / "mlartifacts").as_uri())
            settings = Settings()
            write_lake(
                simulate(SimulationConfig(loans_per_month=loans_per_month)), settings.data_dir
            )
            ref = log_and_register(settings, train_model(settings, DEFAULT_WINDOW), Path.cwd())
            assert ref.version
            promote(settings)
            with prefect_test_harness():
                yield settings
    finally:
        mlflow.set_tracking_uri(tracking)
        registry_utils._registry_uri = registry


@pytest.fixture(scope="session")
def production_environment():
    """The context manager that builds the production environment (see its docstring)."""
    return _production_environment
