"""Shared session-scoped fixtures: one small lake written once per test session."""

from datetime import date

import mlflow
import pytest
from mlflow.tracking._model_registry import utils as registry_utils

from credit_mlops.settings import Settings, TrainingWindow
from credit_mlops.simulation.config import SimulationConfig
from credit_mlops.simulation.generator import simulate, write_lake


@pytest.fixture(scope="session")
def small_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    tmp = tmp_path_factory.mktemp("lake")
    settings = Settings(
        _env_file=None,
        data_dir=tmp / "data",
        mlflow_tracking_uri=f"sqlite:///{tmp.as_posix()}/mlflow.db",
        mlflow_artifact_root=(tmp / "mlartifacts").as_uri(),
        hpo_n_estimators=[50, 100],
        hpo_max_depth=[3],
        reference_sample_size=500,
    )
    cfg = SimulationConfig(start=date(2019, 1, 1), end=date(2022, 12, 1), loans_per_month=400)
    write_lake(simulate(cfg), settings.data_dir)
    return settings


@pytest.fixture(scope="session")
def small_window() -> TrainingWindow:
    return TrainingWindow(
        train_start=date(2019, 1, 1),
        train_end=date(2020, 6, 1),
        validation_months=3,
        oot_start=date(2020, 7, 1),
        oot_end=date(2020, 12, 1),
        as_of=date(2021, 12, 1),
    )


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
