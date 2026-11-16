"""End-to-end: six production months through the Prefect cycle on a small full-timeline lake."""

from pathlib import Path

import mlflow
import pytest
from mlflow.tracking._model_registry import utils as registry_utils
from prefect.testing.utilities import prefect_test_harness

from credit_mlops.flows import simulate_production
from credit_mlops.registry.gate import promote
from credit_mlops.registry.logging import log_and_register
from credit_mlops.settings import DEFAULT_WINDOW, Settings
from credit_mlops.simulation.config import SimulationConfig
from credit_mlops.simulation.generator import simulate, write_lake
from credit_mlops.training.train import train_model

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def production_env(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("flows")
    # The function-scoped autouse restore runs after this fixture, so restore here.
    tracking = mlflow.get_tracking_uri()
    registry = registry_utils._registry_uri
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("MLFLOW_REGISTRY_URI", raising=False)
        mp.setenv("CREDIT_MLOPS_DATA_DIR", str(tmp / "data"))
        mp.setenv("CREDIT_MLOPS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp.as_posix()}/mlflow.db")
        mp.setenv("CREDIT_MLOPS_MLFLOW_ARTIFACT_ROOT", (tmp / "mlartifacts").as_uri())
        settings = Settings()
        write_lake(simulate(SimulationConfig(loans_per_month=800)), settings.data_dir)
        ref = log_and_register(settings, train_model(settings, DEFAULT_WINDOW), Path.cwd())
        assert ref.version
        promote(settings)
        try:
            with prefect_test_harness():
                yield settings
        finally:
            mlflow.set_tracking_uri(tracking)
            registry_utils._registry_uri = registry


def test_six_months_are_scored_without_retraining(production_env: Settings) -> None:
    results = simulate_production("2023-01", "2023-06")
    assert [r.month for r in results] == [f"2023-0{i}" for i in range(1, 7)]
    assert all(r.scored for r in results)
    assert all(r.retrained is None for r in results)
