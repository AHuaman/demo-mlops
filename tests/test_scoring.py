import shutil
from datetime import date

import pandas as pd
import pytest
from mlflow import MlflowClient
from typer.testing import CliRunner

from credit_mlops.cli import app
from credit_mlops.data.contract import ContractViolation
from credit_mlops.prediction_log import PredictionLogger
from credit_mlops.registry.gate import promote
from credit_mlops.registry.loading import ModelUnavailableError, load_champion
from credit_mlops.registry.logging import log_and_register
from credit_mlops.registry.pyfunc import PREDICTION_COLUMNS
from credit_mlops.scoring import score_month
from credit_mlops.settings import Settings
from credit_mlops.simulation.config import IncidentConfig, SimulationConfig
from credit_mlops.simulation.generator import simulate, write_lake
from credit_mlops.training.train import train_model


@pytest.fixture
def registry_settings(small_settings, tmp_path) -> Settings:
    """Fresh registry and a private copy of the small lake (predictions land under tmp_path)."""
    shutil.copytree(small_settings.data_dir / "raw", tmp_path / "data" / "raw")
    return small_settings.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "mlflow_tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "mlflow_artifact_root": (tmp_path / "mlartifacts").as_uri(),
        }
    )


@pytest.fixture
def promoted_settings(registry_settings, small_window) -> Settings:
    settings = registry_settings
    log_and_register(settings, train_model(settings, small_window), repo_root=settings.data_dir)
    promote(settings)
    return settings


def test_score_month_writes_predictions_with_champion_version(promoted_settings):
    settings = promoted_settings
    path = score_month(settings, date(2021, 3, 1))
    df = pd.read_parquet(path)
    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    champion = str(
        client.get_model_version_by_alias(settings.registered_model_name, "champion").version
    )
    assert path == settings.predictions_dir / "batch" / "2021-03.parquet"
    assert list(df.columns) == [
        "loan_id",
        "origination_month",
        "model_version",
        "scored_at",
        *PREDICTION_COLUMNS,
    ]
    assert len(df) == 400
    assert (df["model_version"] == champion).all()
    assert df["scored_at"].nunique() == 1 and df["scored_at"].dt.tz is not None
    assert set(df["decision"]) <= {"approve", "review"}


def test_load_champion_reports_window_and_threshold(promoted_settings, small_window):
    loaded = load_champion(promoted_settings)
    assert loaded.threshold == promoted_settings.decision_threshold
    assert loaded.window["train_start"] == "2019-01"
    assert loaded.window["oot_end"] == "2020-12"
    assert loaded.ref.name == promoted_settings.registered_model_name


def test_load_champion_without_champion_points_to_promote(registry_settings):
    with pytest.raises(ModelUnavailableError, match="credit-mlops promote"):
        load_champion(registry_settings)


def test_load_champion_with_unregistered_model_points_to_promote(registry_settings):
    absent = registry_settings.model_copy(update={"registered_model_name": "never_registered"})
    with pytest.raises(ModelUnavailableError, match="credit-mlops promote"):
        load_champion(absent)


def test_load_champion_unreachable_server_fails_fast(small_settings):
    dead = small_settings.model_copy(update={"mlflow_tracking_uri": "http://127.0.0.1:1"})
    with pytest.raises(ModelUnavailableError, match="credit-mlops mlflow-server") as raised:
        load_champion(dead)
    assert "http://127.0.0.1:1" in str(raised.value)


def test_missing_month_raises_before_needing_a_server(small_settings):
    dead = small_settings.model_copy(update={"mlflow_tracking_uri": "http://127.0.0.1:1"})
    with pytest.raises(FileNotFoundError, match="2031-01"):
        score_month(dead, date(2031, 1, 1))


def test_income_bug_month_violates_contract(small_settings, tmp_path):
    cfg = SimulationConfig(
        start=date(2019, 1, 1),
        end=date(2019, 6, 1),
        loans_per_month=60,
        incidents=IncidentConfig(income_in_thousands_from=date(2019, 5, 1)),
    )
    write_lake(simulate(cfg), tmp_path / "bug")
    # the contract runs before the champion is loaded, so no server is needed
    settings = small_settings.model_copy(
        update={"data_dir": tmp_path / "bug", "mlflow_tracking_uri": "http://127.0.0.1:1"}
    )
    with pytest.raises(ContractViolation, match="monthly_income_pen"):
        score_month(settings, date(2019, 5, 1))


def test_cli_score_missing_month_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_MLOPS_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["score", "--month", "2031-01"])
    assert result.exit_code == 1
    assert "2031-01" in result.stdout


def test_cli_score_prints_path_and_decision_counts(promoted_settings, monkeypatch):
    monkeypatch.setenv("CREDIT_MLOPS_DATA_DIR", str(promoted_settings.data_dir))
    monkeypatch.setenv("CREDIT_MLOPS_MLFLOW_TRACKING_URI", promoted_settings.mlflow_tracking_uri)
    result = CliRunner().invoke(app, ["score", "--month", "2021-03"])
    assert result.exit_code == 0, result.output
    assert "2021-03.parquet" in result.stdout
    assert "approve" in result.stdout


def test_prediction_logger_appends_across_calls(tmp_path):
    logger = PredictionLogger(tmp_path / "nested" / "log.jsonl")
    logger.append([{"loan_id": "a", "p": 0.1}])
    logger.append([{"loan_id": "b", "p": 0.2}])
    lines = (tmp_path / "nested" / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2
