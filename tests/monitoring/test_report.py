import shutil
from datetime import date

import pytest
from mlflow import MlflowClient
from typer.testing import CliRunner

from credit_mlops.cli import app
from credit_mlops.monitoring.report import CohortPerformance, decide_status, monitor_month
from credit_mlops.registry.gate import promote
from credit_mlops.registry.logging import log_and_register
from credit_mlops.scoring import score_month
from credit_mlops.settings import Settings
from credit_mlops.simulation.config import IncidentConfig, SimulationConfig
from credit_mlops.simulation.generator import simulate, write_lake
from credit_mlops.training.evaluate import EvalMetrics, SegmentMetrics
from credit_mlops.training.train import train_model


def perf(auc=0.78, cal=1.0, regions=None, z=0.0) -> CohortPerformance:
    metrics = EvalMetrics(
        n=3000,
        default_rate=0.1,
        roc_auc=auc,
        pr_auc=0.3,
        brier=0.08,
        calibration_ratio=cal,
        calibration_z=z,
        by_segment={
            name: SegmentMetrics(
                n=n, default_rate=0.1, roc_auc=0.7, calibration_ratio=ratio, calibration_z=seg_z
            )
            for name, (n, ratio, seg_z) in (regions or {}).items()
        },
    )
    return CohortPerformance(cohorts=["2023-01"], model_version="1", metrics=metrics)


DEGRADED = perf(auc=0.70)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def test_status_precedence(settings):
    block = decide_status(False, {"bureau_score": 0.4}, 0.4, DEGRADED, 0.78, settings)
    assert block[0] == "block"
    investigate = decide_status(True, {"bureau_score": 0.4}, 0.05, DEGRADED, 0.78, settings)
    assert investigate[0] == "investigate"
    assert decide_status(True, {"channel": 0.15}, 0.05, DEGRADED, 0.78, settings)[0] == "retrain"
    status, perf_alert, drift_alert, reasons = decide_status(
        True, {"channel": 0.15}, 0.05, None, 0.78, settings
    )
    assert status == "ok" and not perf_alert and not drift_alert
    assert any("channel" in r and "watch" in r for r in reasons)


def test_prediction_psi_triggers_investigate(settings):
    status, _, drift_alert, reasons = decide_status(True, {}, 0.3, None, 0.78, settings)
    assert status == "investigate" and drift_alert
    assert any("prediction" in r for r in reasons)


def test_auc_drop_within_tolerance_is_ok(settings):
    assert decide_status(True, {}, 0.0, perf(auc=0.76), 0.78, settings)[0] == "ok"


def test_region_calibration_alert(settings):
    status, perf_alert, _, reasons = decide_status(
        True, {}, 0.0, perf(regions={"norte": (1500, 0.55, -9.0)}), 0.78, settings
    )
    assert status == "retrain" and perf_alert
    assert any("norte" in r and "0.55" in r and "z = -9.0" in r for r in reasons)


def test_calibration_outside_tolerance_but_not_significant_is_ok(settings):
    status, perf_alert, _, _ = decide_status(
        True, {}, 0.0, perf(regions={"oriente": (592, 0.74, -2.5)}), 0.78, settings
    )
    assert status == "ok" and not perf_alert


def test_small_region_ignored(settings):
    status, perf_alert, _, _ = decide_status(
        True, {}, 0.0, perf(regions={"oriente": (300, 0.5, -9.0)}), 0.78, settings
    )
    assert status == "ok" and not perf_alert


def test_overall_calibration_alert(settings):
    assert decide_status(True, {}, 0.0, perf(cal=1.3, z=8.0), 0.78, settings)[0] == "retrain"


@pytest.fixture
def monitor_settings(small_settings, small_window, tmp_path) -> Settings:
    shutil.copytree(small_settings.data_dir / "raw", tmp_path / "data" / "raw")
    settings = small_settings.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "mlflow_tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "mlflow_artifact_root": (tmp_path / "mlartifacts").as_uri(),
            "production_start": date(2022, 1, 1),
            "region_min_loans": 100,
        }
    )
    log_and_register(settings, train_model(settings, small_window), repo_root=settings.data_dir)
    promote(settings)
    return settings


def test_monitor_before_labels_mature(monitor_settings):
    score_month(monitor_settings, date(2022, 1, 1))
    report = monitor_month(monitor_settings, date(2022, 1, 1))
    assert report.performance is None and not report.performance_alert
    assert "labels not yet matured for cohorts scored by v1" in report.reasons
    assert report.n_rows == 400 and report.contract_ok
    assert set(report.feature_psi) >= {"bureau_score", "region"}
    assert report.prediction_psi is not None


def test_monitor_pools_matured_cohorts_and_logs_to_mlflow(monitor_settings):
    settings = monitor_settings
    for month in (date(2021, 1, 1), date(2021, 2, 1), date(2021, 3, 1), date(2022, 3, 1)):
        score_month(settings, month)
    report = monitor_month(settings, date(2022, 3, 1))
    assert report.performance is not None
    assert report.performance.cohorts == ["2021-01", "2021-02", "2021-03"]
    assert report.performance.metrics.n == 1200
    # regions below region_min_loans (100 here) are skipped
    by_segment = report.performance.metrics.by_segment
    assert "lima" in by_segment and all(s.n >= 100 for s in by_segment.values())
    client = MlflowClient(settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(settings.monitoring_experiment)
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.run_name == "monitor-2022-03"
    assert run.data.tags["status"] == report.status
    assert run.data.tags["month"] == "2022-03" and run.data.tags["model_version"] == "1"
    assert "matured_roc_auc" in run.data.metrics and "psi__bureau_score" in run.data.metrics
    assert "matured_calibration_ratio__norte" in run.data.metrics
    history = client.get_metric_history(run.info.run_id, "prediction_psi")
    assert [m.step for m in history] == [2]
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert "report.json" in artifacts
    watched = {f for f, v in report.feature_psi.items() if v >= settings.psi_watch}
    drift_files = {a.path for a in client.list_artifacts(run.info.run_id, "drift")}
    assert drift_files == {f"drift/{f}.png" for f in watched}


def test_monitor_ignores_rows_from_other_versions(monitor_settings, small_window):
    settings = monitor_settings
    for month in (date(2021, 1, 1), date(2021, 2, 1), date(2022, 3, 1)):
        score_month(settings, month)
    log_and_register(settings, train_model(settings, small_window), repo_root=settings.data_dir)
    client = MlflowClient(settings.mlflow_tracking_uri)
    client.set_registered_model_alias(settings.registered_model_name, "champion", "2")
    score_month(settings, date(2021, 3, 1))
    report = monitor_month(settings, date(2022, 3, 1))
    assert report.model_version == "2"
    assert report.performance.cohorts == ["2021-03"]
    assert any("800" in r and "v1" in r for r in report.reasons)


def test_monitor_contract_failure_blocks_and_logs(small_settings, small_window, tmp_path):
    cfg = SimulationConfig(
        start=date(2019, 1, 1),
        end=date(2022, 6, 1),
        loans_per_month=60,
        incidents=IncidentConfig(income_in_thousands_from=date(2022, 5, 1)),
    )
    write_lake(simulate(cfg), tmp_path / "bug")
    settings = small_settings.model_copy(
        update={
            "data_dir": tmp_path / "bug",
            "mlflow_tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "mlflow_artifact_root": (tmp_path / "mlartifacts").as_uri(),
            "production_start": date(2022, 1, 1),
        }
    )
    # champion trained on the small lake in the same private registry, monitored on the bug lake
    shutil.copytree(small_settings.data_dir / "raw", tmp_path / "train" / "raw")
    trainer = settings.model_copy(update={"data_dir": tmp_path / "train"})
    log_and_register(trainer, train_model(trainer, small_window), repo_root=tmp_path)
    promote(trainer)
    report = monitor_month(settings, date(2022, 5, 1))
    assert report.status == "block" and not report.contract_ok
    assert report.feature_psi == {} and report.prediction_psi is None
    assert report.performance is None
    assert any("monthly_income_pen" in f for f in report.contract_failed)
    client = MlflowClient(settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(settings.monitoring_experiment)
    assert client.search_runs([experiment.experiment_id])[0].data.tags["status"] == "block"


def test_monitor_missing_batch_file_points_to_score(monitor_settings):
    with pytest.raises(FileNotFoundError, match="credit-mlops score --month 2022-02"):
        monitor_month(monitor_settings, date(2022, 2, 1))


def test_cli_monitor_missing_month_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_MLOPS_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["monitor", "--month", "2031-01"])
    assert result.exit_code == 1
    assert "2031-01" in result.stdout


def test_cli_monitor_prints_status_reasons_and_top_psi(monitor_settings, monkeypatch):
    score_month(monitor_settings, date(2022, 1, 1))
    monkeypatch.setenv("CREDIT_MLOPS_DATA_DIR", str(monitor_settings.data_dir))
    monkeypatch.setenv("CREDIT_MLOPS_MLFLOW_TRACKING_URI", monitor_settings.mlflow_tracking_uri)
    monkeypatch.setenv("CREDIT_MLOPS_PRODUCTION_START", "2022-01-01")
    result = CliRunner().invoke(app, ["monitor", "--month", "2022-01"])
    assert result.exit_code == 0, result.output
    assert "status " in result.stdout
    assert "labels not yet matured" in result.stdout
    assert "Top PSI:" in result.stdout
