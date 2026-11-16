"""Lab acceptance: the monitoring lessons asserted on the full simulated timeline."""

from datetime import date
from pathlib import Path

import pytest

from credit_mlops.flows import CycleResult, monthly_cycle, simulate_production
from credit_mlops.monitoring.report import MonitoringReport
from credit_mlops.months import month_label
from credit_mlops.settings import Settings
from credit_mlops.simulation.config import (
    LAB_BUREAU_SHIFT_MONTH,
    LAB_INCOME_BUG_MONTH,
    IncidentConfig,
    SimulationConfig,
)
from credit_mlops.simulation.generator import simulate, write_lake

pytestmark = pytest.mark.slow

LOANS_PER_MONTH = 2500


@pytest.fixture(scope="module")
def lab_env(tmp_path_factory: pytest.TempPathFactory, production_environment):
    with production_environment(
        tmp_path_factory.mktemp("lab"), loans_per_month=LOANS_PER_MONTH
    ) as settings:
        yield settings


@pytest.fixture(scope="module")
def clean_run(lab_env: Settings) -> list[CycleResult]:
    return simulate_production("2023-01", "2025-12")


@pytest.fixture(scope="module")
def reports(clean_run: list[CycleResult]) -> dict[str, MonitoringReport]:
    return {r.month: r.report for r in clean_run}


def _incident_report(
    lab_env: Settings, tmp: Path, incidents: IncidentConfig, month: date
) -> MonitoringReport:
    """Regenerate the lake with the incident on (same seed), then run one cycle for `month`.

    The MLflow store is shared, so the champion is the one promoted at the end of the clean run.
    """
    data_dir = tmp / "data"
    write_lake(
        simulate(SimulationConfig(loans_per_month=LOANS_PER_MONTH, incidents=incidents)), data_dir
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CREDIT_MLOPS_DATA_DIR", str(data_dir))
        result, _ = monthly_cycle(month_label(month), 0)
    return result.report


@pytest.fixture(scope="module")
def income_bug_reports(
    clean_run: list[CycleResult], lab_env: Settings, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, MonitoringReport]:
    incidents = IncidentConfig(income_in_thousands_from=LAB_INCOME_BUG_MONTH)
    report = _incident_report(
        lab_env, tmp_path_factory.mktemp("income-bug"), incidents, LAB_INCOME_BUG_MONTH
    )
    return {report.month: report}


@pytest.fixture(scope="module")
def bureau_shift_reports(
    clean_run: list[CycleResult], lab_env: Settings, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, MonitoringReport]:
    incidents = IncidentConfig(bureau_shift_from=LAB_BUREAU_SHIFT_MONTH)
    report = _incident_report(
        lab_env, tmp_path_factory.mktemp("bureau-shift"), incidents, LAB_BUREAU_SHIFT_MONTH
    )
    return {report.month: report}


def test_no_performance_alert_before_shock_matures(clean_run: list[CycleResult]) -> None:
    assert not any(r.status == "retrain" for r in clean_run if r.month < "2024-07")


def test_shock_invisible_to_psi_but_caught_by_norte_calibration(
    reports: dict[str, MonitoringReport],
) -> None:
    rep = reports["2024-07"]
    assert all(v < 0.25 for v in rep.feature_psi.values()) and rep.prediction_psi < 0.25
    assert rep.performance_alert and any("norte" in r for r in rep.reasons)


def test_retrain_and_promotion_happen_in_2024_12(clean_run: list[CycleResult]) -> None:
    r = next(r for r in clean_run if r.retrained)
    assert r.month == "2024-12" and r.gate.promote


def test_income_bug_blocks(income_bug_reports: dict[str, MonitoringReport]) -> None:
    assert income_bug_reports["2025-03"].status == "block"


def test_bureau_shift_triggers_investigate(
    bureau_shift_reports: dict[str, MonitoringReport],
) -> None:
    rep = bureau_shift_reports["2025-06"]
    assert rep.feature_psi["bureau_score"] >= 0.25 and rep.status == "investigate"
