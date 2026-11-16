"""Prefect flows: the monthly validate -> score -> monitor cycle and the retrain feedback loop."""

from datetime import date
from pathlib import Path

from prefect import flow, task
from pydantic import BaseModel

from credit_mlops.data.contract import ContractResult, validate_applications
from credit_mlops.data.lake import load_applications
from credit_mlops.monitoring.report import MonitoringReport, monitor_month
from credit_mlops.months import add_months, month_label, month_range, parse_month
from credit_mlops.registry.gate import GateDecision, promote
from credit_mlops.registry.logging import ModelRef, log_and_register
from credit_mlops.scoring import score_month
from credit_mlops.settings import Settings, TrainingWindow
from credit_mlops.training.train import train_model


class CycleResult(BaseModel):
    month: str
    status: str
    scored: bool
    retrained: ModelRef | None = None
    gate: GateDecision | None = None
    report: MonitoringReport


@task
def validate_task(month: date) -> ContractResult:
    return validate_applications(load_applications(Settings(), [month]))


@task
def score_task(month: date) -> Path:
    return score_month(Settings(), month)


@task
def monitor_task(month: date) -> MonitoringReport:
    return monitor_month(Settings(), month)


@task
def train_task(as_of: date) -> ModelRef:
    settings = Settings()
    window = TrainingWindow.rolled(as_of, half_life_months=settings.retrain_half_life_months)
    return log_and_register(settings, train_model(settings, window), repo_root=Path.cwd())


@task
def promote_task() -> GateDecision:
    return promote(Settings())


def next_streak(streak: int, performance_alert: bool, status: str) -> int:
    """An alert adds one, a blocked month says nothing, any other month resets."""
    if performance_alert:
        return streak + 1
    return streak if status == "block" else 0


def should_retrain(streak: int, settings: Settings) -> bool:
    return streak >= settings.retrain_after_consecutive_alerts


def _previous_month() -> date:
    return add_months(date.today().replace(day=1), -1)


@flow
def monthly_cycle(month: str | None = None, alert_streak: int = 0) -> tuple[CycleResult, int]:
    """Validate, score and monitor one month; retrain after enough consecutive alerts.

    `month=None` means the previous calendar month, resolved when the run starts.
    """
    settings = Settings()
    cycle_month = parse_month(month) if month else _previous_month()
    label = month_label(cycle_month)

    contract = validate_task(cycle_month)
    scored = False
    if contract.success:
        score_task(cycle_month)
        scored = True
    report = monitor_task(cycle_month)

    alert_streak = next_streak(alert_streak, report.performance_alert, report.status)

    retrained: ModelRef | None = None
    gate: GateDecision | None = None
    if should_retrain(alert_streak, settings):
        retrained = train_task(cycle_month)
        gate = promote_task()
        alert_streak = 0

    result = CycleResult(
        month=label,
        status=report.status,
        scored=scored,
        retrained=retrained,
        gate=gate,
        report=report,
    )
    return result, alert_streak


@flow
def simulate_production(start: str, end: str) -> list[CycleResult]:
    """Run `monthly_cycle` over a range of months, carrying the alert streak forward."""
    results: list[CycleResult] = []
    streak = 0
    for month in month_range(parse_month(start), parse_month(end)):
        result, streak = monthly_cycle(month_label(month), streak)
        results.append(result)
    return results
