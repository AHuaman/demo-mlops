"""Monthly monitoring: contract, PSI drift, matured-label performance and the status decision."""

import tempfile
from datetime import date
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.artifacts
import pandas as pd
from mlflow import MlflowClient
from pydantic import BaseModel

from credit_mlops.data.contract import validate_applications
from credit_mlops.data.lake import attach_labels, is_matured, load_applications, load_outcomes
from credit_mlops.monitoring.psi import psi_categorical, psi_numeric
from credit_mlops.months import add_months, month_label, months_between
from credit_mlops.registry.loading import ModelUnavailableError, load_champion
from credit_mlops.registry.mlflow_setup import configure_mlflow
from credit_mlops.schema import CATEGORICAL_FEATURES, FEATURES, SEGMENT_COLUMN, TARGET
from credit_mlops.settings import Settings
from credit_mlops.training.evaluate import EvalMetrics, evaluate

PROBABILITY = "probability_default"


class CohortPerformance(BaseModel):
    cohorts: list[str]
    model_version: str
    metrics: EvalMetrics


class MonitoringReport(BaseModel):
    month: str
    model_version: str
    contract_ok: bool
    contract_failed: list[str]
    n_rows: int
    feature_psi: dict[str, float]
    prediction_psi: float | None
    performance: CohortPerformance | None
    performance_alert: bool
    drift_alert: bool
    status: Literal["ok", "investigate", "retrain", "block"]
    reasons: list[str]


def decide_status(
    contract_ok: bool,
    feature_psi: dict[str, float],
    prediction_psi: float | None,
    performance: CohortPerformance | None,
    baseline_auc: float,
    settings: Settings,
) -> tuple[str, bool, bool, list[str]]:
    """Status precedence: block > investigate (PSI) > retrain (performance) > ok."""
    if not contract_ok:
        return "block", False, False, ["data contract failed"]
    reasons: list[str] = []
    drift_alert = False
    psis = dict(feature_psi)
    if prediction_psi is not None:
        psis["prediction"] = prediction_psi
    for name, value in psis.items():
        if value >= settings.psi_act:
            drift_alert = True
            reasons.append(f"PSI act: {name} = {value:.3f} (>= {settings.psi_act})")
        elif value >= settings.psi_watch:
            reasons.append(f"PSI watch: {name} = {value:.3f} (>= {settings.psi_watch})")

    performance_alert = False
    if performance is not None:
        tol = settings.calibration_tolerance
        metrics = performance.metrics
        if metrics.roc_auc is not None and baseline_auc - metrics.roc_auc > (
            settings.monitor_max_auc_drop
        ):
            performance_alert = True
            reasons.append(
                f"AUC {metrics.roc_auc:.3f} is {baseline_auc - metrics.roc_auc:.3f} below "
                f"the champion's OOT AUC {baseline_auc:.3f} (limit {settings.monitor_max_auc_drop})"
            )
        if metrics.calibration_ratio is not None and abs(metrics.calibration_ratio - 1) > tol:
            performance_alert = True
            reasons.append(
                f"calibration ratio overall = {metrics.calibration_ratio:.2f} outside 1 ± {tol}"
            )
        for region, seg in metrics.by_segment.items():
            if seg.n < settings.region_min_loans or seg.calibration_ratio is None:
                continue
            if abs(seg.calibration_ratio - 1) > tol:
                performance_alert = True
                reasons.append(
                    f"calibration ratio for region {region} = {seg.calibration_ratio:.2f} "
                    f"outside 1 ± {tol} (n={seg.n})"
                )
    status = "investigate" if drift_alert else "retrain" if performance_alert else "ok"
    return status, performance_alert, drift_alert, reasons


def _feature_psi(reference: pd.DataFrame, current: pd.DataFrame, settings: Settings) -> dict:
    out: dict[str, float] = {}
    for feature in FEATURES:
        if feature in CATEGORICAL_FEATURES:
            out[feature] = psi_categorical(reference[feature], current[feature])
        else:
            out[feature] = psi_numeric(
                reference[feature].to_numpy(), current[feature].to_numpy(), bins=settings.psi_bins
            )
    return out


def _matured_performance(
    settings: Settings, month: date, version: str
) -> tuple[CohortPerformance | None, list[str]]:
    """Pool the most recent matured cohorts scored by `version`; extra reasons are informational."""
    window = settings.label_window_months
    candidates = [
        add_months(month, -window - i)
        for i in range(settings.monitor_pool_cohorts)
        if is_matured(add_months(month, -window - i), month, window)
    ]
    batch_dir = settings.predictions_dir / "batch"
    scored = [c for c in sorted(candidates) if (batch_dir / f"{month_label(c)}.parquet").exists()]
    not_matured = f"labels not yet matured for cohorts scored by v{version}"
    if not scored:
        return None, [not_matured]
    batches = pd.concat(
        [pd.read_parquet(batch_dir / f"{month_label(c)}.parquet") for c in scored],
        ignore_index=True,
    )
    reasons: list[str] = []
    other = batches[batches["model_version"] != version]
    if len(other):
        versions = ", ".join(f"v{v}" for v in sorted(other["model_version"].unique()))
        reasons.append(
            f"ignored {len(other)} matured rows scored by other model versions ({versions})"
        )
    batches = batches[batches["model_version"] == version]
    if batches.empty:
        return None, [*reasons, not_matured]
    labeled = attach_labels(
        load_applications(settings, scored), load_outcomes(settings), month, window
    )
    pooled = batches[["loan_id", "origination_month", PROBABILITY]].merge(
        labeled[["loan_id", SEGMENT_COLUMN, TARGET]], on="loan_id", how="inner"
    )
    metrics = evaluate(
        pooled[TARGET].to_numpy(), pooled[PROBABILITY].to_numpy(), pooled[SEGMENT_COLUMN]
    )
    metrics = metrics.model_copy(
        update={
            "by_segment": {
                name: seg
                for name, seg in metrics.by_segment.items()
                if seg.n >= settings.region_min_loans
            }
        }
    )
    cohorts = sorted({month_label(c) for c in pooled["origination_month"]})
    return CohortPerformance(cohorts=cohorts, model_version=version, metrics=metrics), reasons


def _log_run(
    settings: Settings,
    month: date,
    report: MonitoringReport,
    reference: pd.DataFrame | None,
    current: pd.DataFrame | None,
) -> None:
    from credit_mlops.monitoring.plots import plot_feature_drift

    step = months_between(settings.production_start, month)
    metrics: dict[str, float] = {f"psi__{f}": v for f, v in report.feature_psi.items()}
    if report.prediction_psi is not None:
        metrics["prediction_psi"] = report.prediction_psi
    if report.performance is not None:
        m = report.performance.metrics
        if m.roc_auc is not None:
            metrics["matured_roc_auc"] = m.roc_auc
        if m.calibration_ratio is not None:
            metrics["matured_calibration_ratio"] = m.calibration_ratio
        for region, seg in m.by_segment.items():
            if seg.calibration_ratio is not None:
                metrics[f"matured_calibration_ratio__{region}"] = seg.calibration_ratio
    configure_mlflow(settings, settings.monitoring_experiment)
    with mlflow.start_run(run_name=f"monitor-{month_label(month)}"):
        mlflow.set_tags(
            {
                "month": month_label(month),
                "status": report.status,
                "model_version": report.model_version,
            }
        )
        mlflow.log_metrics(metrics, step=step)
        mlflow.log_text(report.model_dump_json(indent=2), "report.json")
        if reference is not None and current is not None:
            import matplotlib.pyplot as plt

            for feature, value in report.feature_psi.items():
                if value >= settings.psi_watch:
                    fig = plot_feature_drift(reference, current, feature)
                    mlflow.log_figure(fig, f"drift/{feature}.png")
                    plt.close(fig)


def _baseline_auc(client: MlflowClient, run_id: str) -> float:
    metrics = client.get_run(run_id).data.metrics
    if "oot_roc_auc" not in metrics:
        raise ModelUnavailableError(f"champion run {run_id} has no 'oot_roc_auc' metric")
    return float(metrics["oot_roc_auc"])


def monitor_month(settings: Settings, month: date) -> MonitoringReport:
    """Monitor one production month against the champion and log it to MLflow."""
    applications = load_applications(settings, [month])  # FileNotFoundError names the month
    champion = load_champion(settings)
    version = champion.ref.version
    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    baseline_auc = _baseline_auc(client, champion.ref.run_id)
    contract = validate_applications(applications)

    reference: pd.DataFrame | None = None
    current: pd.DataFrame | None = None
    feature_psi: dict[str, float] = {}
    prediction_psi: float | None = None
    performance: CohortPerformance | None = None
    info_reasons: list[str] = []
    if contract.success:
        batch_path = settings.predictions_dir / "batch" / f"{month_label(month)}.parquet"
        if not batch_path.exists():
            raise FileNotFoundError(
                f"no batch predictions for {month_label(month)} at {batch_path}; "
                f"run: credit-mlops score --month {month_label(month)}"
            )
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                run_id=champion.ref.run_id,
                artifact_path="reference/reference.parquet",
                dst_path=tmp,
            )
            reference = pd.read_parquet(Path(local))
        batch = pd.read_parquet(batch_path)
        current = applications[FEATURES].reset_index(drop=True)
        feature_psi = _feature_psi(reference, current, settings)
        prediction_psi = psi_numeric(
            reference[PROBABILITY].to_numpy(),
            batch[PROBABILITY].to_numpy(),
            bins=settings.psi_bins,
        )
        performance, info_reasons = _matured_performance(settings, month, version)
        if performance is None and not info_reasons:
            info_reasons = [f"labels not yet matured for cohorts scored by v{version}"]

    status, perf_alert, drift_alert, reasons = decide_status(
        contract.success, feature_psi, prediction_psi, performance, baseline_auc, settings
    )
    if not contract.success:
        reasons = [*reasons, *contract.failed]
    report = MonitoringReport(
        month=month_label(month),
        model_version=version,
        contract_ok=contract.success,
        contract_failed=contract.failed,
        n_rows=contract.n_rows,
        feature_psi=feature_psi,
        prediction_psi=prediction_psi,
        performance=performance,
        performance_alert=perf_alert,
        drift_alert=drift_alert,
        status=status,
        reasons=[*reasons, *info_reasons],
    )
    _log_run(settings, month, report, reference, current)
    return report
