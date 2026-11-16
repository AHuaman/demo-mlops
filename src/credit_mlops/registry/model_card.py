"""Generated model card (Markdown) logged next to every registered model version."""

from mlflow import MlflowClient

from credit_mlops.months import month_label
from credit_mlops.settings import Settings
from credit_mlops.training.evaluate import EvalMetrics
from credit_mlops.training.train import TrainResult


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _metrics_table(metrics: EvalMetrics) -> list[str]:
    lines = [
        "| Scope | Loans | Default rate | ROC AUC | Calibration ratio |",
        "|---|---|---|---|---|",
        f"| overall | {metrics.n} | {_fmt(metrics.default_rate)} | "
        f"{_fmt(metrics.roc_auc)} | {_fmt(metrics.calibration_ratio)} |",
    ]
    for region, seg in metrics.by_segment.items():
        lines.append(
            f"| {region} | {seg.n} | {_fmt(seg.default_rate)} | "
            f"{_fmt(seg.roc_auc)} | {_fmt(seg.calibration_ratio)} |"
        )
    return lines


def render_model_card(settings: Settings, ref, result: TrainResult) -> str:
    window = result.window
    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    tags = client.get_run(ref.run_id).data.tags
    train = f"{month_label(window.train_start)} to {month_label(window.train_end)}"
    oot = f"{month_label(window.oot_start)} to {month_label(window.oot_end)}"
    lines = [
        f"# Model card: {ref.name} v{ref.version}",
        "",
        "## Intended use",
        "",
        "Estimate the 12-month probability of default (PD) of a new consumer loan "
        "application and return a decision at the configured threshold "
        f"({settings.decision_threshold}) with the top {settings.explanation_top_k} "
        "TreeSHAP factors. Teaching model: not for real lending decisions.",
        "",
        "## Training window",
        "",
        f"- Training months: {train} (validation: last {window.validation_months} months)",
        f"- Out-of-time (OOT) months: {oot}",
        f"- Labels observed as of: {month_label(window.as_of)} "
        f"({window.label_window_months}-month default window)",
        f"- Recency half-life (months): {window.half_life_months}",
        "",
        "## Data version",
        "",
        f"- Data version: {tags.get('data_version', 'unknown')}",
        f"- Git commit: {tags.get('git_sha', 'unknown')}",
        f"- Settings hash: {tags.get('settings_hash', 'unknown')}",
        "",
        "## Metrics",
        "",
        f"Computed on the OOT months ({oot}); calibration ratio is mean predicted PD "
        "divided by the observed default rate (1.0 is perfectly calibrated).",
        "",
        *_metrics_table(result.oot_metrics),
        "",
        "## Known limits",
        "",
        "- The data is simulated; it reproduces the structure of a consumer-credit portfolio "
        "but not a real population.",
        f"- The model only saw loans originated in {train}; behavior outside that window "
        "is not validated.",
        f"- Probabilities are calibrated on the OOT window ({oot}) only; calibration "
        "can drift when the population changes.",
        "- Loans younger than the label window have no observed outcome and are not "
        "part of the metrics.",
        "",
    ]
    return "\n".join(lines)
