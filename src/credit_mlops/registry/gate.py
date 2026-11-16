"""Promotion gate (challenger vs champion) and rollback."""

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from pydantic import BaseModel

from credit_mlops.data.lake import attach_labels, load_applications, load_outcomes
from credit_mlops.months import month_range, parse_month
from credit_mlops.registry.loading import alias_version
from credit_mlops.registry.logging import ModelRef
from credit_mlops.registry.mlflow_setup import configure_mlflow
from credit_mlops.schema import FEATURES, SEGMENT_COLUMN, TARGET
from credit_mlops.settings import Settings
from credit_mlops.training.evaluate import EvalMetrics, evaluate


class GateDecision(BaseModel):
    promote: bool
    reasons: list[str]
    challenger: EvalMetrics
    champion: EvalMetrics | None


def decide(
    challenger: EvalMetrics, champion: EvalMetrics | None, settings: Settings
) -> GateDecision:
    """Pure gate rules; every failed rule adds a reason."""
    if champion is None:
        return GateDecision(
            promote=True, reasons=["no champion yet"], challenger=challenger, champion=None
        )
    reasons: list[str] = []
    if challenger.roc_auc is None or champion.roc_auc is None:
        reasons.append("AUC unavailable: cannot compare challenger with champion")
    else:
        gain = challenger.roc_auc - champion.roc_auc
        if gain < settings.gate_min_auc_gain:
            reasons.append(
                f"AUC gain {gain:+.4f} is below the required {settings.gate_min_auc_gain:+.4f}"
            )
    ratio = challenger.calibration_ratio
    if ratio is None:
        reasons.append("calibration ratio unavailable for the challenger")
    elif abs(ratio - 1) > settings.calibration_tolerance:
        reasons.append(
            f"calibration ratio {ratio:.3f} is outside 1 +/- {settings.calibration_tolerance}"
        )
    for region, champ_seg in champion.by_segment.items():
        chall_seg = challenger.by_segment.get(region)
        if chall_seg is None or chall_seg.roc_auc is None or champ_seg.roc_auc is None:
            continue
        drop = champ_seg.roc_auc - chall_seg.roc_auc
        if drop > settings.gate_max_region_auc_drop:
            reasons.append(
                f"region {region} AUC drops {drop:.4f}, above the allowed "
                f"{settings.gate_max_region_auc_drop}"
            )
    return GateDecision(
        promote=not reasons, reasons=reasons, challenger=challenger, champion=champion
    )


def _score(settings: Settings, alias_uri: str, oot) -> EvalMetrics:
    model = mlflow.pyfunc.load_model(alias_uri)
    prob = model.predict(oot[FEATURES])["probability_default"].to_numpy()
    return evaluate(oot[TARGET].to_numpy(), prob, oot[SEGMENT_COLUMN])


def promote(settings: Settings) -> GateDecision:
    configure_mlflow(settings, settings.training_experiment)
    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    name = settings.registered_model_name
    challenger = alias_version(client, name, "challenger")
    if challenger is None:
        raise ValueError("no @challenger to promote; run credit-mlops train first")
    champion = alias_version(client, name, "champion")
    if champion == challenger:
        raise ValueError(f"version {challenger} is already champion")

    run_id = client.get_model_version(name, challenger).run_id
    params = client.get_run(run_id).data.params
    oot_months = month_range(parse_month(params["oot_start"]), parse_month(params["oot_end"]))
    label_window = int(params.get("label_window_months", settings.label_window_months))
    oot = attach_labels(
        load_applications(settings, oot_months),
        load_outcomes(settings),
        parse_month(params["as_of"]),
        label_window,
    )

    challenger_metrics = _score(settings, f"models:/{name}@challenger", oot)
    champion_metrics = (
        _score(settings, f"models:/{name}@champion", oot) if champion is not None else None
    )
    decision = decide(challenger_metrics, champion_metrics, settings)

    client.log_text(run_id, decision.model_dump_json(indent=2), "gate_decision.json")
    client.set_tag(run_id, "gate", "passed" if decision.promote else "failed")
    if decision.promote:
        if champion is not None:
            client.set_registered_model_alias(name, "previous_champion", champion)
        client.set_registered_model_alias(name, "champion", challenger)
    return decision


def rollback(settings: Settings) -> ModelRef:
    configure_mlflow(settings, settings.training_experiment)
    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    name = settings.registered_model_name
    previous = alias_version(client, name, "previous_champion")
    if previous is None:
        raise ValueError("no previous_champion to roll back to")
    client.set_registered_model_alias(name, "champion", previous)
    run_id = client.get_model_version(name, previous).run_id
    return ModelRef(name=name, version=previous, run_id=run_id)
