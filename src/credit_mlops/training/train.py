"""Time-aware HPO, refit and out-of-time evaluation (pure: no MLflow)."""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from credit_mlops.data.lake import attach_labels, load_applications, load_outcomes
from credit_mlops.features import build_pipeline
from credit_mlops.months import month_range
from credit_mlops.schema import FEATURES, SEGMENT_COLUMN, TARGET
from credit_mlops.settings import Settings, TrainingWindow
from credit_mlops.training.evaluate import EvalMetrics, evaluate, recency_weights


@dataclass
class TrainResult:
    pipeline: Pipeline
    best_params: dict[str, int]
    hpo_results: list[dict]
    oot_metrics: EvalMetrics
    window: TrainingWindow
    reference: pd.DataFrame


def _fit(
    settings: Settings,
    window: TrainingWindow,
    data: pd.DataFrame,
    n_estimators: int,
    max_depth: int,
) -> Pipeline:
    y = data[TARGET]
    pipeline = build_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=settings.learning_rate,
        # No class reweighting: the output must stay a calibrated PD.
        scale_pos_weight=1.0,
        seed=settings.seed,
    )
    fit_params = {}
    if window.half_life_months is not None:
        fit_params["model__sample_weight"] = recency_weights(
            data["origination_month"], data["origination_month"].max(), window.half_life_months
        )
    return pipeline.fit(data, y, **fit_params)


def train_model(settings: Settings, window: TrainingWindow) -> TrainResult:
    outcomes = load_outcomes(settings)

    def load(start, end) -> pd.DataFrame:
        apps = load_applications(settings, month_range(start, end))
        return attach_labels(apps, outcomes, window.as_of, window.label_window_months)

    train = load(window.train_start, window.train_end)
    oot = load(window.oot_start, window.oot_end)

    fit_part = train[train["origination_month"] < window.validation_start]
    valid_part = train[train["origination_month"] >= window.validation_start]

    hpo_results = []
    for n_estimators in settings.hpo_n_estimators:
        for max_depth in settings.hpo_max_depth:
            model = _fit(settings, window, fit_part, n_estimators, max_depth)
            auc = roc_auc_score(valid_part[TARGET], model.predict_proba(valid_part)[:, 1])
            hpo_results.append(
                {"n_estimators": n_estimators, "max_depth": max_depth, "validation_auc": float(auc)}
            )
    best = max(
        hpo_results,
        key=lambda r: (r["validation_auc"], -r["n_estimators"], -r["max_depth"]),
    )
    best_params = {"n_estimators": best["n_estimators"], "max_depth": best["max_depth"]}

    pipeline = _fit(settings, window, train, **best_params)

    oot_prob = pipeline.predict_proba(oot)[:, 1]
    oot_metrics = evaluate(oot[TARGET].to_numpy(), oot_prob, oot[SEGMENT_COLUMN])

    reference = train.sample(
        n=min(settings.reference_sample_size, len(train)), random_state=settings.seed
    )[FEATURES].reset_index(drop=True)
    reference["probability_default"] = pipeline.predict_proba(reference)[:, 1]

    return TrainResult(pipeline, best_params, hpo_results, oot_metrics, window, reference)
