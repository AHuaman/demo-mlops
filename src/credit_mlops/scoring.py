"""Batch scoring of one month of applications with the champion model."""

from datetime import UTC, date, datetime
from pathlib import Path

from credit_mlops.data.contract import ContractViolation, validate_applications
from credit_mlops.data.lake import load_applications
from credit_mlops.months import month_label
from credit_mlops.registry.loading import load_champion
from credit_mlops.registry.pyfunc import PREDICTION_COLUMNS
from credit_mlops.schema import FEATURES
from credit_mlops.settings import Settings


def score_month(settings: Settings, month: date) -> Path:
    """Score one month and write `predictions_dir/batch/YYYY-MM.parquet`."""
    applications = load_applications(settings, [month])
    result = validate_applications(applications)
    if not result.success:
        raise ContractViolation(result)
    champion = load_champion(settings)
    predictions = champion.model.predict(applications[FEATURES])
    scored = applications[["loan_id", "origination_month"]].reset_index(drop=True).copy()
    scored["model_version"] = champion.ref.version
    scored["scored_at"] = datetime.now(UTC)
    for column in PREDICTION_COLUMNS:
        scored[column] = predictions[column].to_numpy()
    path = settings.predictions_dir / "batch" / f"{month_label(month)}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(path, index=False)
    return path
