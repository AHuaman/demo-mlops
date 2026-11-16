"""Load the raw lake and attach labels only for cohorts whose outcome window has closed."""

from datetime import date

import pandas as pd

from credit_mlops.months import add_months, month_label
from credit_mlops.schema import TARGET
from credit_mlops.settings import Settings


def is_matured(cohort: date, as_of: date, label_window_months: int = 12) -> bool:
    """A cohort's 12-month default label is final once its window has closed by `as_of`."""
    return add_months(cohort, label_window_months) <= as_of


def load_applications(settings: Settings, months: list[date]) -> pd.DataFrame:
    directory = settings.applications_dir
    paths = {m: directory / f"{month_label(m)}.parquet" for m in months}
    missing = [month_label(m) for m, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing application files for months {', '.join(missing)} in {directory}"
        )
    return pd.concat([pd.read_parquet(p) for p in paths.values()], ignore_index=True)


def load_outcomes(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(settings.outcomes_path)


def attach_labels(
    apps: pd.DataFrame,
    outcomes: pd.DataFrame,
    as_of: date,
    label_window_months: int = 12,
) -> pd.DataFrame:
    """Keep matured cohorts only and add `default_12m` (1 if a default_date exists, else 0)."""
    matured = apps["origination_month"].map(
        lambda cohort: is_matured(cohort, as_of, label_window_months)
    )
    labeled = apps[matured].merge(outcomes, on="loan_id", how="left")
    labeled[TARGET] = labeled["default_date"].notna().astype(int)
    return labeled.drop(columns="default_date").reset_index(drop=True)
