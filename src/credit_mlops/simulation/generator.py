"""Seeded generator of a synthetic loan book, stored as a monthly "data lake".

Each month is generated from its own RNG (`seed`, year, month), so months are independent
of each other and reproducible given the same config. Note that the config still matters:
the digital-channel share is interpolated between `start` and `end`, so changing either
changes the channel mix of every month. Features are generated first, then the
12-month default outcome is drawn from a logistic score over those features.
"""

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from credit_mlops.months import add_months, month_label, month_range, months_between
from credit_mlops.schema import CATEGORICAL_FEATURES, FEATURES, ID_COLUMNS, NUMERIC_FEATURES
from credit_mlops.simulation.config import IncidentConfig, SimulationConfig

REGION_SHARES = {"lima": 0.45, "norte": 0.20, "sur": 0.15, "centro": 0.12, "oriente": 0.08}
PRODUCT_SHARES = {"consumer": 0.55, "small_business": 0.30, "vehicle": 0.15}

# Intercept of the default score, calibrated so the overall default rate lands near 10%.
# Years after 2023-07 run higher (about 12%) because of the Norte shock and the growing
# digital mix (digital applicants are slightly riskier).
SCORE_INTERCEPT = -3.15


@dataclass
class Lake:
    applications: pd.DataFrame  # ID_COLUMNS + FEATURES, one row per loan
    outcomes: pd.DataFrame  # loan_id, default_date (null when the loan did not default)


def _digital_share(cfg: SimulationConfig, month: date) -> float:
    """Linear growth of the digital channel from the first to the last month."""
    total = months_between(cfg.start, cfg.end)
    progress = months_between(cfg.start, month) / total if total else 0.0
    return cfg.digital_share_start + progress * (cfg.digital_share_end - cfg.digital_share_start)


def _choose(rng: np.random.Generator, shares: dict[str, float], n: int) -> np.ndarray:
    return rng.choice(list(shares), size=n, p=list(shares.values()))


def _generate_features(
    cfg: SimulationConfig, month: date, rng: np.random.Generator
) -> pd.DataFrame:
    n = cfg.loans_per_month
    region = _choose(rng, REGION_SHARES, n)
    product = _choose(rng, PRODUCT_SHARES, n)
    digital = rng.random(n) < _digital_share(cfg, month)

    # Digital applicants are younger, have shorter job tenure and ask for smaller loans.
    age = np.where(digital, rng.normal(32, 7, n), rng.normal(42, 11, n))
    working_years = np.clip(age - 18, 0, None) * rng.uniform(0.1, 0.8, n)
    months_at_job = working_years * 12 * np.where(digital, 0.6, 1.0)

    # Income is log-normal; business owners and Lima residents earn more.
    income_scale = np.where(product == "small_business", 1.3, 1.0) * np.where(
        region == "lima", 1.25, 1.0
    )
    income = rng.lognormal(mean=np.log(2200), sigma=0.5, size=n) * income_scale

    bureau_score = rng.normal(650, 85, n) + 20 * np.log(income / 2200)
    num_active_lenders = rng.poisson(2.5, n)

    # Most applicants have no recent delinquency; the rest have an exponential tail.
    has_delinquency = rng.random(n) < np.clip(0.45 - (bureau_score - 650) / 400, 0.05, 0.9)
    max_days_past_due = np.where(has_delinquency, rng.exponential(35, n), 0)

    debt_to_income = rng.normal(0.30 + 0.04 * num_active_lenders, 0.12, n)

    term_months = np.where(
        product == "vehicle",
        rng.choice([36, 48, 60], n),
        np.where(
            product == "small_business", rng.choice([12, 24, 36], n), rng.choice([6, 12, 24], n)
        ),
    )
    loan_multiple = rng.lognormal(mean=np.log(4.0), sigma=0.5, size=n) * np.where(
        product == "vehicle", 3.0, 1.0
    )
    loan_amount = income * loan_multiple * np.where(digital, 0.6, 1.0)

    frame = pd.DataFrame(
        {
            "loan_id": [f"{month.year:04d}{month.month:02d}-{i:05d}" for i in range(n)],
            "origination_month": [month] * n,
            "bureau_score": bureau_score,
            "monthly_income_pen": income,
            "debt_to_income": debt_to_income,
            "loan_amount_pen": loan_amount,
            "term_months": term_months,
            "age": age,
            "months_at_job": months_at_job,
            "num_active_lenders": num_active_lenders,
            "max_days_past_due_12m": max_days_past_due,
            "product": product,
            "region": region,
            "channel": np.where(digital, "digital", "branch"),
        }
    )
    return _clip_and_round(frame)


def _clip_and_round(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the schema ranges; integers become int64, money 2 decimals, ratios 4."""
    for name, spec in NUMERIC_FEATURES.items():
        values = frame[name].clip(spec.min, spec.max)
        if spec.integer:
            frame[name] = values.round().astype("int64")
        elif name.endswith("_pen"):
            frame[name] = values.round(2)
        else:
            frame[name] = values.round(4)
    for name in CATEGORICAL_FEATURES:
        frame[name] = frame[name].astype(str)
    return frame


def _default_log_odds(
    cfg: SimulationConfig, month: date, rng: np.random.Generator, f: pd.DataFrame
):
    """Log-odds of default within 12 months."""
    z = (f["bureau_score"] - 650) / 100
    score = (
        SCORE_INTERCEPT
        - 0.7 * z
        # Non-linearity: very low bureau scores are disproportionately risky.
        + 1.5 * np.clip(-(z + 0.5), 0, None) ** 2
        # Interaction: leverage hurts small businesses far more than other products.
        + 1.2 * f["debt_to_income"] * (f["product"] == "small_business")
        + 0.6 * f["debt_to_income"]
        + 0.008 * f["max_days_past_due_12m"]
        + 0.07 * f["num_active_lenders"]
        - 0.25 * np.log1p(f["months_at_job"] / 12)
        + 0.5 * f["loan_amount_pen"] / (12 * f["monthly_income_pen"]) / 3
        # Digital applicants are slightly riskier, so the channel shift matters for drift.
        + 0.15 * (f["channel"] == "digital")
    )
    # Month-level macro factor shared by the whole cohort.
    score = score + rng.normal(0, 0.15)
    # Regional shock: cohorts from `shock_start` on in `shock_region` get riskier.
    if month >= cfg.shock_start:
        score = score + cfg.shock_log_odds * (f["region"] == cfg.shock_region)
    return score


def _generate_month(cfg: SimulationConfig, month: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng([cfg.seed, month.year, month.month])
    features = _generate_features(cfg, month, rng)
    probability = 1 / (1 + np.exp(-_default_log_odds(cfg, month, rng, features)))
    defaulted = rng.random(len(features)) < probability
    months_to_default = rng.integers(1, 13, len(features))
    default_date = [
        add_months(month, int(k)) if d else None
        for d, k in zip(defaulted, months_to_default, strict=True)
    ]
    outcomes = pd.DataFrame(
        {"loan_id": features["loan_id"], "default_date": pd.Series(default_date, dtype=object)}
    )
    return features[ID_COLUMNS + FEATURES], outcomes


def _apply_incidents(applications: pd.DataFrame, incidents: IncidentConfig) -> pd.DataFrame:
    applications = applications.copy()
    if incidents.income_in_thousands_from is not None:
        hit = applications["origination_month"] >= incidents.income_in_thousands_from
        # Deliberately not clipped: the point is that the values violate the contract minimum.
        applications.loc[hit, "monthly_income_pen"] = (
            applications.loc[hit, "monthly_income_pen"] / 1000
        ).round(2)
    if incidents.bureau_shift_from is not None:
        hit = applications["origination_month"] >= incidents.bureau_shift_from
        shifted = applications.loc[hit, "bureau_score"] + incidents.bureau_shift_points
        applications.loc[hit, "bureau_score"] = shifted.clip(300, 999).astype(
            applications["bureau_score"].dtype
        )
    return applications


def simulate(cfg: SimulationConfig) -> Lake:
    """Generate the whole lake, then inject `cfg.incidents` into the recorded features.

    Incidents never touch outcomes: they simulate data-quality failures, not real risk changes.
    """
    parts = [_generate_month(cfg, m) for m in month_range(cfg.start, cfg.end)]
    applications = pd.concat([a for a, _ in parts], ignore_index=True)
    outcomes = pd.concat([o for _, o in parts], ignore_index=True)
    applications = _apply_incidents(applications, cfg.incidents)
    return Lake(applications=applications, outcomes=outcomes)


def write_lake(lake: Lake, data_dir: Path) -> None:
    """Write one Parquet file per origination month plus a single outcomes file."""
    raw = data_dir / "raw"
    app_dir = raw / "applications"
    app_dir.mkdir(parents=True, exist_ok=True)
    for stale in app_dir.glob("*.parquet"):
        stale.unlink()
    for month, frame in lake.applications.groupby("origination_month", sort=True):
        frame.to_parquet(app_dir / f"{month_label(month)}.parquet", index=False)
    lake.outcomes.to_parquet(raw / "outcomes.parquet", index=False)


def content_hash(df: pd.DataFrame) -> str:
    """Platform-independent sha256 of the content (row order does not matter)."""
    ordered = df.sort_values("loan_id").reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered, index=True)
    return hashlib.sha256(hashed.to_numpy().astype("<u8").tobytes()).hexdigest()
