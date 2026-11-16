"""Configuration of the loan simulator (what to generate, which incidents to inject)."""

from datetime import date

from pydantic import BaseModel


class IncidentConfig(BaseModel):
    """Data incidents injected into the *recorded* features after outcomes are drawn."""

    # From this month on, `monthly_income_pen` is recorded in thousands (a unit bug).
    income_in_thousands_from: date | None = None
    # From this month on, `bureau_score` is shifted by `bureau_shift_points` (a bureau change).
    bureau_shift_from: date | None = None
    bureau_shift_points: int = -80


class SimulationConfig(BaseModel):
    seed: int = 42
    start: date = date(2019, 1, 1)
    end: date = date(2025, 12, 1)
    loans_per_month: int = 2500
    # Share of applications coming from the digital channel grows linearly over the period.
    digital_share_start: float = 0.15
    digital_share_end: float = 0.60
    # From `shock_start`, loans in `shock_region` get riskier (a regional economic shock).
    shock_start: date = date(2023, 7, 1)
    shock_region: str = "norte"
    shock_log_odds: float = 0.9
    incidents: IncidentConfig = IncidentConfig()
