"""Configuration of the loan simulator (what to generate, which incidents to inject)."""

from datetime import date

from pydantic import BaseModel, field_validator, model_validator

from credit_mlops.schema import CATEGORICAL_FEATURES

# Named lab incidents, used by the CLI (`--incident income-bug|bureau-shift`).
LAB_INCOME_BUG_MONTH = date(2025, 3, 1)
LAB_BUREAU_SHIFT_MONTH = date(2025, 6, 1)


def _require_first_of_month(name: str, value: date | None) -> None:
    if value is not None and value.day != 1:
        raise ValueError(f"{name} must be the first day of a month, got {value}")


class IncidentConfig(BaseModel):
    """Data incidents injected into the *recorded* features after outcomes are drawn."""

    # From this month on, `monthly_income_pen` is recorded in thousands (a unit bug).
    income_in_thousands_from: date | None = None
    # From this month on, `bureau_score` is shifted by `bureau_shift_points` (a bureau change).
    bureau_shift_from: date | None = None
    bureau_shift_points: int = -80

    @field_validator("income_in_thousands_from", "bureau_shift_from")
    @classmethod
    def _month_start(cls, value: date | None, info) -> date | None:
        _require_first_of_month(info.field_name, value)
        return value


class SimulationConfig(BaseModel):
    seed: int = 42
    start: date = date(2019, 1, 1)
    end: date = date(2025, 12, 1)
    loans_per_month: int = 2500
    # Share of applications coming from the digital channel grows linearly (0.15 to 0.45).
    digital_share_start: float = 0.15
    digital_share_end: float = 0.45
    # From `shock_start`, loans in `shock_region` get riskier (a regional economic shock).
    shock_start: date = date(2023, 7, 1)
    shock_region: str = "norte"
    shock_log_odds: float = 0.9
    incidents: IncidentConfig = IncidentConfig()

    @field_validator("start", "end", "shock_start")
    @classmethod
    def _month_start(cls, value: date, info) -> date:
        _require_first_of_month(info.field_name, value)
        return value

    @field_validator("digital_share_start", "digital_share_end")
    @classmethod
    def _share_in_unit_interval(cls, value: float, info) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{info.field_name} must be in [0, 1], got {value}")
        return value

    @field_validator("shock_region")
    @classmethod
    def _known_region(cls, value: str) -> str:
        regions = CATEGORICAL_FEATURES["region"]
        if value not in regions:
            raise ValueError(f"shock_region must be one of {regions}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _start_not_after_end(self) -> "SimulationConfig":
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must not be after end ({self.end})")
        return self
