"""Feature schema: the single source of truth for the credit-default contract.

The simulator, the data contract, the training pipeline and the API all import
from this module, so a feature is defined in exactly one place.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericSpec:
    min: float
    max: float
    integer: bool


NUMERIC_FEATURES: dict[str, NumericSpec] = {
    "bureau_score": NumericSpec(min=300, max=999, integer=True),
    "monthly_income_pen": NumericSpec(min=500, max=50000, integer=False),
    "debt_to_income": NumericSpec(min=0, max=1.5, integer=False),
    "loan_amount_pen": NumericSpec(min=500, max=200000, integer=False),
    "term_months": NumericSpec(min=6, max=60, integer=True),
    "age": NumericSpec(min=18, max=75, integer=True),
    "months_at_job": NumericSpec(min=0, max=480, integer=True),
    "num_active_lenders": NumericSpec(min=0, max=10, integer=True),
    "max_days_past_due_12m": NumericSpec(min=0, max=360, integer=True),
}

CATEGORICAL_FEATURES: dict[str, tuple[str, ...]] = {
    "product": ("consumer", "small_business", "vehicle"),
    "region": ("lima", "norte", "sur", "centro", "oriente"),
    "channel": ("branch", "digital"),
}

FEATURES: list[str] = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]

ID_COLUMNS = ["loan_id", "origination_month"]
TARGET = "default_12m"
SEGMENT_COLUMN = "region"
