"""Request and response models of the scoring API.

`Application` is written out field by field so students can read the contract;
`tests/api/test_schema_sync.py` guarantees it matches `credit_mlops.schema`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from credit_mlops.registry.pyfunc import Explanation


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bureau_score: int = Field(ge=300, le=999)
    monthly_income_pen: float = Field(ge=500, le=50000)
    debt_to_income: float = Field(ge=0, le=1.5)
    loan_amount_pen: float = Field(ge=500, le=200000)
    term_months: int = Field(ge=6, le=60)
    age: int = Field(ge=18, le=75)
    months_at_job: int = Field(ge=0, le=480)
    num_active_lenders: int = Field(ge=0, le=10)
    max_days_past_due_12m: int = Field(ge=0, le=360)
    product: Literal["consumer", "small_business", "vehicle"]
    region: Literal["lima", "norte", "sur", "centro", "oriente"]
    channel: Literal["branch", "digital"]


class PredictRequest(BaseModel):
    applications: list[Application] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    probability_default: float
    decision: Literal["approve", "review"]
    explanation: Explanation


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_name: str
    alias: str
    model_version: str
    training_window: dict[str, str]
