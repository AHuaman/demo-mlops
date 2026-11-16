"""MLflow pyfunc model: calibrated PD, decision and per-row TreeSHAP explanations."""

import json

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

PREDICTION_COLUMNS = ["probability_default", "decision", "base_log_odds", "top_factors"]


class Factor(BaseModel):
    feature: str
    value: float | str | None  # None when the transformed value is missing (e.g. unseen category)
    contribution: float  # log-odds contribution


class Explanation(BaseModel):
    base_log_odds: float
    top_factors: list[Factor]


def _factor_value(value: object) -> float | str | None:
    if pd.isna(value):
        return None
    return float(value) if isinstance(value, (int, float, np.number)) else str(value)


def explain(pipeline: Pipeline, X: pd.DataFrame, top_k: int) -> list[Explanation]:
    """Exact TreeSHAP contributions in log-odds, ranked by absolute value."""
    Xt = pipeline[:-1].transform(X)
    booster = pipeline[-1].get_booster()
    contribs = booster.predict(xgb.DMatrix(Xt, enable_categorical=True), pred_contribs=True)
    names = list(Xt.columns)
    explanations = []
    for row_contribs, (_, row) in zip(contribs, Xt.iterrows(), strict=True):
        order = np.argsort(-np.abs(row_contribs[:-1]), kind="stable")[:top_k]
        factors = [
            Factor(
                feature=names[i],
                value=_factor_value(row.iloc[i]),
                contribution=float(row_contribs[i]),
            )
            for i in order
        ]
        explanations.append(Explanation(base_log_odds=float(row_contribs[-1]), top_factors=factors))
    return explanations


class CreditRiskModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.pipeline: Pipeline = joblib.load(context.artifacts["pipeline"])
        self.threshold: float = float(context.model_config["threshold"])
        self.top_k: int = int(context.model_config["top_k"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        probability = self.pipeline.predict_proba(model_input)[:, 1]
        explanations = explain(self.pipeline, model_input, self.top_k)
        return pd.DataFrame(
            {
                "probability_default": probability,
                "decision": np.where(probability >= self.threshold, "review", "approve"),
                "base_log_odds": [e.base_log_odds for e in explanations],
                "top_factors": [
                    json.dumps([f.model_dump() for f in e.top_factors]) for e in explanations
                ],
            },
            index=model_input.index,
            columns=PREDICTION_COLUMNS,
        )
