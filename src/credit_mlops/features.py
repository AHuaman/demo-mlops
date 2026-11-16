"""Feature transformers and the scikit-learn pipeline shared by training and serving."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_mlops.schema import CATEGORICAL_FEATURES, FEATURES

DERIVED_FEATURES = ["payment_to_income", "loan_to_income"]
MODEL_FEATURES = [*FEATURES, *DERIVED_FEATURES]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Select the schema features in order and add the derived ratios."""

    def __sklearn_is_fitted__(self) -> bool:
        return True  # stateless: usable without fit

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureBuilder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[FEATURES].copy()
        income = out["monthly_income_pen"]
        out["payment_to_income"] = (out["loan_amount_pen"] / out["term_months"] / income).round(4)
        out["loan_to_income"] = (out["loan_amount_pen"] / (12 * income)).round(4)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(MODEL_FEATURES, dtype=str)


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Cast categoricals to a fixed category list; unseen values become NaN."""

    def __sklearn_is_fitted__(self) -> bool:
        return True  # stateless: usable without fit

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalCaster":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, categories in CATEGORICAL_FEATURES.items():
            known = out[col].where(out[col].isin(categories))  # unseen -> NaN, no pandas warning
            out[col] = known.astype(pd.CategoricalDtype(categories=list(categories)))
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(MODEL_FEATURES, dtype=str)


def build_pipeline(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    scale_pos_weight: float,
    seed: int,
) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("categoricals", CategoricalCaster()),
            (
                "model",
                XGBClassifier(
                    enable_categorical=True,
                    tree_method="hist",
                    n_jobs=4,
                    random_state=seed,
                    eval_metric="logloss",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )
