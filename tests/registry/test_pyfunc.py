import json

import joblib
import mlflow.pyfunc
import numpy as np
import pytest
import shap

from credit_mlops.data.lake import attach_labels, load_applications, load_outcomes
from credit_mlops.features import MODEL_FEATURES
from credit_mlops.months import month_range
from credit_mlops.registry.pyfunc import PREDICTION_COLUMNS, CreditRiskModel, explain
from credit_mlops.schema import FEATURES
from credit_mlops.training.train import train_model


@pytest.fixture(scope="module")
def pipeline(small_settings, small_window):
    return train_model(small_settings, small_window).pipeline


@pytest.fixture(scope="module")
def X(small_settings, small_window):
    apps = load_applications(
        small_settings, month_range(small_window.oot_start, small_window.oot_end)
    )
    return attach_labels(
        apps, load_outcomes(small_settings), small_window.as_of, small_window.label_window_months
    )


def max_abs_diff(ours, shap_values, names) -> float:
    """Largest gap between our per-feature contributions and the SHAP matrix."""
    diffs = []
    for explanation, row in zip(ours, shap_values, strict=True):
        by_name = {f.feature: f.contribution for f in explanation.top_factors}
        diffs.extend(abs(by_name[name] - row[i]) for i, name in enumerate(names))
    return max(diffs)


def test_contributions_sum_to_logit(pipeline, X):
    full = explain(pipeline, X.head(20), top_k=len(MODEL_FEATURES))
    p = pipeline.predict_proba(X.head(20))[:, 1]
    logit = np.log(p / (1 - p))
    totals = [e.base_log_odds + sum(f.contribution for f in e.top_factors) for e in full]
    assert totals == pytest.approx(logit, abs=1e-4)


def test_matches_shap_tree_explainer(pipeline, X):
    Xt = pipeline[:-1].transform(X.head(50))
    sv = shap.TreeExplainer(pipeline[-1]).shap_values(Xt)
    ours = explain(pipeline, X.head(50), top_k=len(MODEL_FEATURES))
    assert max_abs_diff(ours, sv, MODEL_FEATURES) < 1e-3


def test_top_k_sorted_by_absolute_value(pipeline, X):
    e = explain(pipeline, X.head(5), top_k=3)[0]
    assert len(e.top_factors) == 3
    magnitudes = [abs(f.contribution) for f in e.top_factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_unseen_category_has_missing_value(pipeline, X):
    row = X.head(1).copy()
    row["region"] = "Atlantis"
    e = explain(pipeline, row, top_k=len(MODEL_FEATURES))[0]
    assert next(f for f in e.top_factors if f.feature == "region").value is None


def test_pyfunc_predict_columns_and_decision(tmp_path, pipeline, X):
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, artifact)
    path = tmp_path / "model"
    mlflow.pyfunc.save_model(
        path=path,
        python_model=CreditRiskModel(),
        artifacts={"pipeline": str(artifact)},
        model_config={"threshold": 0.2, "top_k": 3},
    )
    loaded = mlflow.pyfunc.load_model(path)
    batch = X.head(10)[FEATURES]
    out = loaded.predict(batch)
    assert list(out.columns) == PREDICTION_COLUMNS
    assert out.index.equals(batch.index)
    assert set(out.decision) <= {"approve", "review"}
    assert all(len(json.loads(s)) == 3 for s in out.top_factors)
    expected = np.where(out.probability_default >= 0.2, "review", "approve")
    assert (out.decision == expected).all()
    assert out.probability_default.to_numpy() == pytest.approx(
        pipeline.predict_proba(batch)[:, 1], abs=1e-6
    )
