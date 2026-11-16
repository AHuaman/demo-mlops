import pandas as pd
from sklearn.pipeline import Pipeline

from credit_mlops.schema import CATEGORICAL_FEATURES
from credit_mlops.training.train import train_model


def test_train_model_on_small_lake(small_settings, small_window):
    r = train_model(small_settings, small_window)
    assert len(r.hpo_results) == 2 and r.best_params["n_estimators"] in (50, 100)
    assert r.oot_metrics.roc_auc > 0.70  # quality floor: clearly better than chance
    assert set(r.oot_metrics.by_segment) == set(CATEGORICAL_FEATURES["region"])
    assert len(r.reference) == 500 and "probability_default" in r.reference


def test_hpo_selection_never_sees_oot(small_settings, small_window, monkeypatch):
    fitted_months: list[pd.Series] = []
    original_fit = Pipeline.fit

    def spy(self, X, y=None, **params):
        fitted_months.append(X["origination_month"])
        return original_fit(self, X, y, **params)

    monkeypatch.setattr(Pipeline, "fit", spy)
    train_model(small_settings, small_window)
    assert len(fitted_months) == 3  # 2 grid points + the refit
    for months in fitted_months:
        assert months.max() < small_window.oot_start


def test_recency_weights_reach_every_fit(small_settings, small_window, monkeypatch):
    seen: list[float] = []
    original_fit = Pipeline.fit

    def spy(self, X, y=None, **params):
        seen.append(float(params["model__sample_weight"].max()))
        return original_fit(self, X, y, **params)

    monkeypatch.setattr(Pipeline, "fit", spy)
    window = small_window.model_copy(update={"half_life_months": 3.0})
    train_model(small_settings, window)
    assert seen == [1.0, 1.0, 1.0]
