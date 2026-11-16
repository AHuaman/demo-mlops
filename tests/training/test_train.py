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


def test_predicted_pd_is_calibrated(small_settings, small_window):
    r = train_model(small_settings, small_window)
    assert abs(r.oot_metrics.calibration_ratio - 1) <= small_settings.calibration_tolerance


def test_hpo_selection_never_sees_oot(small_settings, small_window, monkeypatch):
    fitted_months: list[pd.Series] = []
    scored_months: list[pd.Series] = []
    original_fit = Pipeline.fit
    original_predict = Pipeline.predict_proba

    def fit_spy(self, X, y=None, **params):
        fitted_months.append(X["origination_month"])
        return original_fit(self, X, y, **params)

    def predict_spy(self, X, **params):
        if "origination_month" in X:  # the reference scoring passes features only
            scored_months.append(X["origination_month"])
        return original_predict(self, X, **params)

    monkeypatch.setattr(Pipeline, "fit", fit_spy)
    monkeypatch.setattr(Pipeline, "predict_proba", predict_spy)
    r = train_model(small_settings, small_window)

    n_hpo = len(r.hpo_results)
    assert len(fitted_months) == n_hpo + 1  # grid points + the refit
    # HPO fits use only [train_start, validation_start)
    for months in fitted_months[:-1]:
        assert months.min() == small_window.train_start
        assert months.max() < small_window.validation_start
    # the refit covers the whole train range and nothing from OOT
    assert fitted_months[-1].min() == small_window.train_start
    assert fitted_months[-1].max() == small_window.train_end
    # HPO scores are computed on validation rows only
    for months in scored_months[:n_hpo]:
        assert months.min() == small_window.validation_start
        assert months.max() == small_window.train_end


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
