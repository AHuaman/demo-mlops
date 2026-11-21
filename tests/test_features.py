from datetime import date

import numpy as np
import pandas as pd
import pytest

from credit_mlops.data.lake import attach_labels, load_applications, load_outcomes
from credit_mlops.features import MODEL_FEATURES, build_pipeline
from credit_mlops.months import month_range
from credit_mlops.schema import CATEGORICAL_FEATURES, FEATURES, TARGET


@pytest.fixture(scope="module")
def apps(small_settings) -> pd.DataFrame:
    months = month_range(date(2019, 1, 1), date(2019, 6, 1))
    return load_applications(small_settings, months)


def _pipeline():
    return build_pipeline(
        n_estimators=20, max_depth=3, learning_rate=0.1, scale_pos_weight=1.0, seed=42
    )


def _prefix(apps):
    # the transformer prefix is stateless, so it works without fitting the model
    return _pipeline()[:-1]


def test_output_columns_equal_model_features(apps):
    out = _prefix(apps).transform(apps)
    assert list(out.columns) == MODEL_FEATURES


def test_extra_columns_are_dropped(apps):
    assert "loan_id" in apps.columns
    out = _prefix(apps).transform(apps)
    assert "loan_id" not in out.columns


def test_derived_features_values(apps):
    out = _prefix(apps).transform(apps)
    row, src = out.iloc[0], apps.iloc[0]
    expected_pti = round(src.loan_amount_pen / src.term_months / src.monthly_income_pen, 4)
    expected_lti = round(src.loan_amount_pen / (12 * src.monthly_income_pen), 4)
    assert row.payment_to_income == pytest.approx(expected_pti)
    assert row.loan_to_income == pytest.approx(expected_lti)


def test_unseen_category_becomes_nan(apps):
    bad = apps.head(3).copy()
    bad["region"] = ["lima", "atlantis", "sur"]
    out = _prefix(apps).transform(bad)
    assert out["region"].isna().tolist() == [False, True, False]


def test_categorical_dtypes_carry_full_category_list(apps):
    subset = apps[apps["region"] != "oriente"]
    out = _prefix(apps).transform(subset)
    for col, cats in CATEGORICAL_FEATURES.items():
        assert list(out[col].cat.categories) == list(cats)


def test_feature_names_out(apps):
    pipe = _pipeline()
    assert list(pipe["features"].get_feature_names_out()) == MODEL_FEATURES
    assert list(pipe["categoricals"].get_feature_names_out()) == MODEL_FEATURES
    assert MODEL_FEATURES[: len(FEATURES)] == FEATURES


def test_pipeline_fits_and_predicts_on_small_lake(small_settings):
    months = month_range(date(2019, 1, 1), date(2020, 6, 1))
    labeled = attach_labels(
        load_applications(small_settings, months),
        load_outcomes(small_settings),
        as_of=date(2021, 12, 1),
    )
    pipe = _pipeline().fit(labeled, labeled[TARGET])
    proba = pipe.predict_proba(labeled)
    assert proba.shape == (len(labeled), 2)
    assert np.all((proba >= 0) & (proba <= 1))
