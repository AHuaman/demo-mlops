from credit_mlops.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    SEGMENT_COLUMN,
)


def test_features_are_numeric_then_categorical() -> None:
    assert len(FEATURES) == 12
    assert FEATURES[:9] == list(NUMERIC_FEATURES)
    assert SEGMENT_COLUMN in CATEGORICAL_FEATURES


def test_income_floor() -> None:
    assert NUMERIC_FEATURES["monthly_income_pen"].min == 5000
