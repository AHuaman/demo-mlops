import pytest

from credit_mlops.data.contract import (
    ContractResult,
    ContractViolation,
    build_expectation_suite,
    validate_applications,
)
from credit_mlops.data.lake import load_applications
from credit_mlops.months import parse_month


@pytest.fixture(scope="module")
def clean_month(small_settings):
    return load_applications(small_settings, [parse_month("2020-03")])


def test_clean_month_passes(clean_month):
    result = validate_applications(clean_month)
    assert result.success
    assert result.failed == []
    assert result.n_rows == len(clean_month)


def test_income_in_thousands_fails_naming_column(clean_month):
    bad = clean_month.assign(monthly_income_pen=clean_month["monthly_income_pen"] / 1000)
    result = validate_applications(bad)
    assert not result.success
    assert any(
        m.startswith("monthly_income_pen: values outside [500, 50000] (") for m in result.failed
    )
    assert not any(m.startswith("age") for m in result.failed)


def test_unknown_region_fails_naming_column(clean_month):
    bad = clean_month.copy()
    bad.loc[bad.index[:7], "region"] = "lima metropolitana"
    result = validate_applications(bad)
    assert not result.success
    assert any(m.startswith("region: ") and "(7 rows)" in m for m in result.failed)


def test_null_feature_fails(clean_month):
    bad = clean_month.copy()
    bad.loc[bad.index[:3], "age"] = None
    result = validate_applications(bad)
    assert not result.success
    assert any(m.startswith("age: ") for m in result.failed)


def test_float_dtype_for_integer_column_fails(clean_month):
    bad = clean_month.assign(bureau_score=clean_month["bureau_score"].astype(float))
    result = validate_applications(bad)
    assert not result.success
    assert any(m.startswith("bureau_score: ") and "integer" in m for m in result.failed)


def test_empty_frame_fails(clean_month):
    result = validate_applications(clean_month.iloc[0:0])
    assert not result.success
    assert result.n_rows == 0
    assert any("row count" in m for m in result.failed)


def test_missing_column_fails_naming_column(clean_month):
    result = validate_applications(clean_month.drop(columns="channel"))
    assert not result.success
    assert any(m.startswith("channel: ") and "missing" in m for m in result.failed)


def test_contract_violation_carries_result():
    result = ContractResult(success=False, n_rows=0, failed=["x: bad"])
    error = ContractViolation(result)
    assert error.result is result
    assert "x: bad" in str(error)


def test_suite_is_generated_from_schema():
    suite = build_expectation_suite()
    columns = {e.column for e in suite.expectations if hasattr(e, "column")}
    assert {"loan_id", "origination_month", "monthly_income_pen", "region"} <= columns
