"""Data contract: Great Expectations checks generated from `schema.py`.

The contract runs on an ephemeral GX 1.x context (nothing is written to disk) and
turns the validation result into short, human-readable messages that name the column.
"""

import os
from functools import cache

# Must be set before great_expectations is imported.
os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")

import great_expectations as gx  # noqa: E402
import pandas as pd  # noqa: E402
from great_expectations.data_context.types.base import ProgressBarsConfig  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from credit_mlops.schema import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COLUMNS,
    NUMERIC_FEATURES,
)

INTEGER_DTYPES = ["int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"]
MIN_ROWS = 1


class ContractResult(BaseModel):
    success: bool
    n_rows: int
    failed: list[str]


class ContractViolation(Exception):
    """Raised by callers that must stop on a failed contract; carries the result."""

    def __init__(self, result: ContractResult) -> None:
        self.result = result
        super().__init__("data contract violated: " + "; ".join(result.failed))


def build_expectation_suite() -> gx.ExpectationSuite:
    expect = gx.expectations
    suite = gx.ExpectationSuite(name="credit_applications_contract")
    suite.add_expectation(expect.ExpectTableRowCountToBeBetween(min_value=MIN_ROWS))
    for column in [*ID_COLUMNS, *FEATURES]:
        suite.add_expectation(expect.ExpectColumnToExist(column=column))
    for column in FEATURES:
        suite.add_expectation(expect.ExpectColumnValuesToNotBeNull(column=column))
    for column, spec in NUMERIC_FEATURES.items():
        suite.add_expectation(
            expect.ExpectColumnValuesToBeBetween(
                column=column, min_value=spec.min, max_value=spec.max
            )
        )
        if spec.integer:
            suite.add_expectation(
                expect.ExpectColumnValuesToBeInTypeList(column=column, type_list=INTEGER_DTYPES)
            )
    for column, allowed in CATEGORICAL_FEATURES.items():
        suite.add_expectation(
            expect.ExpectColumnValuesToBeInSet(column=column, value_set=list(allowed))
        )
    return suite


@cache
def _batch_definition_and_suite():
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False, metric_calculations=False)
    asset = context.data_sources.add_pandas("applications").add_dataframe_asset("applications")
    return asset.add_batch_definition_whole_dataframe("month"), build_expectation_suite()


def _fmt(value: float) -> str:
    return f"{value:g}"


def _describe(result, missing: set[str]) -> str | None:
    config = result.expectation_config
    kind = config.type
    kwargs = config.kwargs
    column = kwargs.get("column")
    if column in missing and kind != "expect_column_to_exist":
        return None  # already reported as a missing column
    detail = result.result or {}
    rows = detail.get("unexpected_count")
    match kind:
        case "expect_table_row_count_to_be_between":
            return (
                f"table: row count {detail.get('observed_value')}, "
                f"expected at least {kwargs['min_value']:g}"
            )
        case "expect_column_to_exist":
            return f"{column}: column is missing"
        case "expect_column_values_to_not_be_null":
            return f"{column}: null values ({rows} rows)"
        case "expect_column_values_to_be_between":
            low, high = _fmt(kwargs["min_value"]), _fmt(kwargs["max_value"])
            return f"{column}: values outside [{low}, {high}] ({rows} rows)"
        case "expect_column_values_to_be_in_type_list":
            return f"{column}: expected integer dtype, got {detail.get('observed_value')}"
        case "expect_column_values_to_be_in_set":
            allowed = ", ".join(kwargs["value_set"])
            return f"{column}: values outside {{{allowed}}} ({rows} rows)"
        case _:
            return f"{column or 'table'}: {kind} failed"


def validate_applications(df: pd.DataFrame) -> ContractResult:
    batch_definition, suite = _batch_definition_and_suite()
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    validation = batch.validate(suite)
    missing = {c for c in [*ID_COLUMNS, *FEATURES] if c not in df.columns}
    failed = [
        message
        for result in validation.results
        if not result.success and (message := _describe(result, missing)) is not None
    ]
    return ContractResult(success=not failed, n_rows=len(df), failed=failed)
