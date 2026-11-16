from datetime import date

import pytest

from credit_mlops.months import add_months, month_label, month_range, months_between, parse_month


def test_add_months_crosses_year():
    assert add_months(date(2023, 11, 1), 3) == date(2024, 2, 1)


def test_add_months_negative():
    assert add_months(date(2023, 2, 1), -3) == date(2022, 11, 1)


def test_month_range_inclusive():
    assert len(month_range(date(2019, 1, 1), date(2025, 12, 1))) == 84


def test_months_between():
    assert months_between(date(2021, 11, 1), date(2022, 2, 1)) == 3


def test_parse_month_and_label_roundtrip():
    assert parse_month("2023-07") == date(2023, 7, 1)
    assert month_label(date(2023, 7, 1)) == "2023-07"


def test_parse_month_rejects_bad_format():
    with pytest.raises(ValueError):
        parse_month("2023/07")
