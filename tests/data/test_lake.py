from datetime import date

import pytest

from credit_mlops.data.lake import attach_labels, is_matured, load_applications, load_outcomes
from credit_mlops.months import month_range


def test_is_matured_boundary():
    assert is_matured(date(2023, 1, 1), date(2024, 1, 1))
    assert not is_matured(date(2023, 2, 1), date(2024, 1, 1))


def test_attach_labels_drops_immature(small_settings):
    apps = load_applications(small_settings, month_range(date(2022, 1, 1), date(2022, 12, 1)))
    labeled = attach_labels(apps, load_outcomes(small_settings), as_of=date(2023, 6, 1))
    assert labeled.origination_month.max() == date(2022, 6, 1)
    assert set(labeled.default_12m.unique()) <= {0, 1}
    assert labeled.default_12m.dtype.kind == "i"


def test_missing_month_message_names_month(small_settings):
    with pytest.raises(FileNotFoundError, match="2031-01"):
        load_applications(small_settings, [date(2031, 1, 1)])
