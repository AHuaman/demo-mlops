from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from credit_mlops.simulation.config import (
    LAB_BUREAU_SHIFT_MONTH,
    LAB_INCOME_BUG_MONTH,
    IncidentConfig,
    SimulationConfig,
)
from credit_mlops.simulation.generator import simulate

SHOCK = date(2023, 7, 1)


@pytest.fixture(scope="module")
def lake():
    return simulate(SimulationConfig(loans_per_month=1500))


def _default_rate_by(lake, region, before=None, after=None):
    joined = lake.applications.merge(lake.outcomes, on="loan_id")
    joined = joined[joined["region"] == region]
    if before is not None:
        joined = joined[joined["origination_month"] < before]
    if after is not None:
        joined = joined[joined["origination_month"] >= after]
    return joined["default_date"].notna().mean()


def test_digital_share_grows(lake):
    a = lake.applications
    share = a.groupby(a.origination_month.map(lambda d: d.year))["channel"].apply(
        lambda s: (s == "digital").mean()
    )
    assert share[2019] < 0.22 and share[2025] > 0.38


def test_norte_shock_raises_default_rate_after_start(lake):
    before = _default_rate_by(lake, "norte", before=SHOCK)
    after = _default_rate_by(lake, "norte", after=SHOCK)
    assert after / before > 1.6


def test_shock_leaves_other_regions(lake):
    before = _default_rate_by(lake, "lima", before=SHOCK)
    after = _default_rate_by(lake, "lima", after=SHOCK)
    assert abs(after / before - 1) < 0.2


def test_shock_leaves_norte_features_unchanged(lake):
    # concept drift: same inputs, different outcome
    n = lake.applications[lake.applications.region == "norte"]
    pre, post = n[n.origination_month < SHOCK], n[n.origination_month >= SHOCK]
    assert abs(pre.bureau_score.mean() - post.bureau_score.mean()) < 10


def test_income_bug_only_from_its_month():
    cfg = SimulationConfig(
        start=date(2025, 1, 1),
        end=date(2025, 4, 1),
        loans_per_month=300,
        incidents=IncidentConfig(income_in_thousands_from=date(2025, 3, 1)),
    )
    a = simulate(cfg).applications
    assert a[a.origination_month < date(2025, 3, 1)].monthly_income_pen.min() >= 500
    assert a[a.origination_month >= date(2025, 3, 1)].monthly_income_pen.max() < 60


def test_bureau_shift_only_from_its_month_and_clipped():
    base = SimulationConfig(start=date(2025, 1, 1), end=date(2025, 4, 1), loans_per_month=300)
    shifted = base.model_copy(
        update={"incidents": IncidentConfig(bureau_shift_from=date(2025, 3, 1))}
    )
    a, b = simulate(base).applications, simulate(shifted).applications
    early = a.origination_month < date(2025, 3, 1)
    assert (a.loc[early, "bureau_score"] == b.loc[early, "bureau_score"]).all()
    expected = (a.loc[~early, "bureau_score"] - 80).clip(300, 999)
    assert (b.loc[~early, "bureau_score"] == expected).all()
    assert (b.loc[~early, "bureau_score"] < a.loc[~early, "bureau_score"]).any()


def test_incidents_do_not_change_outcomes():
    base = SimulationConfig(start=date(2025, 1, 1), end=date(2025, 4, 1), loans_per_month=300)
    bug = base.model_copy(update={"incidents": IncidentConfig(bureau_shift_from=date(2025, 2, 1))})
    pd.testing.assert_frame_equal(simulate(base).outcomes, simulate(bug).outcomes)


def test_lab_incident_months_are_first_of_month():
    assert LAB_INCOME_BUG_MONTH == date(2025, 3, 1)
    assert LAB_BUREAU_SHIFT_MONTH == date(2025, 6, 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": date(2025, 2, 1), "end": date(2025, 1, 1)}, "must not be after end"),
        ({"start": date(2019, 1, 15)}, "start must be the first day"),
        ({"end": date(2025, 12, 31)}, "end must be the first day"),
        ({"shock_start": date(2023, 7, 2)}, "shock_start must be the first day"),
        ({"digital_share_start": -0.1}, "digital_share_start must be in"),
        ({"digital_share_end": 1.1}, "digital_share_end must be in"),
        ({"shock_region": "Norte"}, "shock_region must be one of"),
        (
            {"incidents": {"income_in_thousands_from": date(2025, 3, 5)}},
            "income_in_thousands_from must be the first day",
        ),
        (
            {"incidents": {"bureau_shift_from": date(2025, 6, 30)}},
            "bureau_shift_from must be the first day",
        ),
    ],
)
def test_invalid_config_fails_loudly(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        SimulationConfig(**kwargs)
