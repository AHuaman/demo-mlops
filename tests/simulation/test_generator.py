from datetime import date

import pandas as pd
import pytest

from credit_mlops.months import add_months
from credit_mlops.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_mlops.simulation.config import SimulationConfig
from credit_mlops.simulation.generator import content_hash, simulate, write_lake

GOLDEN_HASH = "6fcf8e523081d0e6728f13b9b97f64a8f5d7d82e02275d0aa9e0c7ec07b0b87f"


@pytest.fixture
def cfg() -> SimulationConfig:
    return SimulationConfig(start=date(2019, 1, 1), end=date(2019, 6, 1), loans_per_month=500)


def test_same_seed_same_content(cfg):
    assert content_hash(simulate(cfg).applications) == content_hash(simulate(cfg).applications)


def test_different_seed_different_content(cfg):
    other = cfg.model_copy(update={"seed": 7})
    assert content_hash(simulate(cfg).applications) != content_hash(simulate(other).applications)


def test_values_within_schema(cfg):
    apps = simulate(cfg).applications
    for name, spec in NUMERIC_FEATURES.items():
        assert apps[name].between(spec.min, spec.max).all()
    for name, cats in CATEGORICAL_FEATURES.items():
        assert set(apps[name]) <= set(cats)


def test_default_rate_in_target_band():
    lake = simulate(
        SimulationConfig(start=date(2019, 1, 1), end=date(2019, 12, 1), loans_per_month=2500)
    )
    assert 0.08 <= lake.outcomes["default_date"].notna().mean() <= 0.12


def test_default_rate_on_shipped_config_stays_in_band():
    # Full default period: the digital ramp and the Norte shock both play out.
    lake = simulate(SimulationConfig(loans_per_month=800))
    joined = lake.applications.merge(lake.outcomes, on="loan_id")
    joined["defaulted"] = joined["default_date"].notna()
    assert 0.08 <= joined["defaulted"].mean() <= 0.12
    by_year = joined.groupby(joined["origination_month"].map(lambda m: m.year))["defaulted"].mean()
    assert (by_year <= 0.13).all(), by_year.to_dict()


def test_default_dates_within_12_months(cfg):
    lake = simulate(cfg)
    joined = lake.applications.merge(lake.outcomes, on="loan_id")
    defaults = joined[joined["default_date"].notna()]
    assert len(defaults) > 0
    for origination, default in zip(
        defaults["origination_month"], defaults["default_date"], strict=True
    ):
        assert origination < default <= add_months(origination, 12)


def test_write_lake_writes_one_file_per_month(cfg, tmp_path):
    write_lake(simulate(cfg), tmp_path)
    files = (tmp_path / "raw" / "applications").glob("*.parquet")
    assert sorted(p.stem for p in files) == [
        "2019-01",
        "2019-02",
        "2019-03",
        "2019-04",
        "2019-05",
        "2019-06",
    ]
    assert (tmp_path / "raw" / "outcomes.parquet").exists()


def test_write_lake_roundtrips_and_removes_stale_files(cfg, tmp_path):
    lake = simulate(cfg)
    stale = tmp_path / "raw" / "applications" / "1999-01.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    write_lake(lake, tmp_path)
    assert not stale.exists()
    back = pd.read_parquet(tmp_path / "raw" / "applications" / "2019-03.parquet")
    assert isinstance(back["origination_month"].iloc[0], date)
    assert set(back["origination_month"]) == {date(2019, 3, 1)}
    outcomes = pd.read_parquet(tmp_path / "raw" / "outcomes.parquet")
    assert len(outcomes) == len(lake.outcomes)


def test_golden_content_hash():
    # Cross-platform guard: CI runs this on Ubuntu and Windows.
    small = SimulationConfig(start=date(2019, 1, 1), end=date(2019, 2, 1), loans_per_month=100)
    assert content_hash(simulate(small).applications) == GOLDEN_HASH
