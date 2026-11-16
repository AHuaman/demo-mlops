"""End-to-end: six production months through the Prefect cycle on a small full-timeline lake."""

import pytest

from credit_mlops.flows import simulate_production
from credit_mlops.settings import Settings

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def production_env(tmp_path_factory: pytest.TempPathFactory, production_environment):
    with production_environment(tmp_path_factory.mktemp("flows"), loans_per_month=800) as settings:
        yield settings


def test_six_months_are_scored_without_retraining(production_env: Settings) -> None:
    results = simulate_production("2023-01", "2023-06")
    assert [r.month for r in results] == [f"2023-0{i}" for i in range(1, 7)]
    assert all(r.scored for r in results)
    assert all(r.retrained is None for r in results)
