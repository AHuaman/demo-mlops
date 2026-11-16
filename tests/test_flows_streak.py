import pytest

from credit_mlops.flows import next_streak, should_retrain
from credit_mlops.settings import Settings


@pytest.mark.parametrize(
    ("streak", "alert", "status", "expected"),
    [
        (0, True, "retrain", 1),
        (3, True, "investigate", 4),
        (3, False, "block", 3),
        (0, False, "block", 0),
        (3, False, "ok", 0),
        (3, False, "investigate", 0),
        (3, False, "retrain", 0),
    ],
)
def test_next_streak(streak: int, alert: bool, status: str, expected: int) -> None:
    assert next_streak(streak, alert, status) == expected


def test_retrain_fires_at_the_configured_threshold() -> None:
    settings = Settings(_env_file=None, retrain_after_consecutive_alerts=3)
    assert [should_retrain(n, settings) for n in (1, 2, 3, 4)] == [False, False, True, True]
