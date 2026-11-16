from datetime import date

import pytest
from pydantic import ValidationError

from credit_mlops.settings import DEFAULT_WINDOW, Settings, TrainingWindow


def test_defaults_come_from_toml():
    s = Settings(_env_file=None)
    assert s.decision_threshold == 0.15
    assert s.hpo_n_estimators == [200, 400, 800]
    assert s.production_start == date(2023, 1, 1)
    assert s.outcomes_path == s.data_dir / "raw" / "outcomes.parquet"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("CREDIT_MLOPS_DECISION_THRESHOLD", "0.2")
    assert Settings(_env_file=None).decision_threshold == 0.2


def test_init_kwargs_beat_env(monkeypatch):
    monkeypatch.setenv("CREDIT_MLOPS_SEED", "7")
    assert Settings(_env_file=None, seed=9).seed == 9


def test_databricks_uc_requires_three_level_name():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            mlflow_registry_uri="databricks-uc",
            registered_model_name="credit_default_pd",
        )


def test_settings_hash_is_stable_and_sensitive():
    a, b = Settings(_env_file=None), Settings(_env_file=None, seed=1)
    assert a.settings_hash() == Settings(_env_file=None).settings_hash()
    assert len(a.settings_hash()) == 12
    assert a.settings_hash() != b.settings_hash()


def test_window_rejects_immature_labels():
    with pytest.raises(ValidationError):  # OOT ends 2022-12, labels mature 2023-12 > as_of
        TrainingWindow(
            train_start=date(2020, 1, 1),
            train_end=date(2022, 6, 1),
            oot_start=date(2022, 7, 1),
            oot_end=date(2022, 12, 1),
            as_of=date(2023, 6, 1),
        )


def test_window_rejects_overlap():
    with pytest.raises(ValidationError):
        TrainingWindow(
            train_start=date(2019, 1, 1),
            train_end=date(2021, 7, 1),
            oot_start=date(2021, 7, 1),
            oot_end=date(2021, 12, 1),
            as_of=date(2023, 1, 1),
        )


def test_window_rejects_day_not_first():
    with pytest.raises(ValidationError):
        TrainingWindow(
            train_start=date(2019, 1, 15),
            train_end=date(2021, 6, 1),
            oot_start=date(2021, 7, 1),
            oot_end=date(2021, 12, 1),
            as_of=date(2023, 1, 1),
        )


def test_window_rejects_validation_longer_than_train():
    with pytest.raises(ValidationError):
        TrainingWindow(
            train_start=date(2021, 1, 1),
            train_end=date(2021, 3, 1),
            oot_start=date(2021, 4, 1),
            oot_end=date(2021, 6, 1),
            as_of=date(2023, 1, 1),
        )


def test_rolled_window_at_2024_12():
    w = TrainingWindow.rolled(date(2024, 12, 1))
    assert (w.oot_start, w.oot_end) == (date(2023, 10, 1), date(2023, 12, 1))
    assert (w.train_start, w.train_end) == (date(2022, 10, 1), date(2023, 9, 1))
    assert w.validation_start == date(2023, 7, 1) and w.half_life_months == 3.0


def test_default_window_is_valid():
    assert DEFAULT_WINDOW.validation_start == date(2021, 1, 1)


def test_as_params_format():
    p = TrainingWindow.rolled(date(2024, 12, 1)).as_params()
    assert p["train_start"] == "2022-10"
    assert p["as_of"] == "2024-12"
    assert p["validation_start"] == "2023-07"
    assert p["validation_months"] == "3"
    assert p["half_life_months"] == "3.0"
    assert all(isinstance(v, str) for v in p.values())
