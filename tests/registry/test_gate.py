import mlflow
import pytest
from mlflow import MlflowClient

from credit_mlops.registry.gate import decide, promote, rollback
from credit_mlops.registry.logging import log_and_register
from credit_mlops.settings import Settings
from credit_mlops.training.evaluate import EvalMetrics, SegmentMetrics
from credit_mlops.training.train import train_model


def alias_of(client, name, alias):
    return str(client.get_model_version_by_alias(name, alias).version)


def metrics(auc=0.75, cal=1.0, regions=None) -> EvalMetrics:
    return EvalMetrics(
        n=1000,
        default_rate=0.1,
        roc_auc=auc,
        pr_auc=0.3,
        brier=0.08,
        calibration_ratio=cal,
        by_segment={
            name: SegmentMetrics(n=500, default_rate=0.1, roc_auc=a, calibration_ratio=1.0)
            for name, a in (regions or {}).items()
        },
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def registry_settings(small_settings, tmp_path) -> Settings:
    return small_settings.model_copy(
        update={
            "mlflow_tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "mlflow_artifact_root": (tmp_path / "mlartifacts").as_uri(),
        }
    )


def test_first_model_promotes_without_champion(settings):
    d = decide(metrics(auc=0.75, cal=1.0), None, settings)
    assert d.promote and d.reasons == ["no champion yet"]


def test_promotes_better_calibrated_challenger(settings):
    d = decide(metrics(auc=0.80), metrics(auc=0.78), settings)
    assert d.promote and d.reasons == []


def test_rejects_auc_loss(settings):
    d = decide(metrics(auc=0.77), metrics(auc=0.78), settings)
    assert not d.promote and any("AUC" in r for r in d.reasons)


def test_rejects_miscalibrated_challenger(settings):
    d = decide(metrics(auc=0.80, cal=1.30), metrics(auc=0.78, cal=1.0), settings)
    assert not d.promote and any("calibration" in r for r in d.reasons)


def test_missing_calibration_or_auc_is_a_reason_not_a_crash(settings):
    d = decide(metrics(auc=None, cal=None), metrics(auc=0.78), settings)
    assert not d.promote
    assert any("AUC unavailable" in r for r in d.reasons)
    assert any("calibration ratio unavailable" in r for r in d.reasons)


def test_rejects_region_auc_drop(settings):
    champion = metrics(auc=0.78, regions={"norte": 0.78, "sur": 0.78})
    challenger = metrics(auc=0.80, regions={"norte": 0.83, "sur": 0.70})
    d = decide(challenger, champion, settings)
    assert not d.promote
    assert any("region sur" in r for r in d.reasons)
    assert not any("region norte" in r for r in d.reasons)


def test_promote_and_rollback_move_aliases(registry_settings, small_window):
    settings = registry_settings
    name = settings.registered_model_name
    result = train_model(settings, small_window)
    first = log_and_register(settings, result, repo_root=settings.data_dir)
    assert promote(settings).reasons == ["no champion yet"]
    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    assert alias_of(client, name, "champion") == first.version

    second = log_and_register(settings, result, repo_root=settings.data_dir)
    # An identical model gains exactly 0 AUC; with a positive required gain it is rejected
    # (the default gain of 0.0 accepts a tie, since the rule is `gain >= min`).
    strict = settings.model_copy(update={"gate_min_auc_gain": 0.01})
    rejected = promote(strict)
    assert not rejected.promote and any("AUC" in r for r in rejected.reasons)
    assert alias_of(client, name, "champion") == first.version
    assert client.get_run(second.run_id).data.tags["gate"] == "failed"

    lenient = settings.model_copy(update={"gate_min_auc_gain": -1.0})
    assert promote(lenient).promote
    assert alias_of(client, name, "champion") == second.version
    assert alias_of(client, name, "previous_champion") == first.version
    assert client.get_run(second.run_id).data.tags["gate"] == "passed"
    assert any(a.path == "gate_decision.json" for a in client.list_artifacts(second.run_id))

    assert rollback(settings).version == first.version
    assert alias_of(client, name, "champion") == first.version


def test_promote_without_challenger_raises(registry_settings):
    with pytest.raises(ValueError, match="no @challenger to promote"):
        promote(registry_settings)


def test_promote_same_version_raises(registry_settings, small_window):
    settings = registry_settings
    log_and_register(settings, train_model(settings, small_window), repo_root=settings.data_dir)
    promote(settings)
    with pytest.raises(ValueError, match="already champion"):
        promote(settings)


def test_rollback_without_previous_raises(registry_settings):
    with pytest.raises(ValueError, match="no previous_champion to roll back to"):
        rollback(registry_settings)


def test_model_card_logged(registry_settings, small_window):
    settings = registry_settings
    ref = log_and_register(
        settings, train_model(settings, small_window), repo_root=settings.data_dir
    )
    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    assert any(a.path == "model_card.md" for a in client.list_artifacts(ref.run_id))
    card = mlflow.artifacts.load_text(f"runs:/{ref.run_id}/model_card.md")
    for heading in ("Intended use", "Training window", "Data version", "Known limits"):
        assert heading in card
    assert "simulated" in card and "2019-01 to 2020-06" in card
