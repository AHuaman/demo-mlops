import mlflow
from mlflow import MlflowClient

from credit_mlops.registry.logging import log_and_register
from credit_mlops.registry.pyfunc import PREDICTION_COLUMNS
from credit_mlops.schema import FEATURES
from credit_mlops.training.train import train_model


def test_log_and_register_creates_challenger(small_settings, small_window, tmp_path):
    settings = small_settings.model_copy(
        update={
            "mlflow_tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "mlflow_artifact_root": (tmp_path / "mlartifacts").as_uri(),
        }
    )
    result = train_model(settings, small_window)
    ref = log_and_register(settings, result, repo_root=tmp_path)

    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    mv = client.get_model_version_by_alias(settings.registered_model_name, "challenger")
    assert str(mv.version) == ref.version
    run = client.get_run(ref.run_id)
    assert {"git_sha", "data_version", "settings_hash"} <= set(run.data.tags)
    assert run.data.params["oot_start"] == "2020-07"
    assert run.data.params["oot_end"] == "2020-12"
    assert run.data.params["as_of"] == "2021-12"
    assert "oot_roc_auc" in run.data.metrics
    assert "oot_calibration_ratio__norte" in run.data.metrics
    children = client.search_runs(
        [run.info.experiment_id], f"tags.mlflow.parentRunId = '{ref.run_id}'"
    )
    assert len(children) == 2
    assert all("validation_auc" in c.data.metrics for c in children)
    loaded = mlflow.pyfunc.load_model(f"models:/{settings.registered_model_name}@challenger")
    assert loaded.metadata.signature is not None
    reference_files = client.list_artifacts(ref.run_id, "reference")
    assert any(a.path == "reference/reference.parquet" for a in reference_files)
    example = result.reference[FEATURES].head(5)
    prediction = loaded.predict(example)
    assert list(prediction.columns) == PREDICTION_COLUMNS
    assert len(prediction) == len(example)
