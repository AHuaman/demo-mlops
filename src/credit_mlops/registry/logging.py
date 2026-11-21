"""Log a trained model with full traceability and register it as the challenger."""

import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pydantic import BaseModel

import credit_mlops
from credit_mlops.data.versioning import data_version, git_info
from credit_mlops.months import month_label
from credit_mlops.registry.mlflow_setup import configure_mlflow
from credit_mlops.registry.model_card import render_model_card
from credit_mlops.registry.pyfunc import CreditRiskModel
from credit_mlops.schema import FEATURES
from credit_mlops.settings import Settings
from credit_mlops.training.train import TrainResult


class ModelRef(BaseModel):
    name: str
    version: str
    run_id: str


def _example_prediction(settings: Settings, result: TrainResult, example):
    """Run the pyfunc wrapper in-process to infer the output signature."""
    model = CreditRiskModel()
    model.pipeline = result.pipeline
    model.threshold = settings.decision_threshold
    model.top_k = settings.explanation_top_k
    return model.predict(None, example)


def log_and_register(settings: Settings, result: TrainResult, repo_root: Path) -> ModelRef:
    configure_mlflow(settings, settings.training_experiment)
    window = result.window
    example = result.reference[FEATURES].head(5)
    signature = infer_signature(example, _example_prediction(settings, result, example))

    with mlflow.start_run(run_name=f"train-{month_label(window.as_of)}") as parent:
        mlflow.log_params(window.as_params())
        mlflow.log_params(result.best_params)
        mlflow.log_param("learning_rate", settings.learning_rate)
        mlflow.log_metrics(result.oot_metrics.flat("oot"))
        mlflow.set_tags(
            {
                **git_info(repo_root),
                "data_version": data_version(repo_root),
                "settings_hash": settings.settings_hash(),
            }
        )

        for hpo in result.hpo_results:
            name = f"hpo-n{hpo['n_estimators']}-d{hpo['max_depth']}"
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(
                    {"n_estimators": hpo["n_estimators"], "max_depth": hpo["max_depth"]}
                )
                mlflow.log_metric("validation_auc", hpo["validation_auc"])

        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference.parquet"
            result.reference.to_parquet(reference_path)
            mlflow.log_artifact(str(reference_path), artifact_path="reference")

            pipeline_path = Path(tmp) / "pipeline.joblib"
            joblib.dump(result.pipeline, pipeline_path)
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=CreditRiskModel(),
                artifacts={"pipeline": str(pipeline_path)},
                model_config={
                    "threshold": settings.decision_threshold,
                    "top_k": settings.explanation_top_k,
                },
                signature=signature,
                input_example=example,
                code_paths=[str(Path(credit_mlops.__file__).parent)],
                registered_model_name=settings.registered_model_name,
            )

    version = str(info.registered_model_version)
    client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
    client.set_registered_model_alias(settings.registered_model_name, "challenger", version)
    ref = ModelRef(name=settings.registered_model_name, version=version, run_id=parent.info.run_id)
    client.log_text(ref.run_id, render_model_card(settings, ref, result), "model_card.md")
    return ref
