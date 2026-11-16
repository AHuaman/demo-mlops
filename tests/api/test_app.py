import json
import subprocess
import sys

import joblib
import mlflow.pyfunc
import pytest
from fastapi.testclient import TestClient

from credit_mlops.api.app import create_app
from credit_mlops.registry.loading import LoadedModel, ModelUnavailableError
from credit_mlops.registry.logging import ModelRef
from credit_mlops.registry.pyfunc import CreditRiskModel
from credit_mlops.training.train import train_model

WINDOW = {"train_start": "2019-01", "oot_end": "2020-12"}


def valid_application() -> dict:
    return {
        "bureau_score": 720,
        "monthly_income_pen": 4200.0,
        "debt_to_income": 0.28,
        "loan_amount_pen": 15000.0,
        "term_months": 24,
        "age": 38,
        "months_at_job": 60,
        "num_active_lenders": 2,
        "max_days_past_due_12m": 0,
        "product": "consumer",
        "region": "lima",
        "channel": "digital",
    }


@pytest.fixture(scope="module")
def loaded_model(small_settings, small_window, tmp_path_factory) -> LoadedModel:
    tmp = tmp_path_factory.mktemp("api-model")
    pipeline = train_model(small_settings, small_window).pipeline
    joblib.dump(pipeline, tmp / "pipeline.joblib")
    mlflow.pyfunc.save_model(
        path=tmp / "model",
        python_model=CreditRiskModel(),
        artifacts={"pipeline": str(tmp / "pipeline.joblib")},
        model_config={"threshold": 0.15, "top_k": 3},
    )
    return LoadedModel(
        ref=ModelRef(name="credit_default_pd", version="1", run_id="run-1"),
        model=mlflow.pyfunc.load_model(tmp / "model"),
        threshold=0.15,
        window=WINDOW,
    )


@pytest.fixture
def settings(small_settings, tmp_path):
    return small_settings.model_copy(update={"data_dir": tmp_path / "data"})


@pytest.fixture
def client(settings, loaded_model):
    with TestClient(create_app(settings, loader=lambda s: loaded_model)) as c:
        yield c


@pytest.fixture
def client_without_model(settings):
    def unavailable(s):
        raise ModelUnavailableError("MLflow is down; start it with: credit-mlops mlflow-server")

    with TestClient(create_app(settings, loader=unavailable)) as c:
        yield c


def test_health_always_ok(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_ready_503_names_the_fix(client_without_model):
    response = client_without_model.get("/ready")
    assert response.status_code == 503
    assert "credit-mlops mlflow-server" in response.json()["detail"]


def test_ready_and_model_report_the_champion(client):
    assert client.get("/ready").json() == {"status": "ready", "model_version": "1"}
    assert client.get("/model").json() == {
        "model_name": "credit_default_pd",
        "alias": "champion",
        "model_version": "1",
        "training_window": WINDOW,
    }


def test_model_and_predict_503_without_model(client_without_model):
    assert client_without_model.get("/model").status_code == 503
    response = client_without_model.post("/predict", json={"applications": [valid_application()]})
    assert response.status_code == 503
    assert "credit-mlops mlflow-server" in response.json()["detail"]


def test_loader_runs_once_in_lifespan_not_at_creation(settings, loaded_model):
    calls = []

    def loader(s):
        calls.append(s)
        return loaded_model

    app = create_app(settings, loader=loader)
    assert calls == []
    with TestClient(app) as c:
        c.get("/ready")
        c.get("/model")
    assert len(calls) == 1


def test_predict_returns_explanations(client):
    response = client.post("/predict", json={"applications": [valid_application()] * 2})
    body = response.json()
    assert response.status_code == 200 and body["model_version"] == "1"
    assert body["model_name"] == "credit_default_pd" and body["threshold"] == 0.15
    assert len(body["predictions"]) == 2
    prediction = body["predictions"][0]
    assert 0 <= prediction["probability_default"] <= 1
    assert prediction["decision"] in {"approve", "review"}
    assert len(prediction["explanation"]["top_factors"]) == 3


@pytest.mark.parametrize(
    "patch", [{"region": "lima metropolitana"}, {"bureau_score": 1200}, {"channel": None}]
)
def test_invalid_input_is_422_naming_field(client, patch):
    response = client.post("/predict", json={"applications": [valid_application() | patch]})
    assert response.status_code == 422 and next(iter(patch)) in response.text


def test_missing_field_is_422(client):
    application = valid_application()
    del application["age"]
    response = client.post("/predict", json={"applications": [application]})
    assert response.status_code == 422 and "age" in response.text


def test_extra_field_and_empty_batch_are_422(client):
    extra = valid_application() | {"loan_id": "x"}
    assert client.post("/predict", json={"applications": [extra]}).status_code == 422
    assert client.post("/predict", json={"applications": []}).status_code == 422


def test_predictions_are_logged(client, settings):
    applications = [valid_application(), valid_application() | {"bureau_score": 450}]
    body = client.post("/predict", json={"applications": applications}).json()
    path = settings.predictions_dir / "api" / "predictions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert len({r["request_id"] for r in records}) == 1
    assert [r["inputs"] for r in records] == applications
    for record, prediction in zip(records, body["predictions"], strict=True):
        assert record["model_version"] == "1"
        assert record["probability_default"] == prediction["probability_default"]
        assert record["decision"] == prediction["decision"]
        assert record["timestamp"].endswith("+00:00")


def test_api_does_not_import_heavy_modules():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, credit_mlops.api.app; "
            "print(any(m in sys.modules for m in ('great_expectations','prefect','matplotlib')))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"
