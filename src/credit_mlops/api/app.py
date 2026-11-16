"""FastAPI service: serves `@champion` with per-prediction SHAP explanations."""

import json
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException

from credit_mlops.api.schemas import (
    ModelInfo,
    Prediction,
    PredictRequest,
    PredictResponse,
)
from credit_mlops.prediction_log import PredictionLogger
from credit_mlops.registry.loading import LoadedModel, ModelUnavailableError, load_champion
from credit_mlops.registry.pyfunc import Explanation, Factor
from credit_mlops.schema import FEATURES, NUMERIC_FEATURES
from credit_mlops.settings import Settings

ALIAS = "champion"


def _to_frame(applications: list[dict]) -> pd.DataFrame:
    """Build the model input, keeping integer features int64 to match the MLflow signature."""
    frame = pd.DataFrame(applications, columns=FEATURES)
    integers = {name: "int64" for name, spec in NUMERIC_FEATURES.items() if spec.integer}
    return frame.astype(integers)


def create_app(
    settings: Settings | None = None,
    loader: Callable[[Settings], LoadedModel] = load_champion,
) -> FastAPI:
    settings = settings or Settings()
    logger = PredictionLogger(settings.predictions_dir / "api" / "predictions.jsonl")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.champion = None
        app.state.error = None
        try:
            app.state.champion = loader(settings)
        except ModelUnavailableError as error:
            app.state.error = str(error)
        yield

    app = FastAPI(title="credit-mlops scoring API", lifespan=lifespan)

    def champion(app_state) -> LoadedModel:
        if app_state.champion is None:
            raise HTTPException(status_code=503, detail=app_state.error)
        return app_state.champion

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        loaded = champion(app.state)
        return {"status": "ready", "model_version": loaded.ref.version}

    @app.get("/model")
    def model_info() -> ModelInfo:
        loaded = champion(app.state)
        return ModelInfo(
            model_name=loaded.ref.name,
            alias=ALIAS,
            model_version=loaded.ref.version,
            training_window=loaded.window,
        )

    @app.post("/predict")
    def predict(request: PredictRequest) -> PredictResponse:
        loaded = champion(app.state)
        records = [a.model_dump() for a in request.applications]
        output = loaded.model.predict(_to_frame(records))
        predictions = [
            Prediction(
                probability_default=float(row.probability_default),
                decision=row.decision,
                explanation=Explanation(
                    base_log_odds=float(row.base_log_odds),
                    top_factors=[Factor(**f) for f in json.loads(row.top_factors)],
                ),
            )
            for row in output.itertuples()
        ]
        request_id = uuid.uuid4().hex
        timestamp = datetime.now(UTC).isoformat()
        logger.append(
            [
                {
                    "request_id": request_id,
                    "timestamp": timestamp,
                    "model_version": loaded.ref.version,
                    "inputs": inputs,
                    "probability_default": p.probability_default,
                    "decision": p.decision,
                }
                for inputs, p in zip(records, predictions, strict=True)
            ]
        )
        return PredictResponse(
            model_name=loaded.ref.name,
            model_version=loaded.ref.version,
            threshold=loaded.threshold,
            predictions=predictions,
        )

    return app


app = create_app()
