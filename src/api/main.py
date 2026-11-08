"""FastAPI inference service: loads the production model from MLflow and serves /classify."""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from src.api.schemas import ClassifyRequest, ClassifyResponse, FlightInstance
from src.api.settings import settings

# Logging configuration. Timestamps, levels and logger names are included in every line.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('flight_api')

# The 9 BTS fields declared in the schema. Only these are sent to the model.
INPUT_FIELDS = set(FlightInstance.model_fields)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; the API does not start without it."""

    # Set the MLflow tracking URI, the model is loaded from there.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Resolve the alias to a fixed version first, so the loaded model and /health always match.
    version = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias).version
    model_uri = f'models:/{settings.model_name}/{version}'

    logger.info('Alias @%s points to version %s; loading %s from %s',
                settings.model_alias, version, model_uri, settings.mlflow_tracking_uri)

    # Full pipeline. Input adapter + preprocessor + classifier.
    app.state.model = mlflow.sklearn.load_model(model_uri)
    app.state.model_version = version

    logger.info('Model loaded: %s version %s', settings.model_name, version)
    yield


app = FastAPI(title='Flight Delay Severity API', version='1.0.0', lifespan=lifespan)


def collect_ignored_fields(instances: list[FlightInstance]) -> list[str]:
    """Return the names of extra columns sent in any instance, sorted alphabetically."""
    ignored_fields: set[str] = set()

    for instance in instances:
        # model_extra holds the columns that are not part of the schema.
        extra_columns = instance.model_extra or {}
        ignored_fields.update(extra_columns.keys())

    return sorted(ignored_fields)


def build_input_frame(instances: list[FlightInstance]) -> pd.DataFrame:
    """Build a DataFrame with only the model input fields, one row per flight, in request order."""
    rows: list[dict] = []

    for instance in instances:
        # mode='json' turns FL_DATE into the 'YYYY-MM-DD' text the pipeline expects.
        row = instance.model_dump(mode='json', include=INPUT_FIELDS)
        rows.append(row)

    return pd.DataFrame(rows)


def predict_in_batches(model: Pipeline, frame: pd.DataFrame, batch_size: int) -> list[str]:
    """Predict in chunks of batch_size rows to keep memory stable; results keep the input order."""
    predictions: list[str] = []

    for batch_start in range(0, len(frame), batch_size):
        batch_end = batch_start + batch_size
        batch = frame.iloc[batch_start:batch_end]

        # The pipeline maps BTS names to model names, validates values and predicts.
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions.tolist())

    return predictions


### ---------------- API Endpoints ---------------- ###

@app.get('/health')
def health() -> dict:
    """Report that the API is up and which model version it serves."""
    return {
        'status': 'ok',
        'model': settings.model_name,
        'alias': settings.model_alias,
        'version': str(app.state.model_version),
    }


# Sync endpoint on purpose: prediction is CPU-bound, so FastAPI runs it in a worker thread.
@app.post('/classify', response_model=ClassifyResponse)
def classify(request: ClassifyRequest, response: Response) -> ClassifyResponse:
    """Classify a batch of flights into on_time, moderate_delay or severe_delay."""

    # Extra columns are accepted but not used and they are reported wthout changing the body.
    ignored = collect_ignored_fields(request.instances)
    if ignored:
        response.headers['X-Ignored-Fields'] = ', '.join(ignored)
        logger.info('Ignored fields: %s', ignored)

    frame = build_input_frame(request.instances)
    try:
        predictions = predict_in_batches(app.state.model, frame, settings.batch_size)
    except ValueError as exc:
        # Value errors come from the pipeline's input checks (e.g. HHMM minutes > 59).
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info('Classified %d flights', len(predictions))
    return ClassifyResponse(classifications=predictions)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected errors in full but return a generic message to the client."""
    logger.exception('Unexpected error on %s %s', request.method, request.url.path)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error'})