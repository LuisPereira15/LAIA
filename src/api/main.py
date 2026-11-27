"""FastAPI inference service: loads the production model and serves /classify and /labels.

Built to survive real leaderboard traffic: it never rejects a batch for
malformed input. Missing, null or invalid fields are imputed to safe, in-range
defaults so the bundled preprocessing pipeline always accepts the row, and any
instance the model still cannot score falls back to a default label. Every
request therefore returns exactly one valid label per instance.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from src.api.schemas import ClassifyRequest, ClassifyResponse, LabelsRequest
from src.api.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('flight_api')

VALID_LABELS = {'on_time', 'moderate_delay', 'severe_delay'}

# Returned when an instance cannot be scored at all. on_time is the majority
# class, so it is the safest guess.
FALLBACK_LABEL = 'on_time'

# Safe, in-range defaults for every model input. Used to impute missing, null
# or invalid values. Unknown airports/airlines are handled by the model as
# unseen categories; a neutral date/time/distance keeps the row scorable.
DEFAULTS: dict[str, object] = {
    'FL_DATE': '2024-01-01',
    'OP_CARRIER_AIRLINE_ID': 999999,
    'ORIGIN': 'UNK',
    'ORIGIN_STATE_ABR': 'UNK',
    'DEST': 'UNK',
    'DEST_STATE_ABR': 'UNK',
    'CRS_DEP_TIME': 0,
    'CRS_ARR_TIME': 0,
    'DISTANCE': 800.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; the API does not start without it."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    version = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias).version
    model_uri = f'models:/{settings.model_name}/{version}'
    logger.info('Loading %s (alias @%s -> version %s)', model_uri, settings.model_alias, version)
    app.state.model = mlflow.sklearn.load_model(model_uri)
    app.state.model_version = version
    logger.info('Model loaded: %s version %s', settings.model_name, version)
    yield


app = FastAPI(title='Flight Delay Severity API', version='2.0.0', lifespan=lifespan)


def clean_value(field: str, raw: object) -> object:
    """Return a usable value for one model field, or the field's safe default.

    Null, missing and clearly invalid values (wrong type, empty string,
    non-positive distance, out-of-range HHMM time) fall back to the default.
    """
    default = DEFAULTS[field]
    if raw is None:
        return default
    try:
        if field in ('CRS_DEP_TIME', 'CRS_ARR_TIME'):
            value = int(float(raw))
            valid = (0 <= value <= 2359 and value % 100 < 60) or value == 2400
            return value if valid else default
        if field == 'OP_CARRIER_AIRLINE_ID':
            value = int(float(raw))
            return value if value > 0 else default
        if field == 'DISTANCE':
            value = float(raw)
            return value if value > 0 else default
        if field == 'FL_DATE':
            text = str(raw)
            date.fromisoformat(text)  # raises if not a YYYY-MM-DD date
            return text
        text = str(raw).strip()  # ORIGIN, DEST and the state codes
        return text if text else default
    except (ValueError, TypeError):
        return default


def build_input_frame(instances: list) -> pd.DataFrame:
    """Build one clean, fully-populated row per flight, in request order."""
    rows = [{field: clean_value(field, getattr(instance, field, None)) for field in DEFAULTS}
            for instance in instances]
    return pd.DataFrame(rows, columns=list(DEFAULTS))


def safe_label(label: object) -> str:
    """Coerce a prediction to one of the three valid categories."""
    text = str(label)
    return text if text in VALID_LABELS else FALLBACK_LABEL


def classify_frame(model: Pipeline, frame: pd.DataFrame) -> list[str]:
    """Predict the whole batch; if that fails, predict row by row with a fallback.

    Guarantees exactly one valid label per row, whatever the input.
    """
    try:
        return [safe_label(label) for label in model.predict(frame)]
    except Exception:
        logger.exception('Batch prediction failed; falling back to per-row prediction')
    labels: list[str] = []
    for position in range(len(frame)):
        try:
            labels.append(safe_label(model.predict(frame.iloc[[position]])[0]))
        except Exception:
            logger.exception('Row %d could not be scored; using fallback label', position)
            labels.append(FALLBACK_LABEL)
    return labels


@app.get('/health')
def health() -> dict:
    """Report that the API is up and which model version it serves."""
    return {'status': 'ok', 'model': settings.model_name,
            'alias': settings.model_alias, 'version': str(app.state.model_version)}


@app.post('/classify', response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a batch of flights. Never rejects: unscorable rows get a fallback label."""
    if not request.instances:
        return ClassifyResponse(classifications=[])
    frame = build_input_frame(request.instances)
    labels = classify_frame(app.state.model, frame)
    logger.info('Classified %d flights (mode=%s, request_id=%s)',
                len(labels), request.mode, request.request_id)
    return ClassifyResponse(classifications=labels)


@app.post('/labels')
def labels(request: LabelsRequest) -> dict:
    """Receive the true outcomes after scoring and acknowledge with HTTP 200.

    Live and sandbox traffic are told apart by `mode`, so sandbox labels do not
    pollute live monitoring. Persisting and analysing them is left to the
    monitoring layer.
    """
    count = len(request.labels) if request.labels is not None else 0
    logger.info('Received %d labels (mode=%s, request_id=%s)', count, request.mode, request.request_id)
    return {'status': 'ok', 'received': count}


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: log and return a generic 500 (classify avoids this path)."""
    logger.exception('Unexpected error on %s %s', request.method, request.url.path)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error'})