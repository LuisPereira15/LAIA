import os

# The API settings are created when src.api.main is imported.
# Unit tests provide local values so they do not depend on a real deployment.
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://mlflow.test:5000")
os.environ.setdefault("MLFLOW_MODEL_NAME", "flight-delay-classifier")
os.environ.setdefault("MODEL_ALIAS", "staging")
os.environ.setdefault("API_MAX_INSTANCES", "100")
os.environ.setdefault("API_BATCH_SIZE", "2")

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.settings import settings


class FakeModel:
    """Small deterministic model used only by the API unit tests."""

    def predict(self, frame):
        predictions_by_origin = {
            "ATL": "on_time",
            "JFK": "moderate_delay",
            "ORD": "severe_delay",
        }

        return np.array(
            [
                predictions_by_origin.get(origin, "on_time")
                for origin in frame["ORIGIN"]
            ]
        )


@pytest.fixture(autouse=True)
def configure_app_state():
    """Provide a fake loaded model without contacting MLflow."""
    app.state.model = FakeModel()
    app.state.model_version = "7"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_instance():
    return {
        "FL_DATE": "2025-05-15",
        "OP_CARRIER_AIRLINE_ID": 19805,
        "ORIGIN": "ATL",
        "ORIGIN_STATE_ABR": "GA",
        "DEST": "LAX",
        "DEST_STATE_ABR": "CA",
        "CRS_DEP_TIME": 1430,
        "CRS_ARR_TIME": 1615,
        "DISTANCE": 1946.0,
    }


def test_health_reports_served_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": settings.model_name,
        "alias": settings.model_alias,
        "version": "7",
    }


def test_classify_returns_one_prediction_per_instance_in_order(
    client,
    valid_instance,
):
    second = valid_instance.copy()
    second["ORIGIN"] = "JFK"

    third = valid_instance.copy()
    third["ORIGIN"] = "ORD"

    response = client.post(
        "/classify",
        json={"instances": [valid_instance, second, third]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "classifications": [
            "on_time",
            "moderate_delay",
            "severe_delay",
        ]
    }


def test_classify_reports_ignored_extra_fields(client, valid_instance):
    instance = valid_instance | {
        "ARR_DELAY": 42.0,
        "CANCELLED": 0,
    }

    response = client.post(
        "/classify",
        json={"instances": [instance]},
    )

    assert response.status_code == 200
    assert response.headers["X-Ignored-Fields"] == "ARR_DELAY, CANCELLED"
    assert response.json() == {
        "classifications": ["on_time"]
    }


def test_classify_rejects_empty_instances(client):
    response = client.post(
        "/classify",
        json={"instances": []},
    )

    assert response.status_code == 422


def test_classify_rejects_non_positive_distance(
    client,
    valid_instance,
):
    invalid = valid_instance.copy()
    invalid["DISTANCE"] = 0

    response = client.post(
        "/classify",
        json={"instances": [invalid]},
    )

    assert response.status_code == 422


def test_classify_converts_pipeline_value_error_to_422(
    client,
    valid_instance,
):
    class RejectingModel:
        def predict(self, frame):
            raise ValueError("invalid HHMM value")

    app.state.model = RejectingModel()

    response = client.post(
        "/classify",
        json={"instances": [valid_instance]},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "invalid HHMM value"
    }