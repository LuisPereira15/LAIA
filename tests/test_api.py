"""Unit tests for the inference API's robust /classify and /labels contract.

The API is built to never reject a batch: missing, null or invalid fields are
imputed, and any instance the model cannot score falls back to a default label.
These tests pin that behaviour down.
"""

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

from src.api.main import FALLBACK_LABEL, app
from src.api.settings import settings

VALID_LABELS = {"on_time", "moderate_delay", "severe_delay"}


class FakeModel:
    """Small deterministic model used only by the API unit tests."""

    def predict(self, frame):
        predictions_by_origin = {
            "ATL": "on_time",
            "JFK": "moderate_delay",
            "ORD": "severe_delay",
        }
        return np.array(
            [predictions_by_origin.get(origin, "on_time") for origin in frame["ORIGIN"]]
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


def test_classify_returns_one_prediction_per_instance_in_order(client, valid_instance):
    second = valid_instance | {"ORIGIN": "JFK"}
    third = valid_instance | {"ORIGIN": "ORD"}

    response = client.post(
        "/classify",
        json={"instances": [valid_instance, second, third]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "classifications": ["on_time", "moderate_delay", "severe_delay"]
    }


def test_classify_accepts_the_leaderboard_wrapper(client, valid_instance):
    # request_id and mode are read, not rejected.
    response = client.post(
        "/classify",
        json={"request_id": "r-1", "mode": "live", "instances": [valid_instance]},
    )

    assert response.status_code == 200
    assert response.json() == {"classifications": ["on_time"]}


def test_classify_accepts_and_ignores_extra_fields(client, valid_instance):
    instance = valid_instance | {"ARR_DELAY": 42.0, "CANCELLED": 0, "TAIL_NUM": "N1"}

    response = client.post("/classify", json={"instances": [instance]})

    assert response.status_code == 200
    assert response.json() == {"classifications": ["on_time"]}
    assert "X-Ignored-Fields" not in response.headers


def test_classify_empty_instances_returns_empty_list(client):
    response = client.post("/classify", json={"instances": []})

    assert response.status_code == 200
    assert response.json() == {"classifications": []}


def test_classify_missing_instances_key_returns_empty_list(client):
    response = client.post("/classify", json={"request_id": "r", "mode": "live"})

    assert response.status_code == 200
    assert response.json() == {"classifications": []}


def test_classify_imputes_invalid_distance(client, valid_instance):
    invalid = valid_instance | {"DISTANCE": 0}

    response = client.post("/classify", json={"instances": [invalid]})

    # Imputed, not rejected: ORIGIN is still ATL, so the label is deterministic.
    assert response.status_code == 200
    assert response.json() == {"classifications": ["on_time"]}


def test_classify_handles_nulls_and_missing_carrier(client):
    # No OP_CARRIER_AIRLINE_ID, several nulls; only ORIGIN carries signal here.
    instance = {
        "FL_DATE": None,
        "OP_UNIQUE_CARRIER": "AA",
        "ORIGIN": "JFK",
        "ORIGIN_STATE_ABR": None,
        "DEST": "MIA",
        "CRS_DEP_TIME": None,
        "DISTANCE": None,
    }

    response = client.post("/classify", json={"instances": [instance]})

    assert response.status_code == 200
    assert response.json() == {"classifications": ["moderate_delay"]}


def test_classify_never_rejects_a_mixed_batch(client, valid_instance):
    clean = valid_instance  # ORIGIN ATL -> on_time
    all_null = {"FL_DATE": None, "ORIGIN": None, "DEST": None}  # ORIGIN UNK -> on_time
    broken = {
        "FL_DATE": "bad-date",
        "OP_CARRIER_AIRLINE_ID": "AA",
        "ORIGIN": "ORD",  # -> severe_delay
        "CRS_DEP_TIME": 1470,
        "DISTANCE": -5,
    }

    response = client.post(
        "/classify",
        json={"mode": "live", "instances": [clean, all_null, broken]},
    )

    assert response.status_code == 200
    labels = response.json()["classifications"]
    assert len(labels) == 3
    assert all(label in VALID_LABELS for label in labels)
    assert labels == ["on_time", "on_time", "severe_delay"]


def test_classify_falls_back_when_model_fails(client, valid_instance):
    class RejectingModel:
        def predict(self, frame):
            raise ValueError("boom")

    app.state.model = RejectingModel()

    response = client.post("/classify", json={"instances": [valid_instance]})

    # Never 422: an unscorable instance gets the fallback label.
    assert response.status_code == 200
    assert response.json() == {"classifications": [FALLBACK_LABEL]}


def test_labels_accepts_outcomes_and_returns_200(client):
    payload = {
        "request_id": "r-1",
        "mode": "live",
        "labels": ["on_time", "severe_delay"],
        "outcomes": {"ARR_DELAY": [3, 120], "TAXI_OUT": [12, 20]},
    }

    response = client.post("/labels", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "received": 2}


def test_labels_accepts_minimal_payload(client):
    response = client.post("/labels", json={})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"