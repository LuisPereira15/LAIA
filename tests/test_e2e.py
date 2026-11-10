import os

import httpx


API_URL = os.getenv("API_URL", "http://localhost:8000")


def test_health():
    response = httpx.get(
        f"{API_URL}/health",
        timeout=30.0,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model"]
    assert body["alias"]
    assert body["version"]


def test_classify():
    payload = {
        "instances": [
            {
                "FL_DATE": "2025-05-15",
                "OP_CARRIER_AIRLINE_ID": 19805,
                "ORIGIN": "ATL",
                "ORIGIN_STATE_ABR": "GA",
                "DEST": "LAX",
                "DEST_STATE_ABR": "CA",
                "CRS_DEP_TIME": 1430,
                "CRS_ARR_TIME": 1615,
                "DISTANCE": 1946.0,
            },
            {
                "FL_DATE": "2025-08-20",
                "OP_CARRIER_AIRLINE_ID": 19977,
                "ORIGIN": "JFK",
                "ORIGIN_STATE_ABR": "NY",
                "DEST": "ORD",
                "DEST_STATE_ABR": "IL",
                "CRS_DEP_TIME": 900,
                "CRS_ARR_TIME": 1045,
                "DISTANCE": 740.0,
            },
        ]
    }

    response = httpx.post(
        f"{API_URL}/classify",
        json=payload,
        timeout=60.0,
    )

    assert response.status_code == 200

    body = response.json()

    assert "classifications" in body
    assert len(body["classifications"]) == len(payload["instances"])

    valid_classes = {
        "on_time",
        "moderate_delay",
        "severe_delay",
    }

    assert all(
        prediction in valid_classes
        for prediction in body["classifications"]
    )