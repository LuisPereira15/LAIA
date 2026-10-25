import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


def test_batch_preserves_order_and_matches_offline(client, trained_bundle):
    records = trained_bundle[2].iloc[[2, 0, 1, 2]].to_dict("records")
    response = client.post("/classify", json={"instances": records})
    assert response.status_code == 200
    expected = trained_bundle[1].predict(records).tolist()
    assert response.json() == {"classifications": expected}
    assert len(set(expected)) == 3
    assert response.headers["X-Model-Run-ID"] == "test-run"


def test_unknown_categories_do_not_fail(client, record):
    record.update(ORIGIN="ZZZ", DEST="YYY", OP_CARRIER_AIRLINE_ID="99999")
    response = client.post("/classify", json={"instances": [record]})
    assert response.status_code == 200
    assert len(response.json()["classifications"]) == 1


def test_outcomes_and_redundant_calendar_are_not_features(client, record):
    original = client.post("/classify", json={"instances": [record]}).json()
    record.update(DEP_DELAY=9999, ARR_DELAY_NEW=9999, CANCELLED=1, DIVERTED=1, YEAR=2099, MONTH=12)
    assert client.post("/classify", json={"instances": [record]}).json() == original


@pytest.mark.parametrize("field,value", [
    ("FL_DATE", "2022-02-30"), ("FL_DATE", 1640995200),
    ("CRS_DEP_TIME", "1260"), ("CRS_DEP_TIME", True),
    ("CRS_DEP_TIME", 930.5), ("CRS_ARR_TIME", -1),
    ("DISTANCE", -1), ("DISTANCE", True), ("DISTANCE", "NaN"),
    ("CRS_ELAPSED_TIME", "Infinity"), ("ORIGIN", "ABCD"),
    ("OP_CARRIER_AIRLINE_ID", True), ("OP_CARRIER_AIRLINE_ID", -1),
])
def test_invalid_field_is_rejected(client, record, field, value):
    record[field] = value
    response = client.post("/classify", json={"instances": [record]})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_empty_oversized_and_malformed_requests(client, record):
    for payload in ({"instances": []}, {"instances": [record] * 1001}, {}, [record]):
        assert client.post("/classify", json=payload).status_code == 422
    assert client.post("/classify", content="{", headers={"Content-Type": "application/json"}).status_code == 422


def test_missing_required_field(client, record):
    record.pop("FL_DATE")
    assert client.post("/classify", json={"instances": [record]}).status_code == 422


def test_optional_duration_and_midnight(client, record):
    record.pop("CRS_ELAPSED_TIME")
    record["CRS_DEP_TIME"] = 2400
    assert client.post("/classify", json={"instances": [record]}).status_code == 200


def test_health_reports_loaded_model(client):
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health/ready").json()["run_id"] == "test-run"
    assert client.get("/health/ready").json()["data_kind"] == "synthetic"


def test_normal_configuration_refuses_synthetic_model(trained_bundle):
    with pytest.raises(RuntimeError, match="Synthetic"):
        with TestClient(create_app(str(trained_bundle[0]), allow_synthetic=False)):
            pass


def test_missing_artifact_fails_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot load model"):
        with TestClient(create_app(str(tmp_path / "missing.joblib"))):
            pass
