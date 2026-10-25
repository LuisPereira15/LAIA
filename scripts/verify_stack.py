"""Verify the live API, MLflow run, registry and downloadable serving artifact."""
import json
import os
from pathlib import Path
import tempfile

import httpx
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.contracts import CLASSES
from src.models.artifacts import load_bundle
from src.models.demo import synthetic_dataset


def main():
    base = os.getenv("API_URL", "http://api:8000")
    tracking = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    expected_kind = os.getenv("EXPECTED_DATA_KIND", "synthetic")
    with httpx.Client(base_url=base, timeout=30) as http:
        response = http.get("/health/ready")
        response.raise_for_status()
        ready = response.json()
        assert ready["status"] == "ready" and ready["data_kind"] == expected_kind
        frame, _ = synthetic_dataset(size=6, start="2025-05-01")
        records = frame.iloc[[2, 0, 1, 2]].to_dict("records")
        response = http.post("/classify", json={"instances": records})
        response.raise_for_status()
        batch = response.json()["classifications"]
        assert len(batch) == len(records) and set(batch).issubset(CLASSES)
        for i, record in enumerate(records):
            singleton = http.post("/classify", json={"instances": [record]})
            singleton.raise_for_status()
            assert singleton.json()["classifications"] == [batch[i]]
        unseen = {**records[0], "ORIGIN": "ZZZ", "OP_CARRIER_AIRLINE_ID": "99999"}
        assert http.post("/classify", json={"instances": [unseen]}).status_code == 200
        assert http.post("/classify", json={"instances": []}).status_code == 422
        assert http.post("/classify", content="{", headers={"Content-Type": "application/json"}).status_code == 422
        assert response.headers["X-Model-Run-ID"] == ready["run_id"]

    mlflow.set_tracking_uri(tracking)
    client = MlflowClient()
    run = client.get_run(ready["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["data_kind"] == expected_kind
    assert {"validation_accuracy", "validation_macro_f1"}.issubset(run.data.metrics)
    model = client.get_model_version(ready["registered_model"], ready["model_version"])
    assert model.run_id == ready["run_id"]
    # Restore the registered MLflow model as well as the exact API bundle.
    restored = mlflow.sklearn.load_model(
        f"models:/{ready['registered_model']}/{ready['model_version']}")
    assert restored.predict(records).tolist() == batch
    with tempfile.TemporaryDirectory() as directory:
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{ready['run_id']}/deployment/model.joblib", dst_path=directory)
        bundle = load_bundle(downloaded, allow_synthetic=expected_kind == "synthetic")
        assert bundle["metadata"]["run_id"] == ready["run_id"]
        assert bundle["pipeline"].predict(records).tolist() == batch
    print(json.dumps({"result": "PASS", "data_kind": expected_kind,
                      "run_id": ready["run_id"],
                      "checks": ["batch_order", "offline_parity", "unseen_categories",
                                 "invalid_requests", "tracking", "registry", "registry_model_parity", "artifact_download"]}))


if __name__ == "__main__":
    main()
