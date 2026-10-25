"""Batch inference API. Never trains or downloads a model during a request."""
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, Response
from src.contracts import ClassifyRequest, ClassifyResponse
from src.models.artifacts import load_bundle

logger = logging.getLogger(__name__)


def create_app(model_path=None, allow_synthetic=None):
    path = model_path or os.environ.get("MODEL_PATH", "/models/candidate/model.joblib")
    demo = (os.environ.get("ALLOW_SYNTHETIC_MODEL", "false").lower() == "true"
            if allow_synthetic is None else allow_synthetic)

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.bundle = load_bundle(path, allow_synthetic=demo)
        except Exception as exc:
            raise RuntimeError(f"Cannot load model at {path}: {exc}") from exc
        logger.info("Loaded model run_id=%s data_kind=%s",
                    app.state.bundle["metadata"]["run_id"],
                    app.state.bundle["metadata"]["data_kind"])
        yield

    app = FastAPI(title="LAIA Flight Delay Classification", version="0.1.0", lifespan=lifespan)

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        metadata = app.state.bundle["metadata"]
        return {"status": "ready", "run_id": metadata["run_id"],
                "model_version": metadata.get("model_version"),
                "registered_model": metadata.get("registered_model"),
                "data_kind": metadata["data_kind"],
                "git_commit": metadata.get("git_commit", "unknown")}

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(payload: ClassifyRequest, response: Response):
        bundle = app.state.bundle
        records = [record.model_dump(mode="json") for record in payload.instances]
        predictions = bundle["pipeline"].predict(records)
        response.headers["X-Model-Run-ID"] = bundle["metadata"]["run_id"]
        response.headers["X-Model-Data-Kind"] = bundle["metadata"]["data_kind"]
        return ClassifyResponse(classifications=predictions.tolist())

    return app


app = create_app()
