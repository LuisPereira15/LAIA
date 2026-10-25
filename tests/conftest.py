import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.models.artifacts import save_bundle
from src.models.demo import synthetic_dataset
from src.models.pipeline import make_pipeline


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory):
    frame, labels = synthetic_dataset(size=90)
    model = make_pipeline().fit(frame, labels)
    metadata = {"run_id": "test-run", "data_kind": "synthetic", "model_version": "1",
                "registered_model": "test-model", "git_commit": "test"}
    path = save_bundle(model, metadata, tmp_path_factory.mktemp("model"))
    return path, model, frame


@pytest.fixture
def client(trained_bundle):
    with TestClient(create_app(str(trained_bundle[0]), allow_synthetic=True)) as app:
        yield app


@pytest.fixture
def record(trained_bundle):
    return trained_bundle[2].iloc[0].to_dict()
