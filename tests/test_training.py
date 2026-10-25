import pandas as pd
import pytest
from pydantic import ValidationError

from src.contracts import hhmm_to_minutes
from src.models.demo import synthetic_dataset
from src.models.train import read_prepared, validate_splits


@pytest.mark.parametrize("value,minutes", [("0000", 0), ("0930", 570), (2359, 1439), (2400, 0)])
def test_scheduled_time_conversion(value, minutes):
    assert hhmm_to_minutes(value) == minutes


def test_prepared_contract_and_extra_outcomes(tmp_path):
    frame, labels = synthetic_dataset(size=12)
    frame["delay_severity"] = labels
    frame["DEP_DELAY"] = 999
    frame.loc[0, "CRS_ELAPSED_TIME"] = None
    path = tmp_path / "prepared.csv"
    frame.to_csv(path, index=False)
    loaded, target = read_prepared(path, max_rows=12)
    assert len(loaded) == 12
    assert "DEP_DELAY" not in loaded
    assert target.equals(labels)
    assert pd.isna(loaded.iloc[0]["CRS_ELAPSED_TIME"])
    with pytest.raises(ValueError, match="exceeds"):
        read_prepared(path, max_rows=11)


def test_temporal_overlap_and_missing_classes_rejected():
    train_x, train_y = synthetic_dataset(size=12)
    future_x, _ = synthetic_dataset(size=12, start="2023-01-01")
    validate_splits(train_x, train_y, future_x)
    with pytest.raises(ValueError, match="strictly before"):
        validate_splits(train_x, train_y, train_x)
    with pytest.raises(ValueError, match="all three"):
        validate_splits(train_x, train_y.replace("severe_delay", "on_time"), future_x)


def test_missing_flight_date_rejected(tmp_path):
    frame, labels = synthetic_dataset(size=6)
    frame = frame.drop(columns=["FL_DATE"])
    frame["delay_severity"] = labels
    path = tmp_path / "no_date.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValidationError):
        read_prepared(path, max_rows=10)


def test_mlflow_export_roundtrip_and_cleanup(trained_bundle, tmp_path):
    """Catch missing serializers and read-only code copies before live training."""
    from importlib.metadata import version
    from pathlib import Path
    import tempfile

    import mlflow.pyfunc
    import mlflow.sklearn
    from src.models.train import model_export_options

    _, pipeline, frame = trained_bundle
    dependencies = {name: version(name) for name in
                    ("scikit-learn", "pandas", "numpy", "pydantic", "joblib", "cloudpickle", "mlflow-skinny")}
    example = frame.head(3)
    expected = pipeline.predict(example).tolist()
    with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
        destination = Path(directory) / "model"
        mlflow.sklearn.save_model(
            path=str(destination), **model_export_options(pipeline, example, dependencies))
        restored = mlflow.sklearn.load_model(str(destination))
        assert restored.predict(example).tolist() == expected
        generic = mlflow.pyfunc.load_model(str(destination))
        assert generic.predict(example).tolist() == expected
    assert not Path(directory).exists()
