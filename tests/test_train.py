from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.pipeline import (
    FlightInputAdapter,
    build_pipeline,
    feature_columns,
)


@pytest.fixture
def config():
    path = Path("src/config/config.yaml")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def valid_bts_frame():
    return pd.DataFrame(
        [
            {
                "FL_DATE": "2022-01-03",
                "OP_CARRIER_AIRLINE_ID": 19805,
                "ORIGIN": "ATL",
                "ORIGIN_STATE_ABR": "GA",
                "DEST": "LAX",
                "DEST_STATE_ABR": "CA",
                "CRS_DEP_TIME": 830,
                "CRS_ARR_TIME": 1030,
                "DISTANCE": 1946.0,
            },
            {
                "FL_DATE": "2022-01-04",
                "OP_CARRIER_AIRLINE_ID": 19977,
                "ORIGIN": "JFK",
                "ORIGIN_STATE_ABR": "NY",
                "DEST": "SFO",
                "DEST_STATE_ABR": "CA",
                "CRS_DEP_TIME": 1200,
                "CRS_ARR_TIME": 1515,
                "DISTANCE": 2586.0,
            },
            {
                "FL_DATE": "2022-01-05",
                "OP_CARRIER_AIRLINE_ID": 19393,
                "ORIGIN": "ORD",
                "ORIGIN_STATE_ABR": "IL",
                "DEST": "DFW",
                "DEST_STATE_ABR": "TX",
                "CRS_DEP_TIME": 1600,
                "CRS_ARR_TIME": 1830,
                "DISTANCE": 802.0,
            },
        ]
    )


def test_feature_columns(config):
    columns = feature_columns(config)

    assert columns == [
        "DEPARTURE_TIME",
        "ARRIVAL_TIME",
        "MONTH",
        "DAY_OF_WEEK",
        "AIRLINE_ID",
        "ORIGIN_AIRPORT",
        "ORIGIN_STATE",
        "DEST_AIRPORT",
        "DEST_STATE",
        "DISTANCE",
    ]


def test_input_adapter_accepts_valid_bts_fields(config, valid_bts_frame):
    adapter = FlightInputAdapter(config)

    transformed = adapter.fit_transform(valid_bts_frame)

    assert list(transformed.columns) == feature_columns(config)
    assert len(transformed) == len(valid_bts_frame)


def test_input_adapter_derives_calendar_features(config, valid_bts_frame):
    adapter = FlightInputAdapter(config)

    transformed = adapter.fit_transform(valid_bts_frame)

    assert transformed["MONTH"].tolist() == [1.0, 1.0, 1.0]
    assert transformed["DAY_OF_WEEK"].tolist() == [1.0, 2.0, 3.0]


def test_input_adapter_rejects_invalid_distance(config, valid_bts_frame):
    invalid = valid_bts_frame.copy()
    invalid.loc[0, "DISTANCE"] = 0

    adapter = FlightInputAdapter(config)

    with pytest.raises(ValueError, match="DISTANCE must be positive"):
        adapter.fit_transform(invalid)


def test_pipeline_can_train_and_predict(config, valid_bts_frame):
    y = pd.Series(
        ["on_time", "moderate_delay", "severe_delay"],
        name="ARR_DELAY_CATEGORY",
    )

    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
    )

    pipeline = build_pipeline(config, model)
    pipeline.fit(valid_bts_frame, y)

    predictions = pipeline.predict(valid_bts_frame)

    assert len(predictions) == len(valid_bts_frame)
    assert set(predictions).issubset(
        set(config["features"]["target_classes"])
    )


def test_pipeline_handles_unseen_categories(config, valid_bts_frame):
    y = pd.Series(
        ["on_time", "moderate_delay", "severe_delay"],
        name="ARR_DELAY_CATEGORY",
    )

    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
    )

    pipeline = build_pipeline(config, model)
    pipeline.fit(valid_bts_frame, y)

    unseen = valid_bts_frame.iloc[[0]].copy()
    unseen.loc[unseen.index[0], "ORIGIN"] = "ZZZ"
    unseen.loc[unseen.index[0], "DEST"] = "YYY"
    unseen.loc[unseen.index[0], "OP_CARRIER_AIRLINE_ID"] = 99999

    predictions = pipeline.predict(unseen)

    assert len(predictions) == 1
    assert predictions[0] in config["features"]["target_classes"]