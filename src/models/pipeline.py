"""Small integration baseline. Model selection on real data remains separate."""
import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.contracts import FlightRecord, hhmm_to_minutes

CATEGORICAL = ["carrier", "origin", "destination", "month", "weekday"]
NUMERIC = ["dep_sin", "dep_cos", "arr_sin", "arr_cos", "distance", "scheduled_duration"]


class FlightFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        records = X.astype(object).where(X.notna(), None).to_dict(orient="records") if isinstance(X, pd.DataFrame) else X
        rows = []
        for record in records:
            flight = FlightRecord.model_validate(record)
            dep = hhmm_to_minutes(flight.CRS_DEP_TIME) * 2 * math.pi / 1440
            arr = hhmm_to_minutes(flight.CRS_ARR_TIME) * 2 * math.pi / 1440
            rows.append({
                "carrier": flight.OP_CARRIER_AIRLINE_ID,
                "origin": flight.ORIGIN,
                "destination": flight.DEST,
                "month": str(flight.FL_DATE.month),
                "weekday": str(flight.FL_DATE.isoweekday()),
                "dep_sin": math.sin(dep), "dep_cos": math.cos(dep),
                "arr_sin": math.sin(arr), "arr_cos": math.cos(arr),
                "distance": flight.DISTANCE,
                "scheduled_duration": flight.CRS_ELAPSED_TIME if flight.CRS_ELAPSED_TIME is not None else np.nan,
            })
        return pd.DataFrame(rows, columns=CATEGORICAL + NUMERIC)


def make_pipeline(seed=42):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", numeric, NUMERIC),
    ])
    return Pipeline([
        ("features", FlightFeatures()),
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
    ])
