"""Shared, serializable input contract and model-specific preprocessing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


def feature_columns(config: dict) -> list[str]:
    """Return the ordered list of model features (EDA names) from the config."""
    f = config['features']
    return f['time_hhmm'] + f['temporal'] + f['categorical'] + f['numeric']


class FlightInputAdapter(BaseEstimator, TransformerMixin):
    """Accept BTS or EDA column names and validate required features.

    Maps BTS column names to their EDA equivalents, derives MONTH
    and DAY_OF_WEEK from FL_DATE when present, validates value
    ranges, and normalises categorical strings. Extra columns are
    dropped and row order is preserved.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> 'FlightInputAdapter':
        """Validate the training input. The adapter learns no state."""
        self.transform(X)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a validated DataFrame with EDA column names and clean types.

        Raises:
            ValueError: If the input is malformed, has missing features or
                contains out-of-range values.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Expected a pandas DataFrame of flight records')

        if X.columns.duplicated().any():
            raise ValueError('Duplicate column names are not allowed')
        data = X.copy()

        for raw, renamed in self.config['features']['bts_to_eda_mapping'].items():
            if raw == renamed or raw not in data:
                continue
            if renamed in data:
                raise ValueError(f'Provide only one of {raw} and {renamed}')
            data = data.rename(columns={raw: renamed})

        if 'FL_DATE' in data and any(c not in data for c in ['MONTH', 'DAY_OF_WEEK']):
            try:
                dates = pd.to_datetime(data['FL_DATE'], format='%Y-%m-%d', errors='raise')
            except (ValueError, TypeError) as exc:
                raise ValueError('FL_DATE must use YYYY-MM-DD') from exc
            if dates.isna().any():
                raise ValueError('FL_DATE cannot be missing when deriving calendar features')
            if 'MONTH' not in data:
                data['MONTH'] = dates.dt.month
            if 'DAY_OF_WEEK' not in data:
                # pandas uses 0=Monday; BTS uses 1=Monday.
                data['DAY_OF_WEEK'] = dates.dt.day_of_week + 1
        cols = feature_columns(self.config)
        missing = sorted(set(cols) - set(data.columns))

        if missing:
            raise ValueError(f'Missing required features: {missing}')
        data = data[cols].copy()
        f = self.config['features']
        numeric = f['time_hhmm'] + f['temporal'] + f['numeric'] + ['AIRLINE_ID']

        for col in numeric:
            try:
                data[col] = pd.to_numeric(data[col], errors='raise').astype('float64')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{col} must be numeric') from exc
            if not np.isfinite(data[col].to_numpy()).all():
                raise ValueError(f'{col} must contain finite, non-missing numbers')

        for col in f['time_hhmm']:
            v = data[col]
            valid = (v % 1 == 0) & (((v >= 0) & (v <= 2359) & (v % 100 < 60)) | (v == 2400))
            if not valid.all():
                raise ValueError(f'{col} must be valid HHMM (0000-2359 or 2400)')

        for col, upper in [('MONTH', 12), ('DAY_OF_WEEK', 7)]:
            if not ((data[col] % 1 == 0) & data[col].between(1, upper)).all():
                raise ValueError(f'{col} must be an integer from 1 to {upper}')

        if not ((data['AIRLINE_ID'] % 1 == 0) & (data['AIRLINE_ID'] > 0)).all():
            raise ValueError('AIRLINE_ID must be a positive integer identifier')

        if not (data['DISTANCE'] > 0).all():
            raise ValueError('DISTANCE must be positive')

        for col in f['categorical']:
            if data[col].isna().any():
                raise ValueError(f'{col} cannot be missing')
            if col == 'AIRLINE_ID':
                # Treat the airline ID as a category, not a magnitude.
                data[col] = data[col].astype('int64').astype(str)
            else:
                if not data[col].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                    raise ValueError(f'{col} must contain non-empty strings')
                data[col] = data[col].str.strip().str.upper()
        return data

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return the output feature names (EDA names)."""
        return np.asarray(feature_columns(self.config), dtype=object)


class HourExtractor(BaseEstimator, TransformerMixin):
    """Extract the scheduled local hour (0-23) from HHMM values; 2400 maps to 0."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | None = None) -> 'HourExtractor':
        """Record the number of input columns."""
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert each HHMM value to its hour."""
        check_is_fitted(self)
        return (np.asarray(X, dtype=float) // 100) % 24

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return names in the form '<column>_hour'."""
        if input_features is None:
            input_features = [f'time_{i}' for i in range(self.n_features_in_)]
        return np.asarray([f'{name}_hour' for name in input_features], dtype=object)


def build_preprocessor(config: dict, linear: bool = False) -> ColumnTransformer:
    """Build the column transformer for a model family.

    Linear models get one-hot encoding and scaled numerics; tree models get
    ordinal encoding and raw numerics. Unknown categories never raise.
    """
    f = config['features']
    hour_steps = [('extract_hour', HourExtractor())]
    if linear:
        # Avoid a linear relationship between hour and delay probability.
        hour_steps.append(('encode', OneHotEncoder(handle_unknown='ignore')))
    categorical = (OneHotEncoder(handle_unknown='ignore') if linear else
                   OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    return ColumnTransformer([
        ('time_hhmm', Pipeline(hour_steps), f['time_hhmm']),
        ('temporal', OneHotEncoder(handle_unknown='ignore'), f['temporal']),
        ('categorical', categorical, f['categorical']),
        ('numeric', StandardScaler() if linear else 'passthrough', f['numeric']),
    ], remainder='drop', sparse_threshold=1.0 if linear else 0.0)


def build_pipeline(config: dict, model: BaseEstimator) -> Pipeline:
    """Assemble the full pipeline: input adapter -> preprocessor -> classifier."""
    return Pipeline([
        ('input', FlightInputAdapter(config)),
        ('preprocessor', build_preprocessor(config, linear=isinstance(model, LogisticRegression))),
        ('classifier', model),
    ])


def to_bts_frame(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rename EDA columns back to BTS names for the MLflow signature and API example."""
    reverse = {v: k for k, v in config['features']['bts_to_eda_mapping'].items()}
    result = frame[feature_columns(config)].rename(columns=reverse).copy()
    numeric = config['features']['time_hhmm'] + config['features']['temporal'] + config['features']['numeric'] + ['AIRLINE_ID']
    for col in numeric:
        result[reverse.get(col, col)] = result[reverse.get(col, col)].astype('float64')
    return result