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


def _numeric_feature_columns(config: dict) -> list[str]:
    """Return every feature expected to hold a number, AIRLINE_ID included."""
    f = config['features']
    return f['time_hhmm'] + f['temporal'] + f['numeric'] + ['AIRLINE_ID']


def _is_whole_number(values: pd.Series) -> pd.Series:
    """Return a boolean mask that is True where a value has no fractional part."""
    return values % 1 == 0


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

        Each line below is one step of the contract, in order. The step
        names say what happens; the detail lives in the helper methods.

        Raises:
            ValueError: If the input is malformed, has missing features or
                contains out-of-range values.
        """
        data = self._copy_valid_frame(X)
        data = self._rename_bts_to_eda(data)
        data = self._add_month_and_weekday(data)
        data = self._keep_required_features(data)
        data = self._coerce_numeric_columns(data)
        self._check_value_ranges(data)
        data = self._clean_categorical_text(data)
        return data

    def _copy_valid_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check the input is a DataFrame with unique columns, then return a copy."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Expected a pandas DataFrame of flight records')
        if X.columns.duplicated().any():
            raise ValueError('Duplicate column names are not allowed')
        return X.copy()

    def _rename_bts_to_eda(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rename BTS column names to their EDA equivalents."""
        for bts_name, eda_name in self.config['features']['bts_to_eda_mapping'].items():
            if bts_name == eda_name or bts_name not in data:
                continue
            if eda_name in data:
                raise ValueError(f'Provide only one of {bts_name} and {eda_name}')
            data = data.rename(columns={bts_name: eda_name})
        return data

    def _add_month_and_weekday(self, data: pd.DataFrame) -> pd.DataFrame:
        """Derive MONTH and DAY_OF_WEEK from FL_DATE when they are not supplied."""
        calendar_missing = any(col not in data for col in ['MONTH', 'DAY_OF_WEEK'])
        if 'FL_DATE' not in data or not calendar_missing:
            return data
        
        try:
            dates = pd.to_datetime(data['FL_DATE'], format='%Y-%m-%d', errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError('FL_DATE must use YYYY-MM-DD') from exc
        
        if dates.isna().any():
            raise ValueError('FL_DATE cannot be missing when deriving calendar features')
        if 'MONTH' not in data:
            data['MONTH'] = dates.dt.month
        if 'DAY_OF_WEEK' not in data:
            # pandas counts weekdays from 0 (Monday); BTS counts from 1 (Monday).
            data['DAY_OF_WEEK'] = dates.dt.day_of_week + 1
        return data

    def _keep_required_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Keep only the model feature columns, failing if any are missing."""
        required = feature_columns(self.config)
        missing = sorted(set(required) - set(data.columns))
        if missing:
            raise ValueError(f'Missing required features: {missing}')
        return data[required].copy()

    def _coerce_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert every numeric column to finite float64, failing otherwise."""
        for col in _numeric_feature_columns(self.config):
            try:
                data[col] = pd.to_numeric(data[col], errors='raise').astype('float64')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{col} must be numeric') from exc
            if not np.isfinite(data[col].to_numpy()).all():
                raise ValueError(f'{col} must contain finite, non-missing numbers')
        return data

    def _check_value_ranges(self, data: pd.DataFrame) -> None:
        """Check HHMM times, calendar fields, airline id and distance are in range."""
        for col in self.config['features']['time_hhmm']:
            hhmm = data[col]
            within_clock = (hhmm >= 0) & (hhmm <= 2359) & (hhmm % 100 < 60)  # minutes 00-59
            valid = _is_whole_number(hhmm) & (within_clock | (hhmm == 2400))
            if not valid.all():
                raise ValueError(f'{col} must be valid HHMM (0000-2359 or 2400)')
            
        for col, upper in [('MONTH', 12), ('DAY_OF_WEEK', 7)]:
            in_range = _is_whole_number(data[col]) & data[col].between(1, upper)
            if not in_range.all():
                raise ValueError(f'{col} must be an integer from 1 to {upper}')
            
        airline = data['AIRLINE_ID']
        if not (_is_whole_number(airline) & (airline > 0)).all():
            raise ValueError('AIRLINE_ID must be a positive integer identifier')
        if not (data['DISTANCE'] > 0).all():
            raise ValueError('DISTANCE must be positive')

    def _clean_categorical_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalise categorical columns to clean strings (AIRLINE_ID as a label)."""
        for col in self.config['features']['categorical']:
            if data[col].isna().any():
                raise ValueError(f'{col} cannot be missing')
            if col == 'AIRLINE_ID':
                # Treat the airline id as a label, not a magnitude.
                data[col] = data[col].astype('int64').astype(str)
                continue
            is_non_empty_string = data[col].map(lambda value: isinstance(value, str) and value.strip() != '')
            if not is_non_empty_string.all():
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
        """Convert each HHMM value to its hour of day (2400 wraps to 0)."""
        check_is_fitted(self)
        hhmm = np.asarray(X, dtype=float)
        hour = hhmm // 100   # 1345 -> 13
        return hour % 24     # 2400 -> 0

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return names in the form '<column>_hour'."""
        if input_features is None:
            input_features = [f'time_{i}' for i in range(self.n_features_in_)]
        return np.asarray([f'{name}_hour' for name in input_features], dtype=object)


def build_preprocessor(config: dict, linear: bool = False) -> ColumnTransformer:
    """Build the column transformer for one model family.

    Two strategies, chosen by ``linear``:

    - Linear models (LogisticRegression) assume each input contributes in
      proportion to its value, so categories must NOT look ordered and the
      numbers must share a scale. We one-hot every category (the hour
      included) and standard-scale the numeric columns.
    - Tree models (RandomForest) split on thresholds and ignore scale, so a
      compact ordinal code plus the raw numbers is enough, and cheaper.

    Unknown categories seen only at inference never raise: one-hot ignores
    them, the ordinal encoder maps them to -1.
    """
    f = config['features']

    # Scheduled hour. Trees can split the raw 0-23 integer; linear models get
    # it one-hot, so that hour 23 is not treated as "more" than hour 1.
    hour_steps = [('extract_hour', HourExtractor())]
    if linear:
        hour_steps.append(('encode', OneHotEncoder(handle_unknown='ignore')))

    # Airports, airlines and states: one-hot for linear, ordinal for trees.
    if linear:
        categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    else:
        categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # Distance: standard-scaled for linear, passed through untouched for trees.
    numeric_encoder = StandardScaler() if linear else 'passthrough'

    return ColumnTransformer([
        ('time_hhmm', Pipeline(hour_steps), f['time_hhmm']),
        ('temporal', OneHotEncoder(handle_unknown='ignore'), f['temporal']),
        ('categorical', categorical_encoder, f['categorical']),
        ('numeric', numeric_encoder, f['numeric']),
    ], remainder='drop',
       # Linear: keep the one-hot output sparse (memory). Trees: force it dense.
       sparse_threshold=1.0 if linear else 0.0)


def build_pipeline(config: dict, model: BaseEstimator) -> Pipeline:
    """Assemble the full pipeline: input adapter -> preprocessor -> classifier."""
    return Pipeline([
        ('input', FlightInputAdapter(config)),
        ('preprocessor', build_preprocessor(config, linear=isinstance(model, LogisticRegression))),
        ('classifier', model),
    ])


def to_bts_frame(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rename EDA columns back to their original BTS names.

    The config stores the mapping as BTS -> EDA, so we invert it to
    EDA -> BTS and rename the columns back. Used to build the MLflow model
    signature and the example request, which both use raw BTS names.
    """
    bts_to_eda = config['features']['bts_to_eda_mapping']
    eda_to_bts = {eda_name: bts_name for bts_name, eda_name in bts_to_eda.items()}

    # Columns arrive with EDA names; rename the ones that have a BTS original.
    result = frame[feature_columns(config)].rename(columns=eda_to_bts).copy()

    # Re-assert float64 on the numeric columns, under whatever name they now carry.
    for eda_name in _numeric_feature_columns(config):
        column_name = eda_to_bts.get(eda_name, eda_name)  # BTS name if renamed, else unchanged
        result[column_name] = result[column_name].astype('float64')
    return result