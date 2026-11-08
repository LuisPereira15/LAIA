"""Request and response contracts for the /classify endpoint."""

from datetime import date
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from src.api.settings import settings

# Non-empty text and whitespace
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Scheduled local time in HHMM format
HHMM = Annotated[int, Field(ge=0, le=2400)]

DelayCategory = Literal['on_time', 'moderate_delay', 'severe_delay']


class FlightInstance(BaseModel):
    """One flight, described with the original BTS column names.

    Only attributes known before the scheduled departure are accepted.
    Extra columns are allowed so the API can report them as ignored.
    """

    # Allow extra columns to be present in the request, but ignore them in the model
    model_config = ConfigDict(extra='allow')

    FL_DATE: date = Field(description='Flight date, YYYY-MM-DD')
    OP_CARRIER_AIRLINE_ID: int = Field(gt=0, description='Operating airline identifier')
    ORIGIN: NonEmptyStr = Field(description='Origin airport code, e.g. ATL')
    ORIGIN_STATE_ABR: NonEmptyStr = Field(description='Origin state, e.g. GA')
    DEST: NonEmptyStr = Field(description='Destination airport code')
    DEST_STATE_ABR: NonEmptyStr = Field(description='Destination state')
    CRS_DEP_TIME: HHMM = Field(description='Scheduled departure time, HHMM')
    CRS_ARR_TIME: HHMM = Field(description='Scheduled arrival time, HHMM')
    DISTANCE: float = Field(gt=0, description='Distance in miles')


class ClassifyRequest(BaseModel):
    """Batch of flights to classify."""

    # Number of instances is limited to avoid memory issues.
    instances: list[FlightInstance] = Field(min_length=1, max_length=settings.max_instances)

    # The model_config is used to provide an example of a valid request in the API
    model_config = ConfigDict(json_schema_extra={'examples': [{'instances': [{
        'FL_DATE': '2025-05-15',
        'OP_CARRIER_AIRLINE_ID': 19805,
        'ORIGIN': 'ATL',
        'ORIGIN_STATE_ABR': 'GA',
        'DEST': 'LAX',
        'DEST_STATE_ABR': 'CA',
        'CRS_DEP_TIME': 1430,
        'CRS_ARR_TIME': 1615,
        'DISTANCE': 1946.0,
    }]}]})


class ClassifyResponse(BaseModel):
    """One delay category per input flight, in the same order as the request."""

    classifications: list[DelayCategory]