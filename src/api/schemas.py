"""Request and response contracts for the /classify and /labels endpoints.

The schema is deliberately permissive: the scoring service may send nulls,
omit fields or add fields we do not use, and a single bad instance must never
reject the whole batch. Everything is optional here; missing or invalid
values are imputed in the API layer before prediction.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DelayCategory = Literal['on_time', 'moderate_delay', 'severe_delay']


class FlightInstance(BaseModel):
    """One flight as sent by the leaderboard, with original BTS field names.

    Fields are declared only to document what the model consumes; all are
    optional and untyped so no value can ever fail validation.
    """

    model_config = ConfigDict(extra='allow')

    FL_DATE: Any = None
    OP_CARRIER_AIRLINE_ID: Any = None
    ORIGIN: Any = None
    ORIGIN_STATE_ABR: Any = None
    DEST: Any = None
    DEST_STATE_ABR: Any = None
    CRS_DEP_TIME: Any = None
    CRS_ARR_TIME: Any = None
    DISTANCE: Any = None


class ClassifyRequest(BaseModel):
    """A batch of flights to classify, as sent by the leaderboard."""

    model_config = ConfigDict(extra='allow')

    request_id: str | None = None
    mode: str | None = None
    instances: list[FlightInstance] = Field(default_factory=list)


class ClassifyResponse(BaseModel):
    """One delay category per input flight, in the same order as the request."""

    classifications: list[DelayCategory]


class LabelsRequest(BaseModel):
    """True outcomes delivered by the leaderboard after scoring.

    Accepts any shape; the endpoint only needs to acknowledge it with HTTP 200
    and keep it (by mode) for monitoring and drift detection.
    """

    model_config = ConfigDict(extra='allow')

    request_id: str | None = None
    mode: str | None = None
    labels: list[Any] | None = None
    outcomes: Any = None