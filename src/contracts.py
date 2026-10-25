"""Shared BTS input contract for training and serving; no outcome fields."""
from datetime import date
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CLASSES = ("on_time", "moderate_delay", "severe_delay")
DelayClass = Literal["on_time", "moderate_delay", "severe_delay"]
SCHEMA_VERSION = 1


def hhmm_to_minutes(value: str | int) -> int:
    if isinstance(value, bool):
        raise ValueError("Scheduled time must be an HHMM integer or string")
    if isinstance(value, str):
        if not value.isascii() or not value.isdigit() or not 1 <= len(value) <= 4:
            raise ValueError("Scheduled time must contain 1 to 4 HHMM digits")
        value = int(value)
    if not isinstance(value, int):
        raise ValueError("Scheduled time must be an HHMM integer or string")
    if value == 2400:
        return 0
    if value < 0 or value >= 2400 or value % 100 >= 60:
        raise ValueError("Scheduled time must be valid HHMM (0000 to 2359, or 2400)")
    return (value // 100) * 60 + value % 100


class FlightRecord(BaseModel):
    # Evaluation may supply additional BTS columns. Only this explicit allowlist
    # is retained; outcomes such as DEP_DELAY and ARR_DELAY_NEW cannot be features.
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    FL_DATE: date
    OP_CARRIER_AIRLINE_ID: str
    ORIGIN: str
    DEST: str
    CRS_DEP_TIME: str | int
    CRS_ARR_TIME: str | int
    DISTANCE: Annotated[float, Field(gt=0)]
    CRS_ELAPSED_TIME: Annotated[float, Field(gt=0)] | None = None

    @field_validator("FL_DATE", mode="before")
    @classmethod
    def flight_date(cls, value):
        if type(value) is date:
            return value
        if not isinstance(value, str):
            raise ValueError("FL_DATE must be an ISO date YYYY-MM-DD")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("FL_DATE must be an ISO date YYYY-MM-DD") from exc
        if parsed.isoformat() != value:
            raise ValueError("FL_DATE must be an ISO date YYYY-MM-DD")
        return parsed

    @field_validator("OP_CARRIER_AIRLINE_ID", mode="before")
    @classmethod
    def carrier_id(cls, value):
        if isinstance(value, bool):
            raise ValueError("Airline ID must be a positive integer or digit string")
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, str) or not value.isascii() or not value.isdigit() or int(value) <= 0:
            raise ValueError("Airline ID must be a positive integer or digit string")
        return str(int(value))

    @field_validator("ORIGIN", "DEST")
    @classmethod
    def airport(cls, value):
        if len(value) != 3 or not value.isascii() or not value.isalpha():
            raise ValueError("Airport code must contain three ASCII letters")
        return value.upper()

    @field_validator("CRS_DEP_TIME", "CRS_ARR_TIME", mode="before")
    @classmethod
    def scheduled_time(cls, value):
        hhmm_to_minutes(value)
        return value

    @field_validator("DISTANCE", "CRS_ELAPSED_TIME", mode="before")
    @classmethod
    def finite_number(cls, value):
        if value is None:
            return value
        if isinstance(value, bool):
            raise ValueError("Expected a finite number")
        try:
            if not math.isfinite(float(value)):
                raise ValueError("Expected a finite number")
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected a finite number") from exc
        return value


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Local resource limit, not a limit specified by the assignment.
    instances: Annotated[list[FlightRecord], Field(min_length=1, max_length=1000)]


class ClassifyResponse(BaseModel):
    classifications: list[DelayClass]
