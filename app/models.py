from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Trend(str, Enum):
    DOUBLE_DOWN = "double_down"
    SINGLE_DOWN = "single_down"
    FLAT = "flat"
    SINGLE_UP = "single_up"
    DOUBLE_UP = "double_up"
    UNKNOWN = "unknown"


class EventStatus(str, Enum):
    CHECK_IN_REQUIRED = "check_in_required"
    MONITORING = "monitoring"
    CONTACTING = "contacting"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class ReadingIn(BaseModel):
    value_mg_dl: int = Field(ge=20, le=600)
    trend: Trend = Trend.UNKNOWN
    source: Literal["simulator", "dexcom", "nightscout"] = "simulator"
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Reading(ReadingIn):
    id: str = Field(default_factory=lambda: str(uuid4()))


class EmergencyEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    status: EventStatus
    reason: str
    opened_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    latest_reading: Reading
    patient_response: str | None = None
    acknowledged_by: str | None = None


class PatientResponseIn(BaseModel):
    response: Literal["treating", "need_help", "false_alarm"]


class CaregiverAcknowledgementIn(BaseModel):
    caregiver_name: str = Field(min_length=1, max_length=80)


class Thresholds(BaseModel):
    low: int
    urgent_low: int
    high: int


class AppState(BaseModel):
    latest_reading: Reading | None
    active_event: EmergencyEvent | None
    history: list[EmergencyEvent]
    thresholds: Thresholds
