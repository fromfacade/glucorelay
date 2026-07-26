import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
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


class TimelineEntry(BaseModel):
    """A single factual, application-generated event in an incident's history.

    Messages must describe what the application observed or did - never
    invented medical claims or speculation about the patient's condition.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    message: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] | None = None


class NotificationAttempt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    attempted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    delivery: Literal["simulated", "twilio", "failed", "skipped_duplicate"]
    detail: str | None = None


class PatientCheckInAnalysis(BaseModel):
    """Structured result of Gemma interpreting a patient's spoken check-in.

    Gemma never decides medical urgency - it only extracts explicitly
    stated facts and classifies the transcript into one of a small set of
    allowed intents. Every field must come directly from the transcript;
    nothing here may be inferred or invented.
    """

    action: Literal[
        "okay",
        "treating",
        "need_help",
        "false_alarm",
        "schedule_recheck",
        "unknown",
    ]
    responsive: bool | None = None
    summary: str
    requested_contact: str | None = None
    reported_condition: str | None = None
    reported_action: str | None = None
    supply_location: str | None = None
    follow_up_minutes: int | None = None
    detected_language: str | None = None
    english_summary: str | None = None
    confidence: float | None = None


class ProposedToolCall(BaseModel):
    """A single constrained application action Gemma's analysis maps to.

    Gemma never executes this - `app.tools.propose_tool` derives it
    deterministically from the already-constrained `action` field, and
    `app.tools.validate_tool_call` must approve it before anything runs.
    """

    name: Literal[
        "record_patient_okay",
        "record_patient_treating",
        "request_caregiver_help",
        "schedule_patient_recheck",
        "resolve_false_alarm",
        "report_unclear_response",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)


class CaregiverHandoff(BaseModel):
    """A grounded, caregiver-facing summary generated only from verified facts."""

    headline: str
    handoff: str
    patient_is_responsive: bool | None = None
    reported_condition: str | None = None
    reported_action: str | None = None
    requested_contact: str | None = None
    supply_location: str | None = None
    location_shared: bool
    unknown_information: list[str] = Field(default_factory=list)
    detected_language: str | None = None


GemmaTraceStage = Literal[
    "input_received",
    "interpretation_started",
    "interpretation_completed",
    "interpretation_failed",
    "fallback_used",
    "semantic_correction_applied",
    "tool_proposed",
    "tool_validated",
    "tool_rejected",
    "tool_executed",
    "handoff_started",
    "handoff_completed",
    "handoff_failed",
]


class GemmaTraceStep(BaseModel):
    """One observable, presentation-safe step in a voice check-in's Gemma pipeline.

    Never includes API keys, raw prompts, hidden reasoning, full SDK
    responses, or stack traces - only what application-level thing happened.
    """

    stage: GemmaTraceStage
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] | None = None


class GemmaTrace(BaseModel):
    interpretation_source: Literal["gemma", "fallback", "not_used"] = "not_used"
    handoff_source: Literal["gemma", "fallback", "not_generated"] = "not_generated"
    model_name: str | None = None
    original_language: str | None = None
    proposed_tool: ProposedToolCall | None = None
    validated_tool: ProposedToolCall | None = None
    steps: list[GemmaTraceStep] = Field(default_factory=list)


class EmergencyEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    public_token: str = Field(default_factory=lambda: secrets.token_urlsafe(24))
    status: EventStatus
    reason: str
    opened_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    latest_reading: Reading

    # Legacy field kept for backward compatibility with the button-based
    # patient response flow ("treating" / "need_help" / "false_alarm").
    patient_response: str | None = None

    # Voice check-in fields.
    patient_transcript: str | None = None
    patient_response_summary: str | None = None
    requested_contact: str | None = None
    reported_condition: str | None = None
    reported_action: str | None = None
    supply_location: str | None = None
    original_language: str | None = None
    english_patient_summary: str | None = None

    # Gemma pipeline artifacts (interpret -> propose tool -> handoff).
    patient_check_in_analysis: PatientCheckInAnalysis | None = None
    proposed_tool_call: ProposedToolCall | None = None
    validated_tool_call: ProposedToolCall | None = None
    caregiver_handoff: CaregiverHandoff | None = None
    caregiver_handoff_source: Literal["gemma", "fallback"] | None = None
    gemma_trace: GemmaTrace | None = None

    # Deadline / escalation bookkeeping.
    check_in_deadline: datetime | None = None
    caregiver_alert_sent_at: datetime | None = None

    # Caregiver acknowledgement.
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None

    # Optional browser-supplied location.
    location_latitude: float | None = None
    location_longitude: float | None = None

    timeline: list[TimelineEntry] = Field(default_factory=list)
    notification_attempts: list[NotificationAttempt] = Field(default_factory=list)


class PatientResponseIn(BaseModel):
    response: Literal["treating", "need_help", "false_alarm"]


class VoiceCheckInRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=2000)
    language: str | None = "en-US"


class LocationIn(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class CaregiverAcknowledgementIn(BaseModel):
    caregiver_name: str = Field(min_length=1, max_length=80)


class CaregiverReadingView(BaseModel):
    value_mg_dl: int
    trend: Trend
    recorded_at: datetime


class CaregiverEventView(BaseModel):
    """Limited, caregiver-safe projection of an EmergencyEvent.

    Deliberately excludes internal identifiers, config, and anything not
    needed for a caregiver to understand and respond to the situation.
    """

    status: EventStatus
    reason: str
    opened_at: datetime
    updated_at: datetime
    latest_reading: CaregiverReadingView
    caregiver_handoff_headline: str | None = None
    caregiver_handoff: str | None = None
    patient_response_summary: str | None = None
    requested_contact: str | None = None
    reported_condition: str | None = None
    reported_action: str | None = None
    supply_location: str | None = None
    check_in_deadline: datetime | None = None
    caregiver_alert_sent_at: datetime | None = None
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    maps_url: str | None = None
    timeline: list[TimelineEntry] = Field(default_factory=list)


class Thresholds(BaseModel):
    low: int
    urgent_low: int
    high: int


class AppState(BaseModel):
    latest_reading: Reading | None
    active_event: EmergencyEvent | None
    history: list[EmergencyEvent]
    thresholds: Thresholds
