import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.engine import Decision, DemoThresholds, evaluate_reading
from app.gemma_service import (
    analyze_patient_checkin,
    describe_semantic_correction,
    generate_caregiver_handoff,
    get_default_caregiver_name,
)
from app.gemma_trace import add_trace_step, new_trace
from app.models import (
    AppState,
    CaregiverAcknowledgementIn,
    CaregiverEventView,
    CaregiverReadingView,
    EmergencyEvent,
    EventStatus,
    LocationIn,
    NotificationAttempt,
    PatientResponseIn,
    Reading,
    ReadingIn,
    Thresholds,
    VoiceCheckInRequest,
)
from app.notifications import notify_emergency_contact
from app.store import store
from app.timeline import add_timeline_entry
from app.tools import ToolValidationError, execute_validated_tool, propose_tool, validate_tool_call
from app.transitions import (
    InvalidTransitionError,
    allowed_next_statuses,
    ensure_transition_allowed,
)

load_dotenv()

MAX_TRANSCRIPT_LENGTH = 2000

# Maps a validated patient intent (from buttons or Gemma/fallback voice
# interpretation) to the single status it is allowed to move an event to.
# Both the button-based and voice-based endpoints route through this same
# table so transition rules are never duplicated.
ACTION_TO_STATUS: dict[str, EventStatus] = {
    "okay": EventStatus.MONITORING,
    "treating": EventStatus.MONITORING,
    "need_help": EventStatus.CONTACTING,
    "false_alarm": EventStatus.RESOLVED,
}

# Timeline entries recorded when a validated voice-check-in tool executes.
TOOL_TIMELINE: dict[str, tuple[str, str]] = {
    "record_patient_okay": ("patient_reported_okay", "Patient reported they are okay (voice)."),
    "record_patient_treating": (
        "patient_reported_treatment",
        "Patient reported they are treating the reading (voice).",
    ),
    "request_caregiver_help": ("patient_requested_help", "Patient requested help (voice)."),
    "schedule_patient_recheck": (
        "patient_scheduled_recheck",
        "Patient asked to be checked on again later (voice).",
    ),
    "resolve_false_alarm": ("event_resolved", "Patient marked this as a false alarm (voice)."),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    store.cancel_timeout_task_for_all()


app = FastAPI(
    title="GlucoRelay Prototype API",
    version="0.2.0",
    description=(
        "Hackathon-only simulated emergency coordination prototype. "
        "Not a medical device. Glucose thresholds are decided by "
        "deterministic Python logic only; Gemma only interprets what the "
        "patient says during a voice check-in."
    ),
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

thresholds = DemoThresholds(
    low=int(os.getenv("DEMO_LOW_THRESHOLD", "70")),
    urgent_low=int(os.getenv("DEMO_URGENT_LOW_THRESHOLD", "55")),
    high=int(os.getenv("DEMO_HIGH_THRESHOLD", "250")),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_check_in_timeout_seconds() -> int:
    return int(os.getenv("CHECK_IN_TIMEOUT_SECONDS", "30"))


def get_event_or_404(event_id: str) -> EmergencyEvent:
    event = store.active_event
    if not event or event.id != event_id:
        raise HTTPException(status_code=404, detail="Active event not found")
    return event


def get_caregiver_event_or_404(public_token: str) -> EmergencyEvent:
    event = store.get_by_public_token(public_token)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


def transition_conflict(event: EmergencyEvent, attempted_action: str) -> HTTPException:
    allowed = sorted(s.value for s in allowed_next_statuses(event.status))
    add_timeline_entry(
        event,
        "invalid_transition_attempted",
        f"Rejected '{attempted_action}' while event is '{event.status.value}'.",
    )
    return HTTPException(
        status_code=409,
        detail={
            "message": (
                f"Cannot perform '{attempted_action}' while event is "
                f"'{event.status.value}'."
            ),
            "current_status": event.status.value,
            "allowed_transitions": allowed,
        },
    )


def advance_event_status(
    event: EmergencyEvent, target: EventStatus, attempted_action: str
) -> bool:
    """Moves `event` to `target`. Returns False (no-op) if already there.

    Raises a 409 HTTPException if the transition is not permitted.
    """
    if event.status == target:
        return False
    try:
        ensure_transition_allowed(event.status, target)
    except InvalidTransitionError:
        raise transition_conflict(event, attempted_action) from None
    event.status = target
    event.updated_at = utc_now()
    return True


def attempt_caregiver_alert(event: EmergencyEvent) -> dict:
    """Notifies the caregiver at most once per escalation.

    Never raises: notification failures are recorded on the timeline and
    returned as a safe result so the API never crashes.
    """
    if event.caregiver_alert_sent_at is not None:
        return {
            "delivery": "skipped_duplicate",
            "message": "Caregiver alert already sent for this escalation.",
        }

    event.caregiver_alert_sent_at = utc_now()
    add_timeline_entry(event, "caregiver_alert_attempted", "Attempting to notify caregiver.")

    result = notify_emergency_contact(event)
    delivery = result.get("delivery", "failed")
    event.notification_attempts.append(
        NotificationAttempt(
            delivery=delivery,
            detail=result.get("message") or result.get("error"),
        )
    )

    if delivery == "failed":
        add_timeline_entry(
            event,
            "caregiver_alert_failed",
            result.get("error", "Caregiver alert could not be delivered."),
        )
    else:
        add_timeline_entry(
            event,
            "caregiver_alert_succeeded",
            f"Caregiver alert delivered via {delivery}.",
        )
    return result


def build_caregiver_view(event: EmergencyEvent) -> CaregiverEventView:
    maps_url = None
    if event.location_latitude is not None and event.location_longitude is not None:
        maps_url = (
            f"https://www.google.com/maps?q={event.location_latitude},"
            f"{event.location_longitude}"
        )
    return CaregiverEventView(
        status=event.status,
        reason=event.reason,
        opened_at=event.opened_at,
        updated_at=event.updated_at,
        latest_reading=CaregiverReadingView(
            value_mg_dl=event.latest_reading.value_mg_dl,
            trend=event.latest_reading.trend,
            recorded_at=event.latest_reading.recorded_at,
        ),
        caregiver_handoff_headline=(
            event.caregiver_handoff.headline if event.caregiver_handoff else None
        ),
        caregiver_handoff=(
            event.caregiver_handoff.handoff if event.caregiver_handoff else None
        ),
        patient_response_summary=event.patient_response_summary,
        requested_contact=(
            event.requested_contact
            or (event.caregiver_handoff.requested_contact if event.caregiver_handoff else None)
            or (
                get_default_caregiver_name()
                if event.status
                in {EventStatus.CONTACTING, EventStatus.ACKNOWLEDGED, EventStatus.RESOLVED}
                else None
            )
        ),
        reported_condition=event.reported_condition,
        reported_action=event.reported_action,
        supply_location=event.supply_location,
        check_in_deadline=event.check_in_deadline,
        caregiver_alert_sent_at=event.caregiver_alert_sent_at,
        acknowledged_by=event.acknowledged_by,
        acknowledged_at=event.acknowledged_at,
        resolved_at=event.resolved_at,
        maps_url=maps_url,
        timeline=event.timeline,
    )


def apply_acknowledgement(event: EmergencyEvent, caregiver_name: str) -> bool:
    """Returns False (no-op) if already acknowledged; raises 409 if invalid."""
    if event.status == EventStatus.ACKNOWLEDGED:
        return False
    if event.status != EventStatus.CONTACTING:
        raise transition_conflict(event, "acknowledge")
    event.status = EventStatus.ACKNOWLEDGED
    event.acknowledged_by = caregiver_name
    event.acknowledged_at = utc_now()
    event.updated_at = utc_now()
    add_timeline_entry(
        event, "caregiver_acknowledged", f"{caregiver_name} acknowledged the alert."
    )
    return True


# --- Check-in deadline timer -------------------------------------------
#
# HACKATHON NOTE: this uses an in-process asyncio task as a lightweight
# timer appropriate for a single-process demo. A production deployment
# would need a durable job queue (e.g. Celery/RQ/cloud scheduler) so
# deadlines survive process restarts and work across multiple workers.


def schedule_check_in_timeout(event: EmergencyEvent) -> None:
    timeout_seconds = get_check_in_timeout_seconds()
    event.check_in_deadline = utc_now() + timedelta(seconds=timeout_seconds)
    task = asyncio.create_task(
        _run_timeout(
            event.id,
            timeout_seconds,
            expected_status=EventStatus.CHECK_IN_REQUIRED,
            timeline_event_type="check_in_timed_out",
            timeout_message="Patient did not respond before the check-in deadline.",
            reason_suffix="check-in deadline expired",
        )
    )
    store.register_timeout_task(event.id, task)


def schedule_patient_recheck_timeout(event: EmergencyEvent, minutes: int) -> None:
    """Schedules the same prototype timer mechanism used for check-ins to
    follow up on an explicit patient recheck request. If the event is still
    in "monitoring" once the requested window elapses (i.e. nothing else
    resolved or escalated it in the meantime), it is escalated to caregiver
    contact as a precaution. `register_timeout_task` cancels any existing
    timer for this event first, so at most one timer is ever pending.
    """
    timeout_seconds = minutes * 60
    event.check_in_deadline = utc_now() + timedelta(seconds=timeout_seconds)
    task = asyncio.create_task(
        _run_timeout(
            event.id,
            timeout_seconds,
            expected_status=EventStatus.MONITORING,
            timeline_event_type="recheck_window_elapsed",
            timeout_message=(
                f"The patient's requested {minutes}-minute recheck window "
                "elapsed without further update."
            ),
            reason_suffix="requested recheck window elapsed",
        )
    )
    store.register_timeout_task(event.id, task)


async def _run_timeout(
    event_id: str,
    timeout_seconds: int,
    expected_status: EventStatus,
    timeline_event_type: str,
    timeout_message: str,
    reason_suffix: str,
) -> None:
    try:
        await asyncio.sleep(timeout_seconds)
    except asyncio.CancelledError:
        return

    event = store.active_event
    if not event or event.id != event_id:
        return
    if event.status != expected_status:
        return

    add_timeline_entry(event, timeline_event_type, timeout_message)
    event.reason = f"{event.reason}; {reason_suffix}"
    advance_event_status(event, EventStatus.CONTACTING, "check-in-deadline")
    await asyncio.to_thread(attempt_caregiver_alert, event)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/caregiver/{public_token}", include_in_schema=False)
def caregiver_page(public_token: str) -> FileResponse:
    return FileResponse(STATIC_DIR / "caregiver.html")


@app.get("/api/state", response_model=AppState)
def get_state() -> AppState:
    return AppState(
        latest_reading=store.latest_reading,
        active_event=store.active_event,
        history=store.history,
        thresholds=Thresholds(
            low=thresholds.low,
            urgent_low=thresholds.urgent_low,
            high=thresholds.high,
        ),
    )


@app.post("/api/readings")
async def ingest_reading(payload: ReadingIn) -> dict:
    reading = Reading(**payload.model_dump())
    store.save_reading(reading)

    decision, reason = evaluate_reading(reading, thresholds)

    # Avoid opening duplicate events while one is already active - a later
    # reading updates the existing event instead.
    if store.active_event:
        event = store.active_event
        event.latest_reading = reading
        event.updated_at = utc_now()
        add_timeline_entry(
            event,
            "glucose_reading_received",
            f"New reading {reading.value_mg_dl} mg/dL ({reading.trend.value}) "
            "while an event is active.",
        )

        notification = None
        if (
            decision == Decision.CONTACT_NOW
            and event.status in {EventStatus.CHECK_IN_REQUIRED, EventStatus.MONITORING}
        ):
            store.cancel_timeout_task(event.id)
            event.reason = reason
            advance_event_status(event, EventStatus.CONTACTING, "urgent-reading")
            add_timeline_entry(
                event,
                "urgent_reading_escalation",
                f"A new urgent reading escalated the event: {reason}.",
            )
            notification = await asyncio.to_thread(attempt_caregiver_alert, event)

        return {
            "reading": reading,
            "decision": decision,
            "event": event,
            "notification": notification,
        }

    if decision == Decision.NONE:
        return {"reading": reading, "decision": decision, "event": None}

    status = (
        EventStatus.CONTACTING
        if decision == Decision.CONTACT_NOW
        else EventStatus.CHECK_IN_REQUIRED
    )

    event = EmergencyEvent(status=status, reason=reason, latest_reading=reading)
    add_timeline_entry(
        event,
        "glucose_reading_received",
        f"Reading {reading.value_mg_dl} mg/dL ({reading.trend.value}) triggered evaluation.",
    )
    add_timeline_entry(event, "emergency_event_created", f"Emergency event created: {reason}.")
    store.save_event(event)

    notification = None
    if status == EventStatus.CONTACTING:
        notification = await asyncio.to_thread(attempt_caregiver_alert, event)
    else:
        add_timeline_entry(event, "patient_check_in_requested", "Patient check-in requested.")
        schedule_check_in_timeout(event)

    return {
        "reading": reading,
        "decision": decision,
        "event": event,
        "notification": notification,
    }


@app.post("/api/events/{event_id}/patient-response")
def patient_response(event_id: str, payload: PatientResponseIn) -> dict:
    event = get_event_or_404(event_id)
    target = ACTION_TO_STATUS[payload.response]
    changed = advance_event_status(event, target, f"patient-response:{payload.response}")
    event.patient_response = payload.response

    notification = None
    if changed:
        store.cancel_timeout_task(event.id)
        if payload.response == "treating":
            add_timeline_entry(
                event,
                "patient_reported_treatment",
                "Patient reported they are treating the reading (button).",
            )
        elif payload.response == "need_help":
            add_timeline_entry(
                event, "patient_requested_help", "Patient requested help (button)."
            )
            notification = attempt_caregiver_alert(event)
        elif payload.response == "false_alarm":
            add_timeline_entry(
                event, "event_resolved", "Patient marked this as a false alarm (button)."
            )

    if event.status == EventStatus.RESOLVED:
        event.resolved_at = utc_now()
        resolved_snapshot = event.model_copy(deep=True)
        store.archive_active_event()
        return {"event": resolved_snapshot, "notification": notification}

    return {"event": event, "notification": notification}


@app.post("/api/events/{event_id}/voice-check-in")
async def voice_check_in(event_id: str, payload: VoiceCheckInRequest) -> dict:
    event = get_event_or_404(event_id)

    if event.status != EventStatus.CHECK_IN_REQUIRED:
        raise transition_conflict(event, "voice-check-in")

    transcript = payload.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript must not be empty.")
    if len(transcript) > MAX_TRANSCRIPT_LENGTH:
        raise HTTPException(status_code=400, detail="Transcript is too long.")

    trace = event.gemma_trace or new_trace()
    event.gemma_trace = trace

    # Stage 1-2: record the transcript.
    event.patient_transcript = transcript
    add_timeline_entry(
        event,
        "voice_transcript_received",
        f"Received a {len(transcript)}-character voice transcript.",
    )
    add_trace_step(trace, "input_received", "Voice transcript received.")

    # Stage 3: Gemma (or fallback) understands the transcript.
    add_trace_step(trace, "interpretation_started", "Requesting patient check-in analysis.")
    analysis, interp_source, interp_note = await asyncio.to_thread(
        analyze_patient_checkin, transcript
    )
    trace.interpretation_source = interp_source
    trace.model_name = os.getenv("GEMMA_MODEL", "").strip() or trace.model_name
    trace.original_language = analysis.detected_language

    if interp_source == "fallback":
        add_timeline_entry(
            event,
            "fallback_parser_used",
            "Used the deterministic fallback parser instead of Gemma.",
            metadata={"reason": interp_note} if interp_note else None,
        )
        add_trace_step(
            trace,
            "fallback_used",
            "Used the deterministic fallback parser instead of Gemma.",
            metadata={"reason": interp_note} if interp_note else None,
        )
    else:
        add_timeline_entry(event, "gemma_interpretation_completed", analysis.summary)
        add_trace_step(trace, "interpretation_completed", analysis.summary)

        # A semantic correction is a backend safety check catching Gemma
        # over-inferring an action, NOT a Gemma failure - `interp_source`
        # stays "gemma" and this gets its own distinct trace/timeline stage.
        if interp_note and interp_note.startswith("semantic_correction:"):
            reasons = interp_note.split(":", 1)[1].split(",")
            for reason in reasons:
                description = describe_semantic_correction(reason)
                add_timeline_entry(event, "semantic_correction_applied", description)
                add_trace_step(trace, "semantic_correction_applied", description)

    event.patient_check_in_analysis = analysis
    event.patient_response_summary = analysis.summary
    event.english_patient_summary = analysis.english_summary or analysis.summary
    event.original_language = analysis.detected_language
    event.patient_response = analysis.action
    if analysis.requested_contact:
        event.requested_contact = analysis.requested_contact
    if analysis.reported_condition:
        event.reported_condition = analysis.reported_condition
    if analysis.reported_action:
        event.reported_action = analysis.reported_action
    if analysis.supply_location:
        event.supply_location = analysis.supply_location
    event.updated_at = utc_now()

    # Stage 4: Gemma proposes exactly one constrained tool (Python-derived
    # from the already-constrained `action` field - see app.tools).
    proposed_tool = propose_tool(analysis)
    event.proposed_tool_call = proposed_tool
    trace.proposed_tool = proposed_tool
    add_trace_step(trace, "tool_proposed", f"Proposed tool: {proposed_tool.name}.")

    # Stage 5: backend validates the tool before anything may execute.
    try:
        validated_tool = validate_tool_call(event, analysis, proposed_tool)
    except ToolValidationError as exc:
        add_timeline_entry(
            event,
            "invalid_transition_attempted",
            f"Rejected proposed tool '{proposed_tool.name}': {exc}",
        )
        add_trace_step(trace, "tool_rejected", str(exc))
        event.validated_tool_call = None
        return {
            "transcript": transcript,
            "analysis": analysis,
            "proposed_tool": proposed_tool,
            "validated_tool": None,
            "handoff": None,
            "source": {"interpretation": interp_source, "handoff": "not_generated"},
            "event": event,
            "timeline": event.timeline,
            "gemma_trace": trace,
            "notification": None,
        }

    event.validated_tool_call = validated_tool
    trace.validated_tool = validated_tool
    add_trace_step(trace, "tool_validated", f"Validated tool: {validated_tool.name}.")

    # Stage 6: execute the validated tool (deterministic state change).
    changed = execute_validated_tool(event, validated_tool)
    if changed:
        store.cancel_timeout_task(event.id)
        timeline_event = TOOL_TIMELINE.get(validated_tool.name)
        recheck_minutes = validated_tool.arguments.get("minutes")
        if timeline_event:
            event_type, message = timeline_event
            if validated_tool.name == "schedule_patient_recheck" and recheck_minutes:
                message = f"{message} (in {recheck_minutes} minutes)."
            add_timeline_entry(event, event_type, message)
        if validated_tool.name == "schedule_patient_recheck" and recheck_minutes:
            schedule_patient_recheck_timeout(event, recheck_minutes)
            add_timeline_entry(
                event,
                "recheck_timer_scheduled",
                f"A recheck reminder was scheduled in {recheck_minutes} minute(s).",
            )
        add_trace_step(
            trace,
            "tool_executed",
            f"Executed tool: {validated_tool.name}.",
            metadata={"minutes": recheck_minutes} if recheck_minutes else None,
        )
    elif validated_tool.name == "report_unclear_response":
        add_timeline_entry(
            event,
            "invalid_transition_attempted",
            "Voice check-in could not be classified; check-in remains active.",
        )
        add_trace_step(trace, "tool_executed", "No state change: response was unclear.")

    # Stage 7-8: only escalations to caregiver contact get a grounded handoff.
    notification = None
    handoff_payload = None
    handoff_source_for_response: str = "not_generated"
    if changed and validated_tool.name == "request_caregiver_help":
        add_trace_step(trace, "handoff_started", "Generating caregiver handoff.")
        handoff, handoff_source, handoff_failure = await asyncio.to_thread(
            generate_caregiver_handoff, event, analysis
        )
        event.caregiver_handoff = handoff
        event.caregiver_handoff_source = handoff_source
        trace.handoff_source = handoff_source
        handoff_payload = handoff
        handoff_source_for_response = handoff_source

        if handoff_source == "fallback":
            add_timeline_entry(
                event,
                "caregiver_alert_attempted",
                "Gemma handoff unavailable; used a deterministic handoff instead.",
                metadata={"reason": handoff_failure} if handoff_failure else None,
            )
            add_trace_step(
                trace,
                "handoff_failed",
                "Gemma handoff unavailable; used a deterministic handoff instead.",
                metadata={"reason": handoff_failure} if handoff_failure else None,
            )
        else:
            add_trace_step(trace, "handoff_completed", handoff.headline)

        notification = await asyncio.to_thread(attempt_caregiver_alert, event)

    response_event = event
    if event.status == EventStatus.RESOLVED:
        event.resolved_at = utc_now()
        response_event = event.model_copy(deep=True)
        store.archive_active_event()

    return {
        "transcript": transcript,
        "analysis": analysis,
        "proposed_tool": proposed_tool,
        "validated_tool": validated_tool,
        "handoff": handoff_payload,
        "source": {"interpretation": interp_source, "handoff": handoff_source_for_response},
        "event": response_event,
        "timeline": response_event.timeline,
        "gemma_trace": response_event.gemma_trace,
        "notification": notification,
    }


@app.post("/api/events/{event_id}/location")
def set_event_location(event_id: str, payload: LocationIn) -> EmergencyEvent:
    event = get_event_or_404(event_id)
    event.location_latitude = payload.latitude
    event.location_longitude = payload.longitude
    event.updated_at = utc_now()
    add_timeline_entry(
        event, "location_updated", "Patient location was attached to the event."
    )
    return event


@app.post("/api/events/{event_id}/timeout")
async def simulate_no_response(event_id: str) -> dict:
    event = get_event_or_404(event_id)

    if event.status not in {EventStatus.CHECK_IN_REQUIRED, EventStatus.MONITORING}:
        raise transition_conflict(event, "timeout")

    store.cancel_timeout_task(event.id)
    event.reason = f"{event.reason}; patient check-in unanswered"
    advance_event_status(event, EventStatus.CONTACTING, "timeout")
    add_timeline_entry(
        event, "check_in_timed_out", "Manually simulated a missed check-in."
    )
    notification = await asyncio.to_thread(attempt_caregiver_alert, event)
    return {"event": event, "notification": notification}


@app.post("/api/events/{event_id}/acknowledge")
def acknowledge_event(event_id: str, payload: CaregiverAcknowledgementIn) -> EmergencyEvent:
    event = get_event_or_404(event_id)
    apply_acknowledgement(event, payload.caregiver_name)
    return event


@app.post("/api/events/{event_id}/resolve")
def resolve_event(event_id: str) -> dict:
    if store.active_event and store.active_event.id == event_id:
        event = store.active_event
        advance_event_status(event, EventStatus.RESOLVED, "resolve")
        event.resolved_at = utc_now()
        add_timeline_entry(event, "event_resolved", "Event manually resolved.")
        store.cancel_timeout_task(event.id)
        resolved = event.model_copy(deep=True)
        store.archive_active_event()
        return {"event": resolved, "already_resolved": False}

    for past_event in store.history:
        if past_event.id == event_id:
            return {"event": past_event, "already_resolved": True}

    raise HTTPException(status_code=404, detail="Event not found")


@app.get("/api/caregiver/events/{public_token}", response_model=CaregiverEventView)
def get_caregiver_event(public_token: str) -> CaregiverEventView:
    event = get_caregiver_event_or_404(public_token)
    return build_caregiver_view(event)


@app.post(
    "/api/caregiver/events/{public_token}/acknowledge",
    response_model=CaregiverEventView,
)
def caregiver_acknowledge(
    public_token: str, payload: CaregiverAcknowledgementIn
) -> CaregiverEventView:
    event = get_caregiver_event_or_404(public_token)
    apply_acknowledgement(event, payload.caregiver_name)
    return build_caregiver_view(event)


@app.post("/api/reset")
def reset_demo() -> dict[str, str]:
    store.reset()
    return {"status": "reset"}
