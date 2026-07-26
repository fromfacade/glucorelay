import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.engine import Decision, DemoThresholds, evaluate_reading
from app.models import (
    AppState,
    CaregiverAcknowledgementIn,
    EmergencyEvent,
    EventStatus,
    PatientResponseIn,
    Reading,
    ReadingIn,
    Thresholds,
)
from app.notifications import notify_emergency_contact
from app.store import store

load_dotenv()

app = FastAPI(
    title="GlucoRelay Prototype API",
    version="0.1.0",
    description=(
        "Hackathon-only simulated emergency coordination prototype. "
        "Not a medical device."
    ),
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

thresholds = DemoThresholds(
    low=int(os.getenv("DEMO_LOW_THRESHOLD", "70")),
    urgent_low=int(os.getenv("DEMO_URGENT_LOW_THRESHOLD", "55")),
    high=int(os.getenv("DEMO_HIGH_THRESHOLD", "250")),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_event_or_404(event_id: str) -> EmergencyEvent:
    event = store.active_event
    if not event or event.id != event_id:
        raise HTTPException(status_code=404, detail="Active event not found")
    return event


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
def ingest_reading(payload: ReadingIn) -> dict:
    reading = Reading(**payload.model_dump())
    store.save_reading(reading)

    decision, reason = evaluate_reading(reading, thresholds)

    # Avoid opening duplicate events while one is already active.
    if store.active_event:
        store.active_event.latest_reading = reading
        store.active_event.updated_at = utc_now()

        # A later urgent reading can escalate an existing check-in.
        if (
            decision == Decision.CONTACT_NOW
            and store.active_event.status
            in {EventStatus.CHECK_IN_REQUIRED, EventStatus.MONITORING}
        ):
            store.active_event.status = EventStatus.CONTACTING
            store.active_event.reason = reason
            notification = notify_emergency_contact(store.active_event)
        else:
            notification = None

        return {
            "reading": reading,
            "decision": decision,
            "event": store.active_event,
            "notification": notification,
        }

    if decision == Decision.NONE:
        return {
            "reading": reading,
            "decision": decision,
            "event": None,
        }

    status = (
        EventStatus.CONTACTING
        if decision == Decision.CONTACT_NOW
        else EventStatus.CHECK_IN_REQUIRED
    )

    event = EmergencyEvent(
        status=status,
        reason=reason,
        latest_reading=reading,
    )
    store.save_event(event)

    notification = None
    if status == EventStatus.CONTACTING:
        notification = notify_emergency_contact(event)

    return {
        "reading": reading,
        "decision": decision,
        "event": event,
        "notification": notification,
    }


@app.post("/api/events/{event_id}/patient-response")
def patient_response(
    event_id: str,
    payload: PatientResponseIn,
) -> EmergencyEvent | dict:
    event = get_event_or_404(event_id)
    event.patient_response = payload.response
    event.updated_at = utc_now()

    if payload.response == "treating":
        event.status = EventStatus.MONITORING
        return event

    if payload.response == "need_help":
        event.status = EventStatus.CONTACTING
        notification = notify_emergency_contact(event)
        return {"event": event, "notification": notification}

    # false_alarm
    event.status = EventStatus.RESOLVED
    store.archive_active_event()
    return event


@app.post("/api/events/{event_id}/timeout")
def simulate_no_response(event_id: str) -> dict:
    event = get_event_or_404(event_id)

    if event.status not in {
        EventStatus.CHECK_IN_REQUIRED,
        EventStatus.MONITORING,
    }:
        raise HTTPException(
            status_code=409,
            detail="This event cannot be escalated from its current status",
        )

    event.status = EventStatus.CONTACTING
    event.reason = f"{event.reason}; patient check-in unanswered"
    event.updated_at = utc_now()
    notification = notify_emergency_contact(event)
    return {"event": event, "notification": notification}


@app.post("/api/events/{event_id}/acknowledge")
def acknowledge_event(
    event_id: str,
    payload: CaregiverAcknowledgementIn,
) -> EmergencyEvent:
    event = get_event_or_404(event_id)

    if event.status != EventStatus.CONTACTING:
        raise HTTPException(
            status_code=409,
            detail="Event is not currently awaiting caregiver acknowledgement",
        )

    event.status = EventStatus.ACKNOWLEDGED
    event.acknowledged_by = payload.caregiver_name
    event.updated_at = utc_now()
    return event


@app.post("/api/events/{event_id}/resolve")
def resolve_event(event_id: str) -> EmergencyEvent:
    event = get_event_or_404(event_id)
    event.status = EventStatus.RESOLVED
    event.updated_at = utc_now()
    resolved = event.model_copy(deep=True)
    store.archive_active_event()
    return resolved


@app.post("/api/reset")
def reset_demo() -> dict[str, str]:
    store.latest_reading = None
    store.active_event = None
    store.history = []
    return {"status": "reset"}
