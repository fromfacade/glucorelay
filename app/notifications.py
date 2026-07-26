import os

from dotenv import load_dotenv

from app.gemma_service import get_default_caregiver_name
from app.models import EmergencyEvent

load_dotenv()


def _caregiver_link(event: EmergencyEvent) -> str:
    base_url = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    return f"{base_url}/caregiver/{event.public_token}"


def build_alert_message(event: EmergencyEvent) -> str:
    """Builds the caregiver alert text from only verified event data.

    Prioritizes the Gemma-or-fallback-generated `caregiver_handoff` (see
    app.gemma_service.generate_caregiver_handoff) when one exists - it is
    itself grounded only in verified facts. Falls back to the raw verified
    fields for escalations that didn't go through a voice check-in (e.g. a
    timeout or an urgent reading). Never includes medical advice.
    """
    reading = event.latest_reading
    lines = [
        "GLUCORELAY CHECK-IN ESCALATION",
        "",
        f"Current simulated reading: {reading.value_mg_dl} mg/dL",
        f"Trend: {reading.trend.value}",
        f"Reason: {event.reason}",
        "",
    ]

    if event.caregiver_handoff:
        lines.append("Patient check-in:")
        lines.append(event.caregiver_handoff.handoff)
        lines.append("")
        if event.caregiver_handoff.supply_location:
            lines.append("Reported supply location:")
            lines.append(event.caregiver_handoff.supply_location)
            lines.append("")
    elif event.patient_response_summary:
        lines.append("Patient check-in:")
        lines.append(event.patient_response_summary)
        lines.append("")
        if event.supply_location:
            lines.append("Reported supply location:")
            lines.append(event.supply_location)
            lines.append("")

    if event.location_latitude is not None and event.location_longitude is not None:
        lines.append(
            "Location: "
            f"https://www.google.com/maps?q={event.location_latitude},{event.location_longitude}"
        )
        lines.append("")

    contact = (
        (event.caregiver_handoff.requested_contact if event.caregiver_handoff else None)
        or event.requested_contact
        or get_default_caregiver_name()
    )
    lines.append(f"Notify: {contact}")
    lines.append("")

    lines.append("View and acknowledge:")
    lines.append(_caregiver_link(event))
    return "\n".join(lines)


def notify_emergency_contact(event: EmergencyEvent) -> dict[str, str]:
    """Attempts to notify the caregiver. Never raises - always returns a result dict.

    Callers are responsible for deduplication (see `event.caregiver_alert_sent_at`
    in app.main); this function performs a single delivery attempt.
    """
    message = build_alert_message(event)

    if os.getenv("ENABLE_SMS", "false").lower() != "true":
        print("\n--- SIMULATED CAREGIVER ALERT ---")
        print(message)
        print("--------------------------------\n")
        return {"delivery": "simulated", "message": message}

    required = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
        "EMERGENCY_CONTACT_NUMBER",
    ]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        return {
            "delivery": "failed",
            "error": "SMS is enabled but required configuration is missing.",
        }

    try:
        from twilio.rest import Client

        client = Client(
            os.environ["TWILIO_ACCOUNT_SID"],
            os.environ["TWILIO_AUTH_TOKEN"],
        )
        outbound = client.messages.create(
            body=message,
            from_=os.environ["TWILIO_FROM_NUMBER"],
            to=os.environ["EMERGENCY_CONTACT_NUMBER"],
        )
        return {"delivery": "twilio", "message_sid": outbound.sid}
    except Exception:
        # Never let a Twilio/network failure crash the request. No stack
        # traces or credentials are ever returned to the caller.
        return {
            "delivery": "failed",
            "error": "Unable to deliver the caregiver alert via SMS.",
        }
