import os

from dotenv import load_dotenv

from app.models import EmergencyEvent

load_dotenv()


def build_alert_message(event: EmergencyEvent) -> str:
    reading = event.latest_reading
    return (
        "GlucoRelay DEMO alert: possible glucose emergency. "
        f"Reading: {reading.value_mg_dl} mg/dL; "
        f"trend: {reading.trend.value}; "
        f"reason: {event.reason}. "
        f"Event ID: {event.id}"
    )


def notify_emergency_contact(event: EmergencyEvent) -> dict[str, str]:
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
        raise RuntimeError(
            "SMS is enabled but these environment variables are missing: "
            + ", ".join(missing)
        )

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
