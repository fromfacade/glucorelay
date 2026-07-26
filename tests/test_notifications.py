from app.models import CaregiverHandoff, EmergencyEvent, Reading
from app.notifications import build_alert_message, notify_emergency_contact


def _make_event() -> EmergencyEvent:
    reading = Reading(value_mg_dl=50, trend="double_down")
    return EmergencyEvent(status="contacting", reason="test escalation", latest_reading=reading)


def test_simulated_delivery_when_sms_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_SMS", "false")
    result = notify_emergency_contact(_make_event())
    assert result["delivery"] == "simulated"
    assert "message" in result


def test_missing_twilio_config_returns_safe_failure(monkeypatch):
    monkeypatch.setenv("ENABLE_SMS", "true")
    for key in (
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
        "EMERGENCY_CONTACT_NUMBER",
    ):
        monkeypatch.delenv(key, raising=False)

    result = notify_emergency_contact(_make_event())
    assert result["delivery"] == "failed"
    assert "error" in result


def test_twilio_exception_does_not_crash_and_returns_safe_error(monkeypatch):
    monkeypatch.setenv("ENABLE_SMS", "true")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "sid")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+10000000000")
    monkeypatch.setenv("EMERGENCY_CONTACT_NUMBER", "+19999999999")

    class FailingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated network outage")

    monkeypatch.setattr("twilio.rest.Client", FailingClient)

    result = notify_emergency_contact(_make_event())
    assert result["delivery"] == "failed"
    assert "error" in result
    assert "simulated network outage" not in result["error"]


def test_alert_message_prioritizes_caregiver_handoff_when_present():
    event = _make_event()
    event.caregiver_handoff = CaregiverHandoff(
        headline="Patient requested caregiver assistance",
        handoff="The patient is responsive but reported confusion and asked for Luis.",
        requested_contact="Luis",
        supply_location="red backpack",
        location_shared=False,
        unknown_information=["Location was not shared."],
    )
    message = build_alert_message(event)
    assert "reported confusion and asked for Luis" in message
    assert "red backpack" in message


def test_alert_message_falls_back_to_raw_summary_without_handoff():
    event = _make_event()
    event.patient_response_summary = "Patient said they need help."
    message = build_alert_message(event)
    assert "Patient said they need help." in message


def test_alert_message_defaults_contact_to_helper_when_none_stated():
    event = _make_event()
    message = build_alert_message(event)
    assert "Notify: Helper" in message


def test_alert_message_uses_handoffs_requested_contact_when_present():
    event = _make_event()
    event.caregiver_handoff = CaregiverHandoff(
        headline="Patient requested caregiver assistance",
        handoff="The patient asked that Luis be contacted.",
        requested_contact="Luis",
        location_shared=False,
    )
    message = build_alert_message(event)
    assert "Notify: Luis" in message


def test_alert_message_uses_events_requested_contact_without_handoff():
    event = _make_event()
    event.requested_contact = "Luis"
    message = build_alert_message(event)
    assert "Notify: Luis" in message
