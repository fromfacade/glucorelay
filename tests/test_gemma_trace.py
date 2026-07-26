import json

from app import main as main_module
from app.models import EventStatus, PatientCheckInAnalysis


def _create_check_in_event(client, value: int = 67) -> str:
    response = client.post(
        "/api/readings", json={"value_mg_dl": value, "trend": "single_down"}
    )
    return response.json()["event"]["id"]


def test_trace_contains_expected_stage_sequence_for_escalation(client, monkeypatch):
    event_id = _create_check_in_event(client)

    def fake_analyze(transcript: str):
        return (
            PatientCheckInAnalysis(
                action="need_help",
                summary="Patient asked for help.",
                requested_contact="Luis",
                detected_language="en",
            ),
            "gemma",
            None,
        )

    monkeypatch.setattr(main_module, "analyze_patient_checkin", fake_analyze)

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I need help, call Luis."},
    )
    trace = response.json()["gemma_trace"]
    stages = [step["stage"] for step in trace["steps"]]

    assert stages[:6] == [
        "input_received",
        "interpretation_started",
        "interpretation_completed",
        "tool_proposed",
        "tool_validated",
        "tool_executed",
    ]
    assert "handoff_started" in stages
    assert trace["interpretation_source"] == "gemma"
    assert trace["proposed_tool"]["name"] == "request_caregiver_help"
    assert trace["validated_tool"]["name"] == "request_caregiver_help"


def test_trace_does_not_expose_secrets_or_internal_details(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key-value")
    monkeypatch.setenv("GEMMA_MODEL", "gemma-4-demo")

    event_id = _create_check_in_event(client)

    def fake_analyze(transcript: str):
        return (
            PatientCheckInAnalysis(action="okay", summary="Patient said they are okay."),
            "gemma",
            None,
        )

    monkeypatch.setattr(main_module, "analyze_patient_checkin", fake_analyze)

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I'm okay"},
    )
    body_text = json.dumps(response.json())

    assert "super-secret-key-value" not in body_text
    assert "SYSTEM_INSTRUCTION" not in body_text
    assert "narrow interpretation tool" not in body_text
