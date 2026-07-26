from app import main as main_module
from app.models import EventStatus, PatientCheckInAnalysis


def _create_check_in_event(client, value: int = 67) -> str:
    response = client.post(
        "/api/readings", json={"value_mg_dl": value, "trend": "single_down"}
    )
    return response.json()["event"]["id"]


def _stub_analysis(monkeypatch, action: str, **fields) -> None:
    def fake_analyze(transcript: str):
        return (
            PatientCheckInAnalysis(
                action=action, summary=f"stub-{action}", detected_language="en", **fields
            ),
            "gemma",
            None,
        )

    monkeypatch.setattr(main_module, "analyze_patient_checkin", fake_analyze)


def test_complex_help_request_moves_to_contacting_with_handoff(client, monkeypatch):
    event_id = _create_check_in_event(client)
    _stub_analysis(
        monkeypatch,
        "need_help",
        responsive=True,
        requested_contact="Luis",
        reported_condition="confused",
        supply_location="red backpack",
    )

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={
            "transcript": (
                "I'm awake, but I feel confused. Please contact Luis. "
                "My glucagon is in my red backpack."
            )
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["analysis"]["action"] == "need_help"
    assert data["proposed_tool"]["name"] == "request_caregiver_help"
    assert data["validated_tool"]["name"] == "request_caregiver_help"
    assert data["event"]["status"] == EventStatus.CONTACTING.value
    assert data["event"]["requested_contact"] == "Luis"
    assert data["event"]["reported_condition"] == "confused"
    assert data["event"]["supply_location"] == "red backpack"
    assert data["handoff"] is not None
    assert data["handoff"]["headline"]
    assert data["source"]["interpretation"] == "gemma"
    assert data["source"]["handoff"] in {"gemma", "fallback"}
    assert data["notification"] is not None
    assert data["gemma_trace"]["interpretation_source"] == "gemma"

    stages = [step["stage"] for step in data["gemma_trace"]["steps"]]
    assert stages == [
        "input_received",
        "interpretation_started",
        "interpretation_completed",
        "tool_proposed",
        "tool_validated",
        "tool_executed",
        "handoff_started",
        "handoff_completed" if data["source"]["handoff"] == "gemma" else "handoff_failed",
    ]


def test_treating_response_with_reported_action_moves_to_monitoring(client, monkeypatch):
    event_id = _create_check_in_event(client)
    _stub_analysis(monkeypatch, "treating", reported_action="drank juice")

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I drank some juice and I'm treating it."},
    )
    data = response.json()
    assert data["event"]["status"] == EventStatus.MONITORING.value
    assert data["validated_tool"]["name"] == "record_patient_treating"
    assert data["event"]["reported_action"] == "drank juice"


def test_okay_response_moves_to_monitoring(client, monkeypatch):
    event_id = _create_check_in_event(client)
    _stub_analysis(monkeypatch, "okay", responsive=True)

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I'm okay, check on me again soon."},
    )
    data = response.json()
    assert data["event"]["status"] == EventStatus.MONITORING.value
    assert data["validated_tool"]["name"] == "record_patient_okay"


def test_false_alarm_resolves_event(client, monkeypatch):
    event_id = _create_check_in_event(client)
    _stub_analysis(monkeypatch, "false_alarm")

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "This was a false alarm, cancel it."},
    )
    data = response.json()
    assert data["event"]["status"] == EventStatus.RESOLVED.value
    assert data["validated_tool"]["name"] == "resolve_false_alarm"

    state = client.get("/api/state").json()
    assert state["active_event"] is None


def test_schedule_recheck_records_minutes_and_schedules_one_timer(client, monkeypatch):
    from app.store import store

    event_id = _create_check_in_event(client)
    _stub_analysis(monkeypatch, "schedule_recheck", follow_up_minutes=10)

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "Everything is okay, but check on me again in ten minutes."},
    )
    data = response.json()
    assert data["event"]["status"] == EventStatus.MONITORING.value
    assert data["validated_tool"]["name"] == "schedule_patient_recheck"
    assert data["validated_tool"]["arguments"] == {"minutes": 10}
    assert data["event"]["check_in_deadline"] is not None

    event_types = [e["event_type"] for e in data["timeline"]]
    assert "patient_scheduled_recheck" in event_types
    assert "recheck_timer_scheduled" in event_types

    # Exactly one recheck timer is pending for this event - no duplicates.
    assert len(store._timeout_tasks) == 1
    task = store._timeout_tasks[event_id]
    assert not task.done()


def test_unknown_response_keeps_check_in_required(client, monkeypatch):
    event_id = _create_check_in_event(client)
    _stub_analysis(monkeypatch, "unknown")

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "What's the weather like today?"},
    )
    data = response.json()
    assert data["event"]["status"] == EventStatus.CHECK_IN_REQUIRED.value
    assert data["validated_tool"]["name"] == "report_unclear_response"


def test_voice_check_in_rejected_after_check_in_over(client, monkeypatch):
    event_id = _create_check_in_event(client)
    _stub_analysis(monkeypatch, "treating")
    client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I'm treating it"},
    )

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I'm okay now"},
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["current_status"] == EventStatus.MONITORING.value


def test_duplicate_voice_submissions_do_not_duplicate_escalation_or_alerts(
    client, monkeypatch
):
    event_id = _create_check_in_event(client)
    _stub_analysis(monkeypatch, "need_help", requested_contact="Luis")

    first = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I need help, please call Luis."},
    )
    assert first.status_code == 200
    assert first.json()["notification"] is not None

    second = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I need help, please call Luis."},
    )
    assert second.status_code == 409

    state = client.get("/api/state").json()
    assert len(state["active_event"]["notification_attempts"]) == 1


def test_voice_check_in_rejects_empty_transcript(client):
    event_id = _create_check_in_event(client)
    response = client.post(
        f"/api/events/{event_id}/voice-check-in", json={"transcript": "   "}
    )
    assert response.status_code == 400


def test_voice_check_in_rejects_excessively_long_transcript(client):
    event_id = _create_check_in_event(client)
    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "a" * 5000},
    )
    assert response.status_code == 400


def test_gemma_unavailable_uses_deterministic_fallback(client, monkeypatch):
    event_id = _create_check_in_event(client)

    import app.gemma_service as gemma_service

    monkeypatch.setattr(gemma_service, "_get_client_and_model", lambda: (None, None))

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I'm treating it"},
    )
    data = response.json()
    assert data["source"]["interpretation"] == "fallback"
    assert data["analysis"]["action"] == "treating"
    assert data["event"]["status"] == EventStatus.MONITORING.value

    event_types = [entry["event_type"] for entry in data["timeline"]]
    assert "fallback_parser_used" in event_types
    trace_stages = [step["stage"] for step in data["gemma_trace"]["steps"]]
    assert "fallback_used" in trace_stages


def test_gemma_invalid_output_uses_fallback(client, monkeypatch):
    event_id = _create_check_in_event(client)

    class FakeResponse:
        parsed = None
        text = "not valid json"

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    import app.gemma_service as gemma_service

    monkeypatch.setattr(
        gemma_service, "_get_client_and_model", lambda: (FakeClient(), "gemma-4")
    )

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "false alarm, cancel it"},
    )
    data = response.json()
    assert data["source"]["interpretation"] == "fallback"
    assert data["analysis"]["action"] == "false_alarm"


def test_tool_arguments_cannot_introduce_invented_facts(client, monkeypatch):
    event_id = _create_check_in_event(client)
    _stub_analysis(monkeypatch, "need_help", requested_contact="Luis")

    from app.models import ProposedToolCall

    monkeypatch.setattr(
        main_module,
        "propose_tool",
        lambda analysis: ProposedToolCall(
            name="request_caregiver_help", arguments={"requested_contact": "Someone Else"}
        ),
    )

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I need help, please call Luis."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["validated_tool"] is None
    assert data["event"]["status"] == EventStatus.CHECK_IN_REQUIRED.value

    trace_stages = [step["stage"] for step in data["gemma_trace"]["steps"]]
    assert "tool_rejected" in trace_stages
