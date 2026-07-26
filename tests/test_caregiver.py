from app import main as main_module
from app.models import PatientCheckInAnalysis


def _create_contacting_event(client) -> dict:
    response = client.post(
        "/api/readings", json={"value_mg_dl": 50, "trend": "double_down"}
    )
    return response.json()["event"]


def _create_contacting_event_via_voice(client, monkeypatch) -> dict:
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "single_down"}
    )
    event_id = response.json()["event"]["id"]

    def fake_analyze(transcript: str):
        return (
            PatientCheckInAnalysis(
                action="need_help",
                summary="Patient asked for help.",
                requested_contact="Luis",
                reported_condition="confused",
            ),
            "gemma",
            None,
        )

    monkeypatch.setattr(main_module, "analyze_patient_checkin", fake_analyze)
    voice_response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I need help, call Luis, I feel confused."},
    )
    return voice_response.json()["event"]


def test_caregiver_acknowledge_only_from_contacting(client):
    event = _create_contacting_event(client)
    token = event["public_token"]

    response = client.post(
        f"/api/caregiver/events/{token}/acknowledge",
        json={"caregiver_name": "Luis"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "acknowledged"
    assert data["acknowledged_by"] == "Luis"


def test_duplicate_acknowledgement_does_not_duplicate_timeline(client):
    event = _create_contacting_event(client)
    token = event["public_token"]

    first = client.post(
        f"/api/caregiver/events/{token}/acknowledge",
        json={"caregiver_name": "Luis"},
    )
    second = client.post(
        f"/api/caregiver/events/{token}/acknowledge",
        json={"caregiver_name": "Luis"},
    )
    assert first.status_code == 200
    assert second.status_code == 200

    ack_entries = [
        e for e in second.json()["timeline"] if e["event_type"] == "caregiver_acknowledged"
    ]
    assert len(ack_entries) == 1


def test_acknowledge_rejected_outside_contacting(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    token = response.json()["event"]["public_token"]

    response = client.post(
        f"/api/caregiver/events/{token}/acknowledge",
        json={"caregiver_name": "Luis"},
    )
    assert response.status_code == 409


def test_event_id_alone_does_not_grant_caregiver_access(client):
    event = _create_contacting_event(client)
    event_id = event["id"]

    response = client.get(f"/api/caregiver/events/{event_id}")
    assert response.status_code == 404


def test_public_caregiver_token_works_and_hides_internal_id(client):
    event = _create_contacting_event(client)
    token = event["public_token"]

    response = client.get(f"/api/caregiver/events/{token}")
    assert response.status_code == 200
    data = response.json()
    assert "id" not in data
    assert "public_token" not in data
    assert data["latest_reading"]["value_mg_dl"] == 50


def test_caregiver_view_includes_generated_handoff(client, monkeypatch):
    event = _create_contacting_event_via_voice(client, monkeypatch)
    token = event["public_token"]

    response = client.get(f"/api/caregiver/events/{token}")
    assert response.status_code == 200
    data = response.json()

    assert data["caregiver_handoff_headline"]
    assert data["caregiver_handoff"]
    assert data["requested_contact"] == "Luis"
    assert data["reported_condition"] == "confused"


def test_caregiver_view_defaults_requested_contact_to_helper(client, monkeypatch):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "single_down"}
    )
    event_id = response.json()["event"]["id"]

    def fake_analyze(transcript: str):
        return (
            PatientCheckInAnalysis(
                action="need_help",
                summary="Patient asked for help.",
            ),
            "gemma",
            None,
        )

    monkeypatch.setattr(main_module, "analyze_patient_checkin", fake_analyze)
    voice_response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={"transcript": "I need help, I feel confused."},
    )
    token = voice_response.json()["event"]["public_token"]

    response = client.get(f"/api/caregiver/events/{token}")
    assert response.status_code == 200
    assert response.json()["requested_contact"] == "Helper"


def test_caregiver_view_does_not_expose_raw_internal_ai_data(client, monkeypatch):
    event = _create_contacting_event_via_voice(client, monkeypatch)
    token = event["public_token"]

    response = client.get(f"/api/caregiver/events/{token}")
    data = response.json()

    for forbidden_key in (
        "id",
        "public_token",
        "patient_check_in_analysis",
        "proposed_tool_call",
        "validated_tool_call",
        "gemma_trace",
        "patient_transcript",
        "confidence",
    ):
        assert forbidden_key not in data
