from app import main as main_module
from app.models import EventStatus, PatientCheckInAnalysis


def _create_check_in_event(client, value: int = 67) -> str:
    response = client.post(
        "/api/readings", json={"value_mg_dl": value, "trend": "single_down"}
    )
    return response.json()["event"]["id"]


def test_spanish_transcript_produces_english_caregiver_handoff(client, monkeypatch):
    event_id = _create_check_in_event(client)

    def fake_analyze(transcript: str):
        return (
            PatientCheckInAnalysis(
                action="need_help",
                summary="Estoy despierta, pero me siento confundida. Por favor llama a Luis.",
                english_summary="I'm awake, but I feel confused. Please call Luis.",
                detected_language="es",
                responsive=True,
                reported_condition="confused",
                requested_contact="Luis",
            ),
            "gemma",
            None,
        )

    monkeypatch.setattr(main_module, "analyze_patient_checkin", fake_analyze)

    response = client.post(
        f"/api/events/{event_id}/voice-check-in",
        json={
            "transcript": (
                "Estoy despierta, pero me siento confundida. Por favor llama a Luis."
            )
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["analysis"]["detected_language"] == "es"
    assert data["analysis"]["action"] == "need_help"
    assert data["analysis"]["requested_contact"] == "Luis"
    assert data["event"]["original_language"] == "es"
    assert data["event"]["english_patient_summary"].startswith("I'm awake")
    assert data["event"]["status"] == EventStatus.CONTACTING.value

    # The original transcript remains available internally.
    assert data["event"]["patient_transcript"].startswith("Estoy despierta")

    # Handoff must be in English regardless of the original language, and
    # must not invent anything beyond what was explicitly stated.
    handoff = data["handoff"]
    assert handoff is not None
    for spanish_word in ("estoy", "despierta", "confundida", "llama"):
        assert spanish_word not in handoff["handoff"].lower()
    assert handoff["requested_contact"] == "Luis"
    assert handoff["detected_language"] == "es"
