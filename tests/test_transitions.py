from app.models import EventStatus


def _create_check_in_event(client, value: int = 67) -> str:
    response = client.post(
        "/api/readings", json={"value_mg_dl": value, "trend": "single_down"}
    )
    return response.json()["event"]["id"]


def test_timeout_escalates_only_once(client):
    event_id = _create_check_in_event(client)

    first = client.post(f"/api/events/{event_id}/timeout")
    assert first.status_code == 200
    assert first.json()["event"]["status"] == EventStatus.CONTACTING.value

    second = client.post(f"/api/events/{event_id}/timeout")
    assert second.status_code == 409

    state = client.get("/api/state").json()
    assert len(state["active_event"]["notification_attempts"]) == 1


def test_invalid_transition_returns_409_with_current_and_allowed(client):
    event_id = _create_check_in_event(client)

    response = client.post(
        f"/api/events/{event_id}/acknowledge", json={"caregiver_name": "Luis"}
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["current_status"] == EventStatus.CHECK_IN_REQUIRED.value
    assert isinstance(detail["allowed_transitions"], list)
    assert len(detail["allowed_transitions"]) > 0


def test_resolving_already_resolved_event_is_clear_not_a_crash(client):
    event_id = _create_check_in_event(client)
    first = client.post(f"/api/events/{event_id}/resolve")
    assert first.status_code == 200
    assert first.json()["already_resolved"] is False

    second = client.post(f"/api/events/{event_id}/resolve")
    assert second.status_code == 200
    assert second.json()["already_resolved"] is True


def test_button_patient_response_idempotent_for_repeated_press(client):
    event_id = _create_check_in_event(client)

    first = client.post(
        f"/api/events/{event_id}/patient-response", json={"response": "treating"}
    )
    assert first.status_code == 200
    assert first.json()["event"]["status"] == EventStatus.MONITORING.value

    second = client.post(
        f"/api/events/{event_id}/patient-response", json={"response": "treating"}
    )
    assert second.status_code == 200
    assert second.json()["event"]["status"] == EventStatus.MONITORING.value

    state = client.get("/api/state").json()
    treatment_entries = [
        e
        for e in state["active_event"]["timeline"]
        if e["event_type"] == "patient_reported_treatment"
    ]
    assert len(treatment_entries) == 1
