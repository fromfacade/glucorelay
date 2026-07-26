def test_timeline_records_reading_and_check_in_lifecycle(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    event_id = response.json()["event"]["id"]

    client.post(
        f"/api/events/{event_id}/patient-response", json={"response": "treating"}
    )

    state = client.get("/api/state").json()
    event_types = [e["event_type"] for e in state["active_event"]["timeline"]]

    assert "glucose_reading_received" in event_types
    assert "emergency_event_created" in event_types
    assert "patient_check_in_requested" in event_types
    assert "patient_reported_treatment" in event_types


def test_timeline_records_escalation_and_alert_outcome(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 45, "trend": "double_down"}
    )
    event = response.json()["event"]
    event_types = [e["event_type"] for e in event["timeline"]]

    assert "emergency_event_created" in event_types
    assert "caregiver_alert_attempted" in event_types
    assert "caregiver_alert_succeeded" in event_types


def test_timeline_records_invalid_transition_attempts(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    event_id = response.json()["event"]["id"]

    client.post(f"/api/events/{event_id}/acknowledge", json={"caregiver_name": "Luis"})

    state = client.get("/api/state").json()
    event_types = [e["event_type"] for e in state["active_event"]["timeline"]]
    assert "invalid_transition_attempted" in event_types
