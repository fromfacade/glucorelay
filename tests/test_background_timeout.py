import time

from app.models import EventStatus


def test_background_timer_escalates_after_deadline(client, monkeypatch):
    monkeypatch.setenv("CHECK_IN_TIMEOUT_SECONDS", "1")

    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    assert response.json()["event"]["status"] == EventStatus.CHECK_IN_REQUIRED.value

    time.sleep(1.5)

    state = client.get("/api/state").json()
    assert state["active_event"]["status"] == EventStatus.CONTACTING.value
    event_types = [e["event_type"] for e in state["active_event"]["timeline"]]
    assert "check_in_timed_out" in event_types


def test_background_timer_does_not_escalate_a_resolved_event(client, monkeypatch):
    monkeypatch.setenv("CHECK_IN_TIMEOUT_SECONDS", "1")

    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    event_id = response.json()["event"]["id"]

    client.post(
        f"/api/events/{event_id}/patient-response", json={"response": "false_alarm"}
    )

    time.sleep(1.5)

    state = client.get("/api/state").json()
    assert state["active_event"] is None
    assert state["history"][0]["status"] == EventStatus.RESOLVED.value
