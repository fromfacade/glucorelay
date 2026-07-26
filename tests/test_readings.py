from app.models import EventStatus


def test_normal_reading_creates_no_event(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 110, "trend": "flat"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["event"] is None
    assert data["decision"] == "none"


def test_low_reading_creates_check_in_required(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "single_down"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["event"]["status"] == EventStatus.CHECK_IN_REQUIRED.value
    assert data["event"]["check_in_deadline"] is not None
    assert data["event"]["public_token"]


def test_urgent_low_reading_creates_contacting(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 50, "trend": "double_down"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["event"]["status"] == EventStatus.CONTACTING.value
    assert data["notification"] is not None
    assert data["event"]["caregiver_alert_sent_at"] is not None


def test_new_reading_updates_existing_event_without_duplicate(client):
    first = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    event_id = first.json()["event"]["id"]

    second = client.post(
        "/api/readings", json={"value_mg_dl": 65, "trend": "flat"}
    )
    assert second.json()["event"]["id"] == event_id

    state = client.get("/api/state").json()
    assert state["active_event"]["id"] == event_id
    assert len(state["history"]) == 0


def test_new_urgent_reading_escalates_existing_check_in(client):
    first = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    event_id = first.json()["event"]["id"]

    second = client.post(
        "/api/readings", json={"value_mg_dl": 45, "trend": "double_down"}
    )
    data = second.json()
    assert data["event"]["id"] == event_id
    assert data["event"]["status"] == EventStatus.CONTACTING.value
    assert data["notification"] is not None
