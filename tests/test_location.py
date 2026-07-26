def test_location_can_be_attached_to_event(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    event_id = response.json()["event"]["id"]

    response = client.post(
        f"/api/events/{event_id}/location",
        json={"latitude": 36.9916, "longitude": -122.0609},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["location_latitude"] == 36.9916
    assert data["location_longitude"] == -122.0609


def test_location_is_optional_and_not_invented(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    event = response.json()["event"]
    assert event["location_latitude"] is None
    assert event["location_longitude"] is None


def test_location_rejects_invalid_latitude(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    event_id = response.json()["event"]["id"]

    response = client.post(
        f"/api/events/{event_id}/location",
        json={"latitude": 999, "longitude": 0},
    )
    assert response.status_code == 400


def test_location_rejects_invalid_longitude(client):
    response = client.post(
        "/api/readings", json={"value_mg_dl": 67, "trend": "flat"}
    )
    event_id = response.json()["event"]["id"]

    response = client.post(
        f"/api/events/{event_id}/location",
        json={"latitude": 0, "longitude": -200},
    )
    assert response.status_code == 400
