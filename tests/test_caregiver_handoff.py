from app.gemma_service import generate_caregiver_handoff
from app.models import EmergencyEvent, EventStatus, PatientCheckInAnalysis, Reading


def _make_event(**overrides) -> EmergencyEvent:
    reading = Reading(value_mg_dl=52, trend="double_down")
    defaults = dict(
        status=EventStatus.CONTACTING,
        reason="Demo urgent-low threshold reached",
        latest_reading=reading,
        patient_transcript="I feel confused, please call Luis.",
    )
    defaults.update(overrides)
    return EmergencyEvent(**defaults)


def _analysis(**fields) -> PatientCheckInAnalysis:
    defaults = dict(action="need_help", summary="Patient asked for help.")
    defaults.update(fields)
    return PatientCheckInAnalysis(**defaults)


def test_fallback_handoff_uses_only_verified_facts_no_gemma_configured():
    event = _make_event()
    analysis = _analysis(
        responsive=True,
        requested_contact="Luis",
        reported_condition="confused",
        supply_location="red backpack",
    )

    handoff, source, reason = generate_caregiver_handoff(event, analysis)

    assert source == "fallback"
    assert handoff.requested_contact == "Luis"
    assert handoff.reported_condition == "confused"
    assert handoff.supply_location == "red backpack"
    assert handoff.location_shared is False
    assert "52 mg/dL" in handoff.handoff
    assert "Location was not shared." in handoff.unknown_information


def test_fallback_handoff_never_invents_missing_facts():
    event = _make_event()
    analysis = _analysis()  # no contact, condition, or location stated

    handoff, source, _ = generate_caregiver_handoff(event, analysis)

    # No contact was stated, so the generic default caregiver name is used
    # for display - this is a configured default, not an invented fact.
    assert handoff.requested_contact == "Helper"
    assert handoff.supply_location is None
    assert "Reported condition was not stated." in handoff.unknown_information
    assert "Supply location was not stated." in handoff.unknown_information


def test_fallback_handoff_defaults_to_helper_when_no_contact_stated(monkeypatch):
    event = _make_event()
    analysis = _analysis()  # no contact stated

    handoff, _source, _reason = generate_caregiver_handoff(event, analysis)

    assert handoff.requested_contact == "Helper"
    assert "Helper" in handoff.handoff
    assert any("Helper" in item for item in handoff.unknown_information)


def test_fallback_handoff_default_name_is_configurable(monkeypatch):
    monkeypatch.setenv("DEFAULT_CAREGIVER_NAME", "Backup Contact")
    event = _make_event()
    analysis = _analysis()

    handoff, _source, _reason = generate_caregiver_handoff(event, analysis)

    assert handoff.requested_contact == "Backup Contact"


def test_explicit_contact_name_overrides_the_default_helper_name():
    event = _make_event()
    analysis = _analysis(requested_contact="Luis")

    handoff, _source, _reason = generate_caregiver_handoff(event, analysis)

    assert handoff.requested_contact == "Luis"
    assert not any("Helper" in item for item in handoff.unknown_information)


def test_handoff_generation_failure_uses_deterministic_fallback(monkeypatch):
    import app.gemma_service as gemma_service

    class FailingModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("simulated network failure")

    class FailingClient:
        models = FailingModels()

    monkeypatch.setattr(
        gemma_service, "_get_client_and_model", lambda: (FailingClient(), "gemma-4")
    )

    event = _make_event()
    analysis = _analysis(requested_contact="Luis")

    handoff, source, reason = generate_caregiver_handoff(event, analysis)
    assert source == "fallback"
    assert reason == "gemma_handoff_request_failed"
    assert handoff.requested_contact == "Luis"


def test_handoff_disabled_via_env_uses_fallback(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMMA_HANDOFF", "false")
    event = _make_event()
    analysis = _analysis()
    handoff, source, reason = generate_caregiver_handoff(event, analysis)
    assert source == "fallback"
    assert reason == "gemma_handoff_disabled"
