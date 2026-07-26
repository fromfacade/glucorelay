import pytest

from app.models import EmergencyEvent, EventStatus, PatientCheckInAnalysis, ProposedToolCall, Reading
from app.tools import (
    ToolValidationError,
    execute_validated_tool,
    propose_tool,
    validate_tool_call,
)


def _make_event(status: EventStatus = EventStatus.CHECK_IN_REQUIRED) -> EmergencyEvent:
    reading = Reading(value_mg_dl=67, trend="flat")
    return EmergencyEvent(status=status, reason="test", latest_reading=reading)


def _analysis(action: str, **fields) -> PatientCheckInAnalysis:
    return PatientCheckInAnalysis(action=action, summary="stub", **fields)


def test_propose_tool_maps_each_action_to_its_tool():
    assert propose_tool(_analysis("okay")).name == "record_patient_okay"
    assert propose_tool(_analysis("treating")).name == "record_patient_treating"
    assert propose_tool(_analysis("need_help")).name == "request_caregiver_help"
    assert propose_tool(_analysis("schedule_recheck")).name == "schedule_patient_recheck"
    assert propose_tool(_analysis("false_alarm")).name == "resolve_false_alarm"
    assert propose_tool(_analysis("unknown")).name == "report_unclear_response"


def test_propose_tool_includes_minutes_argument_sourced_from_follow_up_minutes():
    tool = propose_tool(_analysis("schedule_recheck", follow_up_minutes=15))
    assert tool.arguments == {"minutes": 15}


def test_validate_tool_call_accepts_legitimate_proposal():
    event = _make_event()
    analysis = _analysis("need_help", requested_contact="Luis")
    proposed = propose_tool(analysis)
    validated = validate_tool_call(event, analysis, proposed)
    assert validated.name == "request_caregiver_help"


def test_validate_tool_call_rejects_unsupported_tool_name():
    event = _make_event()
    analysis = _analysis("need_help")
    bogus = ProposedToolCall.model_construct(name="delete_all_events", arguments={})
    with pytest.raises(ToolValidationError):
        validate_tool_call(event, analysis, bogus)


def test_validate_tool_call_rejects_status_that_does_not_allow_tool():
    event = _make_event(status=EventStatus.MONITORING)
    analysis = _analysis("need_help")
    proposed = propose_tool(analysis)
    with pytest.raises(ToolValidationError):
        validate_tool_call(event, analysis, proposed)


def test_validate_tool_call_rejects_invented_argument_not_in_analysis():
    event = _make_event()
    analysis = _analysis("schedule_recheck")  # no follow_up_minutes stated
    proposed = ProposedToolCall(
        name="schedule_patient_recheck", arguments={"minutes": 20}
    )
    with pytest.raises(ToolValidationError):
        validate_tool_call(event, analysis, proposed)


def test_validate_tool_call_rejects_argument_that_disagrees_with_analysis():
    event = _make_event()
    analysis = _analysis("schedule_recheck", follow_up_minutes=10)
    proposed = ProposedToolCall(
        name="schedule_patient_recheck", arguments={"minutes": 999}
    )
    with pytest.raises(ToolValidationError):
        validate_tool_call(event, analysis, proposed)


def test_validate_tool_call_rejects_unsupported_argument_field():
    event = _make_event()
    analysis = _analysis("need_help", requested_contact="Luis")
    proposed = ProposedToolCall(
        name="request_caregiver_help", arguments={"requested_contact": "Luis"}
    )
    with pytest.raises(ToolValidationError):
        validate_tool_call(event, analysis, proposed)


def test_validate_tool_call_rejects_minutes_out_of_range():
    event = _make_event()
    analysis = _analysis("schedule_recheck", follow_up_minutes=500)
    proposed = ProposedToolCall(
        name="schedule_patient_recheck", arguments={"minutes": 500}
    )
    with pytest.raises(ToolValidationError):
        validate_tool_call(event, analysis, proposed)


def test_validate_tool_call_rejects_treatment_language_in_arguments():
    event = _make_event()
    analysis = _analysis("schedule_recheck", follow_up_minutes=10)
    proposed = ProposedToolCall.model_construct(
        name="schedule_patient_recheck",
        arguments={"minutes": "give 10 units of insulin"},
    )
    with pytest.raises(ToolValidationError):
        validate_tool_call(event, analysis, proposed)


def test_execute_validated_tool_changes_status_and_is_idempotent():
    event = _make_event()
    analysis = _analysis("treating")
    validated = validate_tool_call(event, analysis, propose_tool(analysis))

    changed = execute_validated_tool(event, validated)
    assert changed is True
    assert event.status == EventStatus.MONITORING

    changed_again = execute_validated_tool(event, validated)
    assert changed_again is False


def test_execute_validated_tool_report_unclear_response_never_changes_status():
    event = _make_event()
    analysis = _analysis("unknown")
    validated = validate_tool_call(event, analysis, propose_tool(analysis))
    changed = execute_validated_tool(event, validated)
    assert changed is False
    assert event.status == EventStatus.CHECK_IN_REQUIRED
