"""Regression tests for the hardened Gemma structured-response parsing
pipeline (`app.gemma_service._extract_parsed_detailed` and the retry loop
in `analyze_patient_checkin`).

These mock the low-level Gemma client (like `test_gemma_invalid_output_uses_fallback`
in test_voice_check_in.py) with raw `.text` payloads (never `.parsed`) so
the real fence-stripping / balanced-JSON-extraction / retry pipeline runs
end-to-end against a fabricated model response.
"""

import json

from app import gemma_service
from app.gemma_service import analyze_patient_checkin
from app.models import PatientCheckInAnalysis
from app.tools import propose_tool

_VALID_PAYLOAD = {
    "action": "need_help",
    "summary": "The patient is alone and cannot stand.",
    "english_summary": "The patient is alone and cannot stand.",
    "detected_language": "en",
    "responsive": True,
}


class _FakeCandidate:
    def __init__(self, finish_reason=None):
        self.finish_reason = finish_reason


class _FakeFeedback:
    def __init__(self, block_reason=None):
        self.block_reason = block_reason


class _FakeResponse:
    def __init__(self, text=None, parsed=None, finish_reason=None, block_reason=None):
        self.text = text
        self.parsed = parsed
        self.candidates = [_FakeCandidate(finish_reason)] if finish_reason is not None else []
        self.prompt_feedback = _FakeFeedback(block_reason) if block_reason is not None else None


class _ScriptedModels:
    """Returns each response in `responses` in order, one per call, so
    tests can simulate a first (bad) attempt followed by a retry."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def generate_content(self, **kwargs):
        self.call_count += 1
        index = min(self.call_count - 1, len(self._responses) - 1)
        return self._responses[index]


class _ScriptedClient:
    def __init__(self, responses):
        self.models = _ScriptedModels(responses)


def _mock_responses(monkeypatch, responses) -> _ScriptedClient:
    client = _ScriptedClient(responses)
    monkeypatch.setattr(gemma_service, "_get_client_and_model", lambda: (client, "gemma-4-demo"))
    return client


def _fake_analysis_response(**overrides) -> str:
    payload = dict(_VALID_PAYLOAD)
    payload.update(overrides)
    return json.dumps(payload)


# --- 1. response.parsed success --------------------------------------------


def test_prefers_response_parsed_when_present(monkeypatch):
    parsed = PatientCheckInAnalysis(
        action="okay", summary="I'm okay.", english_summary="I'm okay.", detected_language="en"
    )
    client = _mock_responses(monkeypatch, [_FakeResponse(parsed=parsed, text="ignored garbage {")])

    analysis, source, note = analyze_patient_checkin("I'm okay.")

    assert source == "gemma"
    assert analysis.action == "okay"
    assert note is None
    assert client.models.call_count == 1


# --- 2. fenced JSON ----------------------------------------------------------


def test_handles_markdown_fenced_json(monkeypatch):
    text = "```json\n" + _fake_analysis_response() + "\n```"
    client = _mock_responses(monkeypatch, [_FakeResponse(text=text, finish_reason="STOP")])

    analysis, source, note = analyze_patient_checkin(
        "I'm awake, but I'm alone and I can't stand up."
    )

    assert source == "gemma"
    assert analysis.action == "need_help"
    assert note is None
    assert client.models.call_count == 1


# --- 3. prose before JSON ----------------------------------------------------


def test_handles_prose_before_json(monkeypatch):
    text = "Sure, here is the structured result:\n" + _fake_analysis_response()
    client = _mock_responses(monkeypatch, [_FakeResponse(text=text, finish_reason="STOP")])

    analysis, source, _note = analyze_patient_checkin(
        "I'm awake, but I'm alone and I can't stand up."
    )

    assert source == "gemma"
    assert analysis.action == "need_help"
    assert client.models.call_count == 1


# --- 4. prose after JSON -----------------------------------------------------


def test_handles_prose_after_json(monkeypatch):
    text = _fake_analysis_response() + "\nLet me know if you need anything else!"
    client = _mock_responses(monkeypatch, [_FakeResponse(text=text, finish_reason="STOP")])

    analysis, source, _note = analyze_patient_checkin(
        "I'm awake, but I'm alone and I can't stand up."
    )

    assert source == "gemma"
    assert analysis.action == "need_help"
    assert client.models.call_count == 1


# --- 5. empty text ------------------------------------------------------------


def test_empty_text_without_explicit_safety_signal_is_empty_model_response(monkeypatch):
    client = _mock_responses(
        monkeypatch, [_FakeResponse(text="", finish_reason="STOP")]
    )

    analysis, source, note = analyze_patient_checkin("hello?")

    assert source == "fallback"
    assert note.startswith("empty_model_response:")
    assert "finish_reason=STOP" in note
    # Never fabricates a safety-block claim the SDK didn't actually report.
    assert "safety_blocked" not in note
    assert analysis.action == "unknown"  # from the deterministic fallback parser
    assert client.models.call_count == 1  # not retryable, no retry attempted


def test_explicit_safety_finish_reason_is_reported_as_safety_blocked(monkeypatch):
    client = _mock_responses(
        monkeypatch, [_FakeResponse(text=None, finish_reason="SAFETY")]
    )

    _analysis, source, note = analyze_patient_checkin("hello?")

    assert source == "fallback"
    assert note.startswith("safety_blocked:")
    assert "finish_reason=SAFETY" in note


# --- 6. malformed JSON (both attempts fail, ends in fallback) ---------------


def test_malformed_json_on_both_attempts_falls_back(monkeypatch):
    client = _mock_responses(
        monkeypatch,
        [
            _FakeResponse(text="{not valid json", finish_reason="STOP"),
            _FakeResponse(text="{still not valid", finish_reason="STOP"),
        ],
    )

    analysis, source, note = analyze_patient_checkin("false alarm, cancel it")

    assert client.models.call_count == 2  # exactly one retry, never more
    assert source == "fallback"
    assert note.startswith("json_parse_failed:")
    assert "retry_attempted=true" in note
    # The deterministic fallback still classifies this transcript correctly.
    assert analysis.action == "false_alarm"


# --- 7. successful retry ------------------------------------------------------


def test_retry_succeeds_after_initial_malformed_json(monkeypatch):
    good_text = _fake_analysis_response()
    client = _mock_responses(
        monkeypatch,
        [
            _FakeResponse(text="not json at all", finish_reason="STOP"),
            _FakeResponse(text=good_text, finish_reason="STOP"),
        ],
    )

    analysis, source, note = analyze_patient_checkin(
        "I'm awake, but I'm alone and I can't stand up."
    )

    assert client.models.call_count == 2
    assert source == "gemma"
    assert analysis.action == "need_help"
    assert "retry_attempted=true" in note

    # Never proposes/executes more than one tool call even though the
    # underlying Gemma request was retried.
    tool = propose_tool(analysis)
    assert tool.name == "request_caregiver_help"


def test_retry_succeeds_after_initial_schema_validation_failure(monkeypatch):
    bad_payload = json.dumps({"action": "not-a-real-action", "summary": "x"})
    good_text = _fake_analysis_response(action="okay", summary="I'm okay.", english_summary="I'm okay.")
    client = _mock_responses(
        monkeypatch,
        [
            _FakeResponse(text=bad_payload, finish_reason="STOP"),
            _FakeResponse(text=good_text, finish_reason="STOP"),
        ],
    )

    analysis, source, note = analyze_patient_checkin("I'm okay.")

    assert client.models.call_count == 2
    assert source == "gemma"
    assert analysis.action == "okay"
    assert "retry_attempted=true" in note


# --- 8. failed retry followed by fallback (schema validation variant) ------


def test_schema_validation_failure_on_both_attempts_falls_back(monkeypatch):
    bad_payload = json.dumps({"action": "not-a-real-action", "summary": "x"})
    client = _mock_responses(
        monkeypatch,
        [
            _FakeResponse(text=bad_payload, finish_reason="STOP"),
            _FakeResponse(text=bad_payload, finish_reason="STOP"),
        ],
    )

    analysis, source, note = analyze_patient_checkin("My backpack is red.")

    assert client.models.call_count == 2
    assert source == "fallback"
    assert note.startswith("schema_validation_failed:")
    assert "retry_attempted=true" in note
    assert analysis.action == "unknown"


# --- 9. the exact "can't stand up" transcript, need_help without fallback --


def test_cant_stand_up_transcript_produces_need_help_without_fallback(monkeypatch):
    """Regression test for the live smoke-test failure: this transcript
    must produce action=need_help via Gemma (source == "gemma"), not the
    fallback parser, whenever Gemma actually returns a valid structured
    response - regardless of incidental prose/fences around it."""
    transcript = "I'm awake, but I'm alone and I can't stand up."
    text = "Here you go:\n```json\n" + _fake_analysis_response() + "\n```\nAll set."
    client = _mock_responses(monkeypatch, [_FakeResponse(text=text, finish_reason="STOP")])

    analysis, source, _note = analyze_patient_checkin(transcript)

    assert source == "gemma"
    assert analysis.action == "need_help"
    assert client.models.call_count == 1  # no retry needed - first attempt was usable

    tool = propose_tool(analysis)
    assert tool.name == "request_caregiver_help"


def test_safety_denylist_never_blocks_this_transcripts_content(monkeypatch):
    """Companion check: even once parsed, the safety re-scan must not
    reject the resulting fields for this transcript (see
    test_semantic_validation.py for the direct `_contains_unsafe_content`
    unit test)."""
    transcript = "I'm awake, but I'm alone and I can't stand up."
    client = _mock_responses(
        monkeypatch, [_FakeResponse(text=_fake_analysis_response(), finish_reason="STOP")]
    )

    _analysis, source, _note = analyze_patient_checkin(transcript)
    assert source == "gemma"
    assert client.models.call_count == 1
