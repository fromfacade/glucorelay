"""Regression tests for the deterministic post-Gemma semantic validation
layer (`app.gemma_service._apply_deterministic_corrections`).

These mock the low-level Gemma client (like `test_gemma_invalid_output_uses_fallback`
in test_voice_check_in.py) so `analyze_patient_checkin` runs its real,
unmocked pipeline - including the deterministic corrections - against a
fabricated model response. This lets us reproduce (and pin a regression
test for) the exact bug reports from the real Gemma smoke test without
needing network access.
"""

from app import gemma_service
from app.gemma_service import analyze_patient_checkin
from app.models import PatientCheckInAnalysis
from app.tools import propose_tool


class _FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = None


class _FakeModels:
    def __init__(self, parsed):
        self._parsed = parsed

    def generate_content(self, **kwargs):
        return _FakeResponse(self._parsed)


class _FakeClient:
    def __init__(self, parsed):
        self.models = _FakeModels(parsed)


def _mock_gemma_response(monkeypatch, parsed: PatientCheckInAnalysis) -> None:
    monkeypatch.setattr(
        gemma_service, "_get_client_and_model", lambda: (_FakeClient(parsed), "gemma-4-demo")
    )


def test_unsupported_okay_result_downgraded_to_unknown(monkeypatch):
    """Regression test for the real bug report: Gemma incorrectly inferred
    wellness from an unrelated statement about a backpack's color."""
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="okay",
            summary="The patient's backpack is red.",
            supply_location="red backpack",
            detected_language="en",
        ),
    )
    analysis, source, note = analyze_patient_checkin("My backpack is red.")

    assert analysis.action == "unknown"
    assert source == "gemma"  # a semantic correction is NOT a Gemma failure
    assert note is not None
    assert "unsupported_action_downgraded:okay" in note


def test_backpack_color_alone_never_produces_okay_via_fallback():
    """The deterministic fallback parser (no Gemma configured) must also
    classify this as unknown - it never matched an "okay" phrase to begin
    with, but this pins that behavior against regressions."""
    analysis, source, _note = analyze_patient_checkin("My backpack is red.")
    assert analysis.action == "unknown"
    assert source == "fallback"


def test_genuine_okay_statement_remains_okay(monkeypatch):
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="okay",
            summary="The patient said they are okay.",
            english_summary="The patient said they are okay.",
            responsive=True,
            detected_language="en",
        ),
    )
    analysis, source, note = analyze_patient_checkin("I'm okay.")

    assert analysis.action == "okay"
    assert source == "gemma"
    assert note is None


def test_other_actions_also_require_explicit_evidence(monkeypatch):
    # Gemma incorrectly claims "treating" with no actual treatment language.
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="treating",
            summary="The patient mentioned their backpack.",
            detected_language="en",
        ),
    )
    analysis, source, note = analyze_patient_checkin("My backpack is over there.")
    assert analysis.action == "unknown"
    assert source == "gemma"
    assert "unsupported_action_downgraded:treating" in note


def test_explicit_recheck_request_takes_precedence_over_okay(monkeypatch):
    """Regression test: Gemma classified this as 'okay' with
    follow_up_minutes=10 instead of 'schedule_recheck'."""
    transcript = "Everything is okay, but check on me again in ten minutes."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="okay",
            summary=transcript,
            follow_up_minutes=10,
            detected_language="en",
        ),
    )
    analysis, source, note = analyze_patient_checkin(transcript)

    assert analysis.action == "schedule_recheck"
    assert analysis.follow_up_minutes == 10
    assert source == "gemma"
    assert "recheck_request_took_precedence_over_okay" in note

    tool = propose_tool(analysis)
    assert tool.name == "schedule_patient_recheck"
    assert tool.arguments == {"minutes": 10}


def test_recheck_precedence_does_not_fire_without_a_minutes_value(monkeypatch):
    # A vague mention of "okay" with no follow-up minutes stays "okay" even
    # if the transcript happens to also contain recheck-ish language.
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="okay",
            summary="The patient said they are okay.",
            english_summary="The patient said they are okay.",
            detected_language="en",
        ),
    )
    analysis, _source, note = analyze_patient_checkin("I'm okay, check back on me later.")
    assert analysis.action == "okay"
    assert note is None


def test_english_summary_is_not_rewritten_for_english_transcripts(monkeypatch):
    """Regression test: Gemma produced a reworded/rewritten english_summary
    for an already-English transcript."""
    transcript = "I'm awake, but I'm alone and I can't stand up."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="need_help",
            summary=transcript,
            english_summary="The subject reports isolation and an inability to stand.",
            detected_language="en",
        ),
    )
    analysis, source, note = analyze_patient_checkin(transcript)

    assert analysis.english_summary == analysis.summary
    assert source == "gemma"
    assert "english_summary_normalized" in note


def test_non_english_transcript_keeps_its_translated_english_summary(monkeypatch):
    transcript = "Estoy despierta, pero me siento confundida. Por favor llama a Helper."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="need_help",
            summary="Estoy despierta, pero me siento confundida.",
            english_summary="I'm awake, but I feel confused.",
            requested_contact="Helper",
            reported_condition="confused",
            detected_language="es",
        ),
    )
    analysis, source, note = analyze_patient_checkin(transcript)

    assert source == "gemma"
    assert analysis.detected_language == "es"
    assert analysis.english_summary == "I'm awake, but I feel confused."
    assert analysis.requested_contact == "Helper"
    # No english_summary_normalized correction for non-English input.
    assert note is None


def test_spanish_help_request_for_helper_is_understood(monkeypatch):
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="need_help",
            summary="La paciente pide que llamen a Helper.",
            english_summary="The patient asks that Helper be called.",
            requested_contact="Helper",
            detected_language="es",
        ),
    )
    analysis, source, _note = analyze_patient_checkin(
        "Me siento confundida. Por favor llama a Helper."
    )
    assert source == "gemma"
    assert analysis.action == "need_help"
    assert analysis.requested_contact == "Helper"
