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
from app.gemma_service import _contains_unsafe_content, analyze_patient_checkin
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


# --- Regression tests: "can't stand up" safety-denylist false positive,
# Spanish explicit-intent reclassification, and english_summary
# normalization (live Gemma smoke test failures) -------------------------


def test_safety_denylist_does_not_reject_reported_physical_limitations():
    """The treatment-language safety re-scan must never fire on a patient
    reporting they can't stand/move - those are exactly what `need_help`
    exists to capture, not a generated medical instruction."""
    for phrase in (
        "I'm awake, but I'm alone and I can't stand up.",
        "I am unable to stand right now.",
        "I can't move at all.",
        "unable to stand",
        "can't stand up",
    ):
        assert _contains_unsafe_content(phrase) is None


def test_english_cant_stand_up_produces_need_help_via_gemma_without_fallback(monkeypatch):
    """Regression test for the live smoke-test failure where this exact
    transcript fell back to the deterministic parser instead of using
    Gemma's (correct) need_help classification."""
    transcript = "I'm awake, but I'm alone and I can't stand up."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="need_help",
            summary=transcript,
            english_summary=transcript,
            reported_condition="alone and unable to stand",
            responsive=True,
            detected_language="en",
        ),
    )
    analysis, source, _note = analyze_patient_checkin(transcript)

    assert source == "gemma"
    assert analysis.action == "need_help"

    tool = propose_tool(analysis)
    assert tool.name == "request_caregiver_help"


def test_unsupported_spanish_okay_is_reclassified_to_need_help(monkeypatch):
    """Regression test for the live smoke-test failure: Gemma incorrectly
    classified an explicit Spanish help request as 'okay'. The semantic
    validation layer must recognize the explicit contact-request phrase
    ("llama a") and the reported-condition phrase ("me siento confundida")
    and reclassify to need_help - not merely downgrade to 'unknown'."""
    transcript = "Me siento confundida. Por favor llama a Helper."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="okay",
            summary=transcript,
            requested_contact="Helper",
            reported_condition="confundida",
            detected_language="es",
        ),
    )
    analysis, source, note = analyze_patient_checkin(transcript)

    assert analysis.action == "need_help"
    assert source == "gemma"  # a semantic correction is NOT a Gemma failure
    assert "reclassified_to_need_help_from:okay" in note
    assert analysis.detected_language == "es"
    assert analysis.requested_contact == "Helper"
    assert analysis.reported_condition == "confundida"
    assert analysis.english_summary == "I feel confused. Please call Helper."

    tool = propose_tool(analysis)
    assert tool.name == "request_caregiver_help"


def test_spanish_contact_mention_without_a_request_remains_unknown(monkeypatch):
    """A contact name appearing in Spanish text alone - with no help or
    contact-request verb, and no reported inability - must NOT trigger
    need_help."""
    transcript = "Mi amigo Helper vino a visitarme ayer."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="okay",
            summary=transcript,
            requested_contact="Helper",
            detected_language="es",
        ),
    )
    analysis, source, note = analyze_patient_checkin(transcript)

    assert analysis.action == "unknown"
    assert source == "gemma"
    assert "unsupported_action_downgraded:okay" in note


def test_english_summary_not_copied_when_language_is_genuinely_undetermined(monkeypatch):
    """Regression test: when neither Spanish nor English evidence phrases
    are present, detected_language must resolve to "und" (never guessed),
    and a missing summary must never be copied into english_summary (that
    would falsely label unidentified text as an English translation)."""
    transcript = "Static noise on the line, mostly silence."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="unknown",
            summary=transcript,
            detected_language=None,
        ),
    )
    analysis, _source, _note = analyze_patient_checkin(transcript)

    assert analysis.detected_language == "und"
    assert analysis.english_summary is None


def test_spanish_marker_present_but_no_known_translation_leaves_summary_none(monkeypatch):
    """When language normalization detects Spanish but there's no known
    deterministic translation for this exact transcript, english_summary
    must stay None rather than fabricate one - only detected_language is
    filled in."""
    transcript = "Estoy bien, gracias."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="okay",
            summary=transcript,
            detected_language=None,
        ),
    )
    analysis, source, note = analyze_patient_checkin(transcript)

    assert analysis.detected_language == "es"
    assert analysis.english_summary is None
    assert source == "gemma"
    assert "detected_language_normalized:es" in note


# --- Regression tests: deterministic language normalization
# (`normalize_detected_language`) - live smoke-test failure where
# "Me siento confundida. Por favor llama a Helper." correctly produced
# need_help/Helper/english_summary but left detected_language as None. ----


def test_spanish_help_request_gets_es_when_model_omits_language(monkeypatch):
    """The exact live-smoke-test regression: Gemma got everything right
    except detected_language, which it left as None."""
    transcript = "Me siento confundida. Por favor llama a Helper."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="need_help",
            summary=transcript,
            english_summary="I feel confused. Please call Helper.",
            requested_contact="Helper",
            reported_condition="confundida",
            detected_language=None,
        ),
    )
    analysis, source, note = analyze_patient_checkin(transcript)

    assert analysis.detected_language == "es"
    assert analysis.action == "need_help"
    assert analysis.requested_contact == "Helper"
    assert analysis.english_summary == "I feel confused. Please call Helper."
    assert source == "gemma"
    assert "detected_language_normalized:es" in note

    tool = propose_tool(analysis)
    assert tool.name == "request_caregiver_help"


def test_spanish_non_emergency_contact_mention_gets_es_and_stays_unknown(monkeypatch):
    """A Spanish transcript can contain a scoped language marker (so we
    correctly identify it as Spanish) without containing any actual
    request/emergency evidence - the action must still resolve to
    "unknown", never need_help, from a contact mention alone."""
    transcript = "Estoy bien, mi amigo Helper vino a visitarme ayer."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="false_alarm",  # Gemma's guess has no false_alarm evidence either.
            summary=transcript,
            requested_contact="Helper",
            detected_language=None,
        ),
    )
    analysis, source, note = analyze_patient_checkin(transcript)

    assert analysis.detected_language == "es"
    assert analysis.action == "unknown"
    assert source == "gemma"
    assert "detected_language_normalized:es" in note
    assert "unsupported_action_downgraded:false_alarm" in note


def test_english_transcript_gets_en_when_model_omits_language(monkeypatch):
    transcript = "I'm treating it with juice right now."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="treating",
            summary=transcript,
            detected_language=None,
        ),
    )
    analysis, source, note = analyze_patient_checkin(transcript)

    assert analysis.detected_language == "en"
    assert analysis.action == "treating"
    assert analysis.english_summary == analysis.summary
    assert source == "gemma"
    assert "detected_language_normalized:en" in note


def test_model_provided_language_is_always_preserved(monkeypatch):
    """normalize_detected_language must never override a language Gemma
    actually reported, even an unusual/unexpected value."""
    transcript = "I'm okay."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="okay",
            summary=transcript,
            english_summary=transcript,
            detected_language="en-US",
        ),
    )
    analysis, _source, note = analyze_patient_checkin(transcript)

    assert analysis.detected_language == "en-US"
    assert note is None


def test_spanish_summary_mislabeled_as_english_is_corrected(monkeypatch):
    """Regression test: Gemma echoing the Spanish summary verbatim into
    english_summary (i.e. not actually translating) must be corrected, not
    trusted at face value."""
    transcript = "Me siento confundida. Por favor llama a Helper."
    _mock_gemma_response(
        monkeypatch,
        PatientCheckInAnalysis(
            action="need_help",
            summary=transcript,
            english_summary=transcript,  # Gemma failed to translate.
            requested_contact="Helper",
            detected_language="es",
        ),
    )
    analysis, _source, note = analyze_patient_checkin(transcript)

    assert analysis.english_summary != transcript
    assert analysis.english_summary == "I feel confused. Please call Helper."
    assert "english_summary_mislabeled_corrected" in note
