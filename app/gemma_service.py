"""Gemma's three responsibilities in GlucoRelay's emergency workflow.

1. Understand the patient's natural-language check-in
   (`analyze_patient_checkin` -> `PatientCheckInAnalysis`).
2. Propose one constrained application tool - see `app.tools.propose_tool`,
   which derives the tool deterministically from the `action` field this
   module produces (Python, not a second model call).
3. Generate a grounded caregiver handoff from verified facts
   (`generate_caregiver_handoff` -> `CaregiverHandoff`).

SAFETY BOUNDARY: Gemma must never decide whether a glucose reading is
dangerous, recommend insulin/medication/dosages, diagnose the patient,
change thresholds, contact emergency services, invent facts (contacts,
locations, symptoms) not stated in the transcript, or claim the patient is
unconscious. All of that stays deterministic Python in `app.engine`,
`app.transitions`, and `app.tools`. Every structured response from Gemma
is validated with Pydantic *and* re-scanned for banned language
(`_contains_unsafe_content`) before it is trusted - if either check fails,
the deterministic fallback parser is used instead and the event's state
never changes based on unusable or unsafe output.

ON "FUNCTION CALLING": this module intentionally does NOT use the Gemini
API's `tools`/`function_declarations` parameter. Google's documentation
for Gemma-family models describes function calling only for a
locally-hosted model driven through `transformers.apply_chat_template`,
with the developer manually parsing the model's text output - not the
Gemini API's native tool-calling mechanism, which is documented for
Gemini reasoning models. Since GEMMA_MODEL is called here through the
Gemini API (`google-genai`), this module uses `response_schema`-constrained
structured JSON output for both interpretation and (indirectly, via
`app.tools.propose_tool`) tool selection, and never claims this is native
function calling.
"""

import json
import logging
import os
import re
from typing import Any, Literal

from pydantic import ValidationError

from app.models import CaregiverHandoff, EmergencyEvent, PatientCheckInAnalysis

logger = logging.getLogger("glucorelay.gemma")

InterpretationSource = Literal["gemma", "fallback"]
HandoffSource = Literal["gemma", "fallback"]

MAX_TRANSCRIPT_LENGTH = 2000

# --- Safety re-scan -------------------------------------------------------
#
# Applied to every string field Gemma returns, on top of Pydantic schema
# validation and the prompt's own instructions. If Gemma's output slips
# past the schema but violates the safety boundary in substance, we must
# not use it.
#
# SCOPE: this list exists to catch generated MEDICAL INSTRUCTIONS/ADVICE
# (a treatment command, a dosage, a diagnosis) - never a patient-reported
# physical condition or limitation. Phrases like "can't stand", "unable to
# stand", "can't move", "alone", or "confused" must NEVER be added here:
# those are exactly the kind of explicit self-reported symptoms
# `need_help` is supposed to capture (see `_NEED_HELP_EVIDENCE` below), and
# rejecting them would silently break the app's core safety feature -
# escalating a patient who cannot safely respond. Each entry is matched on
# a word boundary (see `_matches_denylist`) so it can never fire on part of
# an unrelated word.
_UNSAFE_PHRASES = (
    "unconscious",
    "unresponsive and not breathing",
    "insulin",
    "dosage",
    "dose",
    "milligram",
    "medication",
    "diagnos",  # stem: diagnose/diagnosis/diagnosed/diagnostic
    "inject",
    "overdose",
)


def _denylist_pattern(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """Builds a word-boundary regex from a tuple of phrases/stems.

    `\\b<phrase>\\w*\\b` isolates each phrase to a whole word (or the start
    of one, e.g. "diagnos" -> "diagnosis") so it can never match as a
    fragment of an unrelated word (e.g. "dose" must never match inside
    "overdose" - that stays its own separate, deliberate entry).
    """
    alternatives = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"\b(?:{alternatives})\w*\b")


_UNSAFE_PHRASES_PATTERN = _denylist_pattern(_UNSAFE_PHRASES)


def _contains_unsafe_content(*texts: str | None) -> str | None:
    for text in texts:
        if not text:
            continue
        match = _UNSAFE_PHRASES_PATTERN.search(text.lower())
        if match:
            return match.group(0)
    return None


# --- Deterministic post-model semantic validation --------------------------
#
# WHY THIS EXISTS: Gemma is asked to classify a transcript into one of six
# actions, but a real model can still over-infer - e.g. reading "My backpack
# is red." and reporting action="okay" simply because nothing sounded
# alarming. Gemma's own classification is never trusted at face value for
# state-changing purposes; each action must be backed by an explicit phrase
# actually present in the *original* transcript, or it is deterministically
# downgraded to "unknown" here. This runs on fallback-parser output too
# (defense in depth) even though the fallback already only assigns an
# action when one of these same phrases matched.
# Split by language (see `_NEED_HELP_EVIDENCE_EN`/`_ES` below for why).
_OKAY_EVIDENCE_EN = (
    "i'm okay", "im okay", "i am okay",
    "i'm ok", "im ok", "i am ok",
    "i'm fine", "im fine", "i am fine",
    "i feel okay", "i feel fine",
    "everything is okay", "everything's okay", "everything is fine",
    "i'm alright", "im alright", "i am alright",
    "feeling fine", "feeling okay",
)
_OKAY_EVIDENCE_ES = (
    "estoy bien", "me siento bien", "todo está bien", "todo esta bien",
)
_OKAY_EVIDENCE = _OKAY_EVIDENCE_EN + _OKAY_EVIDENCE_ES

_TREATING_EVIDENCE = (
    "treating it", "i'm treating", "im treating", "i am treating",
    "drank juice", "drank some juice", "i drank juice", "drinking juice",
    "glucose tab", "ate something", "eating something", "took glucose",
    "took some glucose", "having juice", "had juice", "ate a snack",
    "eating a snack", "had some candy", "eating candy", "handling it",
)

_FALSE_ALARM_EVIDENCE = (
    "false alarm", "cancel it", "cancel this", "cancel the alert",
    "never mind", "it was a mistake", "disregard that", "disregard this",
    "ignore that", "ignore this",
)

# Split by language so both the action-evidence check above AND the
# deterministic language-normalization helper below can reuse the exact
# same, carefully scoped phrase sets - a single source of truth rather than
# a second, looser "does this look Spanish" heuristic.
_NEED_HELP_EVIDENCE_EN = (
    "need help", "i need help", "help me", "send help", "come get me",
    "call someone", "call ", "contact ", "please call", "please contact",
    "can't stand", "cant stand", "can't move", "cant move", "can't get up",
    "cant get up", "unable to stand", "unable to move", "i'm alone",
    "im alone", "i am alone", "too weak", "too dizzy", "feel faint",
    "about to pass out", "confus",
)

# Spanish: explicit help/contact-request phrases. Deliberately scoped to
# request VERBS ("llama a", "contacta a", "ayudame", "necesito ayuda"),
# never a bare name - a contact name alone (e.g. just "Helper") must never
# by itself count as evidence of a help request (or of the transcript being
# Spanish - see `normalize_detected_language`).
_NEED_HELP_EVIDENCE_ES = (
    "necesito ayuda", "ayudame", "ayúdame", "por favor llama a", "llama a",
    "llama ", "por favor contacta a", "contacta a", "contacta ",
    # Explicit reported inability to safely respond.
    "me siento confundida", "me siento confundido", "confundid",
    "no puedo pararme", "no puedo levantarme", "no puedo moverme",
    "estoy débil", "estoy debil", "estoy sola", "estoy solo",
)

_NEED_HELP_EVIDENCE = _NEED_HELP_EVIDENCE_EN + _NEED_HELP_EVIDENCE_ES

_SCHEDULE_RECHECK_EVIDENCE = (
    "check on me", "check again", "recheck", "check back", "call back in",
    "check in on me",
)

_ACTION_EVIDENCE: dict[str, tuple[str, ...]] = {
    "okay": _OKAY_EVIDENCE,
    "treating": _TREATING_EVIDENCE,
    "false_alarm": _FALSE_ALARM_EVIDENCE,
    "need_help": _NEED_HELP_EVIDENCE,
    "schedule_recheck": _SCHEDULE_RECHECK_EVIDENCE,
}

# The Spanish-only subset of every action's evidence phrases, reused
# verbatim (not duplicated) by `normalize_detected_language` below. The
# English-only subset likewise establishes "English explicitly present".
# Neither list includes a bare name - see `_NEED_HELP_EVIDENCE_ES` comment.
_SPANISH_INTENT_MARKERS = _OKAY_EVIDENCE_ES + _NEED_HELP_EVIDENCE_ES
_ENGLISH_INTENT_MARKERS = (
    _OKAY_EVIDENCE_EN
    + _TREATING_EVIDENCE
    + _FALSE_ALARM_EVIDENCE
    + _NEED_HELP_EVIDENCE_EN
    + _SCHEDULE_RECHECK_EVIDENCE
)


def _has_any_phrase(lowered_text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in lowered_text for phrase in phrases)


def _has_evidence(lowered_transcript: str, action: str) -> bool:
    """Does the transcript contain explicit language supporting `action`?

    "unknown" never requires evidence (it's the safe default).
    """
    phrases = _ACTION_EVIDENCE.get(action)
    if not phrases:
        return True
    return _has_any_phrase(lowered_transcript, phrases)


_ENGLISH_TAGS = {"en", "english"}


def _is_english(detected_language: str | None) -> bool:
    """True only when the language was explicitly confirmed as English.

    IMPORTANT: an unset/unknown `detected_language` (None) must NOT default
    to English here - doing so previously caused non-English transcripts
    where Gemma omitted `detected_language` to have their own (non-English)
    `summary` copied straight into `english_summary`, mislabeling foreign
    text as an English translation. When we cannot confirm the language,
    the safe behavior is to leave `english_summary` alone rather than
    guess.
    """
    if not detected_language:
        return False
    normalized = detected_language.strip().lower()
    return (
        normalized in _ENGLISH_TAGS
        or normalized.startswith("en-")
        or normalized.startswith("en_")
    )


# A tiny, explicitly-labeled deterministic translation table for the exact
# demo phrases this project's fallback parser and smoke test use. This is
# NOT a general translator - it only fires when Gemma (a) confirmed the
# transcript is non-English and (b) failed to supply its own English
# translation, so the caregiver view is never left silently blank when a
# safe, known-correct translation is available. Anything not in this table
# simply leaves `english_summary` as None rather than fabricate a
# translation.
_KNOWN_SPANISH_SUMMARIES: dict[str, str] = {
    "me siento confundida. por favor llama a helper.": "I feel confused. Please call Helper.",
    "me siento confundido. por favor llama a helper.": "I feel confused. Please call Helper.",
}


def _deterministic_translation(transcript: str) -> str | None:
    return _KNOWN_SPANISH_SUMMARIES.get(transcript.strip().lower())


UNDETERMINED_LANGUAGE = "und"


def normalize_detected_language(transcript: str, detected_language: str | None) -> str:
    """Resolves a supported language tag deterministically.

    Rules:
    - A language value Gemma actually returned is trusted and preserved
      as-is (Gemma is explicitly instructed to report the transcript's
      real language; this function only fills the gap when it's missing,
      never overrides a stated value).
    - When missing, explicit Spanish evidence - the SAME carefully scoped
      phrases already used for intent validation (`_SPANISH_INTENT_MARKERS`,
      i.e. the Spanish subset of `_OKAY_EVIDENCE`/`_NEED_HELP_EVIDENCE`) -
      sets "es". A bare contact name (e.g. "Helper") is never in that list,
      so a name mention alone can never imply Spanish (or an emergency).
    - When missing and explicit English evidence is present
      (`_ENGLISH_INTENT_MARKERS`), sets "en".
    - Otherwise "und" (undetermined) - this function never fabricates a
      language guess from weak/ambiguous signals.
    """
    if detected_language:
        return detected_language
    lowered = transcript.lower()
    if _has_any_phrase(lowered, _SPANISH_INTENT_MARKERS):
        return "es"
    if _has_any_phrase(lowered, _ENGLISH_INTENT_MARKERS):
        return "en"
    return UNDETERMINED_LANGUAGE


def describe_semantic_correction(reason: str) -> str:
    """Human-readable, presentation-safe explanation of a correction tag."""
    if reason.startswith("unsupported_action_downgraded:"):
        original = reason.split(":", 1)[1]
        return (
            f"The proposed action '{original}' was downgraded to 'unknown' "
            "because the transcript did not contain explicit supporting "
            "language for it. This is a deterministic backend safety "
            "correction, not a Gemma failure."
        )
    if reason == "recheck_request_took_precedence_over_okay":
        return (
            "An explicit follow-up/recheck request was detected, so it took "
            "precedence over a general 'okay' statement."
        )
    if reason == "english_summary_normalized":
        return (
            "english_summary was reset to match the validated summary "
            "instead of a reworded translation, since the transcript was "
            "already in English."
        )
    if reason.startswith("reclassified_to_need_help_from:"):
        original = reason.split(":", 1)[1]
        return (
            f"The proposed action '{original}' had no supporting language, "
            "but the transcript contained an explicit help/contact request "
            "or a stated inability to respond safely, so the action was "
            "corrected to 'need_help'. This is a deterministic backend "
            "safety correction, not a Gemma failure."
        )
    if reason == "english_summary_mislabeled_corrected":
        return (
            "english_summary matched the non-English summary verbatim "
            "(not an actual translation), so it was replaced with a known "
            "safe translation or cleared rather than mislabeling foreign "
            "text as English."
        )
    if reason == "english_summary_filled_from_known_translation":
        return (
            "Gemma did not supply an English translation for a non-English "
            "transcript, so a known, explicitly labeled deterministic "
            "translation was used instead of leaving it blank."
        )
    if reason.startswith("detected_language_normalized:"):
        language = reason.split(":", 1)[1]
        return (
            f"Gemma did not report a language, so it was deterministically "
            f"set to '{language}' based on explicit language markers "
            "actually present in the transcript (never guessed from a bare "
            "name or weak signal)."
        )
    return reason


def _apply_deterministic_corrections(
    transcript: str, analysis: PatientCheckInAnalysis
) -> tuple[PatientCheckInAnalysis, list[str]]:
    """Deterministic Python corrections applied after every interpretation
    (Gemma or fallback), never trusting the model's classification or
    wording at face value. Returns (possibly-corrected analysis, reasons).
    """
    lowered = transcript.lower()
    corrected = analysis
    reasons: list[str] = []

    # An explicit recheck request takes precedence over a general "okay"
    # statement (e.g. "Everything is okay, but check on me again in ten
    # minutes." must be classified as schedule_recheck, not okay).
    if (
        corrected.action == "okay"
        and corrected.follow_up_minutes is not None
        and _has_any_phrase(lowered, _SCHEDULE_RECHECK_EVIDENCE)
    ):
        corrected = corrected.model_copy(update={"action": "schedule_recheck"})
        reasons.append("recheck_request_took_precedence_over_okay")

    # An action may only stand if the transcript actually contains explicit
    # supporting language - guards against the model inferring an action
    # from an unrelated statement. Before giving up entirely, check whether
    # the transcript contains explicit evidence of "need_help" instead: a
    # genuine help/contact request or a stated inability to respond safely
    # must never be silently lost to "unknown" just because Gemma proposed
    # a different, unsupported action (e.g. mistakenly said "okay" for a
    # Spanish help request) - see `_NEED_HELP_EVIDENCE`. This never invents
    # facts; it only re-reads the same transcript against a stricter,
    # safety-priority phrase list.
    if not _has_evidence(lowered, corrected.action):
        original_action = corrected.action
        if original_action != "need_help" and _has_evidence(lowered, "need_help"):
            corrected = corrected.model_copy(update={"action": "need_help"})
            reasons.append(f"reclassified_to_need_help_from:{original_action}")
        else:
            corrected = corrected.model_copy(update={"action": "unknown"})
            reasons.append(f"unsupported_action_downgraded:{original_action}")

    # Deterministic language normalization - MUST run before english_summary
    # normalization/correction below, since that logic branches on whether
    # the language is confirmed English, confirmed non-English, or
    # genuinely undetermined ("und").
    if not corrected.detected_language:
        normalized_language = normalize_detected_language(transcript, corrected.detected_language)
        if normalized_language != corrected.detected_language:
            corrected = corrected.model_copy(update={"detected_language": normalized_language})
            reasons.append(f"detected_language_normalized:{normalized_language}")

    # english_summary normalization/correction. Only ever touched when the
    # language is a *confirmed, non-undetermined* value - never when it's
    # "und" (see `normalize_detected_language`'s docstring for why guessing
    # there is unsafe).
    detected_language = corrected.detected_language
    if _is_english(detected_language):
        # Never let a translation/rewrite attempt alter an already-English
        # summary - english_summary must be exactly the validated summary.
        if corrected.english_summary != corrected.summary:
            corrected = corrected.model_copy(update={"english_summary": corrected.summary})
            reasons.append("english_summary_normalized")
    elif detected_language and detected_language != UNDETERMINED_LANGUAGE:
        # Confirmed non-English. A non-English summary must never
        # masquerade as its own English translation.
        if (
            corrected.english_summary
            and corrected.english_summary.strip().lower() == corrected.summary.strip().lower()
        ):
            corrected = corrected.model_copy(
                update={"english_summary": _deterministic_translation(transcript)}
            )
            reasons.append("english_summary_mislabeled_corrected")
        elif not corrected.english_summary:
            translation = _deterministic_translation(transcript)
            if translation:
                corrected = corrected.model_copy(update={"english_summary": translation})
                reasons.append("english_summary_filled_from_known_translation")
    # else: detected_language is "und" - leave english_summary exactly as
    # Gemma/the fallback parser produced it; we cannot safely confirm or
    # correct a translation without knowing the source language.

    return corrected, reasons


ANALYSIS_SYSTEM_INSTRUCTION = """You are a narrow interpretation tool inside a Type 1 \
diabetes emergency-coordination prototype. A patient was asked to check in \
after a concerning glucose reading, and this is a transcript of their \
spoken response - it may be in English or another language.

Your ONLY job is to extract explicitly stated facts and classify the \
transcript into one of these actions:
- "okay": the patient says they feel fine / are okay right now.
- "treating": the patient says they are actively treating it (e.g. drank \
juice, ate something, took glucose tablets).
- "need_help": the patient reports confusion, inability to stand or move, \
severe weakness, faintness, being alone while asking for assistance, or \
explicitly asks for another person to be contacted or for help.
- "schedule_recheck": the patient asks to be checked on again later \
without asking for help right now (e.g. "check on me again in 10 minutes").
- "false_alarm": the patient says this was a mistake, a false alarm, or \
asks to cancel the alert.
- "unknown": the transcript is unclear, unrelated, contradictory, silent, \
or does not clearly match any of the above.

STRICT RULES:
- Use ONLY information explicitly stated in the transcript. Never infer or \
invent facts, names, locations, or symptoms that were not said.
- Never give medical advice, never recommend insulin, medication, or a \
dosage, never diagnose the patient, and never claim the patient is \
unconscious.
- Do NOT assume someone is safe merely because they say "I'm okay" - \
classify it as "okay" and let the application decide the resulting status \
(it will move to monitoring, not to resolved).
- Confusion, inability to stand/move, severe weakness, faintness, or being \
alone while requesting assistance should produce "need_help".
- An explicit request for help or for a person to be contacted always \
produces "need_help".
- Merely MENTIONING a contact's name, without asking for help or for them \
to be contacted, must NOT by itself produce "need_help".
- Merely MENTIONING a supply location (e.g. a backpack), without asking \
for help, must NOT by itself produce "need_help".
- Contradictory or ambiguous statements (e.g. asking for help then taking \
it back) should produce "unknown" unless the final, clear statement is \
unambiguous.
- Do not invent a contact name that was not said. Leave requested_contact \
null if none was explicitly named.
- Do not invent a supply location that was not said. Leave supply_location \
null if none was explicitly named.
- "summary" must be a short, factual, one-sentence restatement of what the \
patient said, in the same language as the transcript - not a medical \
judgment, and not translated.
- "english_summary" must be the same factual restatement, translated to \
English. If the transcript was already in English, "english_summary" must \
be IDENTICAL to "summary" - do not reword, rephrase, or "improve" it.
- "detected_language" must be a short language name or ISO code for the \
transcript's original language (e.g. "en", "es").
- "responsive" should be true only if the transcript itself demonstrates \
the patient is responsive (i.e. they said something coherent); leave it \
null if you cannot tell.
- Leave requested_contact, reported_condition, reported_action, \
supply_location, and follow_up_minutes as null unless the patient \
explicitly stated them.
- Respond with structured JSON only, matching the provided schema.

Examples:
Transcript: "I'm okay, check on me again soon."
-> action="okay".

Transcript: "Check on me again in ten minutes."
-> action="schedule_recheck", follow_up_minutes=10.

Transcript: "I drank some juice and I'm treating it."
-> action="treating", reported_action="drank juice", supply_location null.

Transcript: "I feel confused. Please call Luis. My glucagon is in my red backpack."
-> action="need_help", reported_condition="confused", requested_contact="Luis", \
supply_location="red backpack", responsive=true.

Transcript: "This was a false alarm, cancel it."
-> action="false_alarm".

Transcript: "Call Luis - actually, never mind, I think I'm okay."
-> action="okay" (the final, clear statement retracts the help request).

Transcript: "What's the weather like today?"
-> action="unknown".
"""


def _build_analysis_contents(transcript: str, *, strict: bool = False) -> str:
    contents = f'{ANALYSIS_SYSTEM_INSTRUCTION}\n\nPatient transcript:\n"""{transcript}"""'
    if strict:
        contents += _STRICT_JSON_REMINDER
    return contents


def _gemma_enabled() -> bool:
    return os.getenv("ENABLE_GEMMA", "true").strip().lower() == "true"


def _get_client_and_model() -> tuple[object | None, str | None]:
    if not _gemma_enabled():
        return None, None
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMMA_MODEL", "").strip()
    if not api_key or not model_name:
        return None, None
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai package is not installed; using fallback parser")
        return None, None
    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        logger.exception("Failed to initialize Gemma client")
        return None, None
    return client, model_name


# --- Deterministic fallback parser -----------------------------------------
#
# Used only when Gemma is disabled/unavailable/misconfigured, or returns
# output that fails schema or safety validation. English-only by design;
# clearly reports when it cannot interpret likely non-English input rather
# than guessing.

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}

_SPANISH_MARKERS = (
    "estoy", "está", "esta despierta", "esta despierto", "siento",
    "confundida", "confundido", "por favor", "llama", "ayuda", "necesito",
    "¿", "¡", "años", "señor", "señora",
)


def _looks_non_english(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SPANISH_MARKERS)


def _extract_minutes(text: str) -> int | None:
    match = re.search(r"(\d+)\s*minute", text)
    if match:
        return int(match.group(1))
    for word, value in _NUMBER_WORDS.items():
        if f"{word} minute" in text:
            return value
    return None


def _fallback_interpret(transcript: str) -> PatientCheckInAnalysis:
    text = transcript.lower()

    def has_any(*phrases: str) -> bool:
        return any(phrase in text for phrase in phrases)

    if _looks_non_english(text) and not has_any(
        "i need help", "help me", "i'm okay", "im okay", "false alarm"
    ):
        return PatientCheckInAnalysis(
            action="unknown",
            summary=(
                "The deterministic fallback parser only supports explicit "
                "English phrases and could not confidently interpret this "
                "transcript. Configure Gemma for multilingual understanding."
            ),
            detected_language="unknown-non-english",
        )

    if has_any(
        "need help", "help me", "call someone", "send help", "come get me",
        "can't stand", "cant stand", "can't move", "cant move", "i'm alone",
        "im alone",
    ):
        return PatientCheckInAnalysis(
            action="need_help",
            summary="Patient explicitly asked for help.",
            detected_language="en",
        )

    for lead_in in ("contact ", "call "):
        if lead_in in text:
            start = text.index(lead_in) + len(lead_in)
            raw_name = transcript[start : start + 40]
            candidate = raw_name.split(".")[0].split(",")[0].strip()
            candidate = candidate.split(" and ")[0].strip()
            name = candidate.title() if candidate else None
            return PatientCheckInAnalysis(
                action="need_help",
                summary=(
                    f"Patient asked to contact {name}."
                    if name
                    else "Patient asked to contact someone."
                ),
                requested_contact=name,
                detected_language="en",
            )

    if has_any("false alarm", "cancel it", "cancel this", "cancel the alert"):
        return PatientCheckInAnalysis(
            action="false_alarm",
            summary="Patient said this was a false alarm.",
            detected_language="en",
        )

    if has_any("check on me", "recheck me", "check again"):
        return PatientCheckInAnalysis(
            action="schedule_recheck",
            summary="Patient asked to be checked on again later.",
            follow_up_minutes=_extract_minutes(text),
            detected_language="en",
        )

    if has_any(
        "treating it", "i'm treating", "im treating", "drank juice",
        "drank some juice", "i drank juice", "glucose tab", "ate something",
        "eating something",
    ):
        return PatientCheckInAnalysis(
            action="treating",
            summary="Patient indicated they are treating the low.",
            detected_language="en",
        )

    if has_any(
        "i'm okay", "im okay", "i am okay", "i'm ok", "im ok", "i am ok",
        "i'm fine", "im fine", "i am fine", "feeling fine",
    ):
        return PatientCheckInAnalysis(
            action="okay",
            summary="Patient said they are okay.",
            responsive=True,
            detected_language="en",
        )

    return PatientCheckInAnalysis(
        action="unknown",
        summary="Transcript did not clearly match a recognized check-in response.",
        detected_language="en" if not _looks_non_english(text) else "unknown-non-english",
    )


# --- Failure-stage reporting -----------------------------------------------
#
# `analyze_patient_checkin` and `generate_caregiver_handoff` can fail to get
# a usable Gemma result at several distinct stages. Collapsing all of them
# into one generic reason (as this module used to) makes real failures
# undiagnosable. Each stage below is reported with only *safe* diagnostic
# detail - never API keys, prompts, hidden reasoning, complete raw SDK
# objects, or full patient transcripts.
#
# The stages this module distinguishes (see `analyze_patient_checkin`):
#   - "sdk_request_failed"       the API call itself raised (network, auth,
#                                 quota, etc.)
#   - "empty_model_response"     the call succeeded but returned no usable
#                                 text/parsed content, and the SDK did NOT
#                                 explicitly report a safety block
#   - "safety_blocked"           the call succeeded but returned no usable
#                                 text/parsed content, and the SDK's own
#                                 finish/block reason explicitly says so
#                                 (e.g. `finish_reason=SAFETY`) - never
#                                 inferred from empty text alone
#   - "json_parse_failed"        response text was present but no valid,
#                                 complete JSON object could be extracted
#                                 from it (after stripping an optional
#                                 Markdown fence and any surrounding prose)
#   - "schema_validation_failed" JSON parsed but didn't match
#                                 `PatientCheckInAnalysis`/`CaregiverHandoff`
#   - "safety_scan_failed"       passed schema validation but tripped
#                                 `_contains_unsafe_content`
#   - "semantic_validation" (not a failure/fallback - reported via the
#                                 "gemma" source's `note` as
#                                 "semantic_correction:...", see
#                                 `describe_semantic_correction`)
#
# `json_parse_failed` and `schema_validation_failed` are retried exactly
# once (see `analyze_patient_checkin`) with a stricter, more deterministic
# request before giving up - a model occasionally wraps valid JSON in
# Markdown or adds a stray sentence, which is a formatting slip, not a
# reason to distrust the whole interpretation.


def _response_finish_reason(response: object) -> object | None:
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        return getattr(candidates[0], "finish_reason", None)
    return None


def _response_block_reason(response: object) -> object | None:
    feedback = getattr(response, "prompt_feedback", None)
    return getattr(feedback, "block_reason", None) if feedback is not None else None


def _safe_reason_name(value: object | None) -> str | None:
    """Renders an SDK enum/string reason code safely (name only, never a
    full repr of the underlying object)."""
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def _is_explicit_safety_block(finish_reason: object, block_reason: object) -> bool:
    """True only when the SDK itself explicitly reports a safety block -
    NEVER inferred merely from an empty/missing response. Distinguishing
    this from a generic `empty_model_response` avoids the earlier,
    unproven assumption that every empty response was a safety block.
    """
    for value in (finish_reason, block_reason):
        name = _safe_reason_name(value)
        if name and "SAFETY" in name.upper():
            return True
    return False


def _response_diagnostics(
    response: object,
    raw_text: str,
    *,
    had_fence: bool = False,
    exception_class: str | None = None,
) -> str:
    """Builds a safe diagnostic string: finish reason, response text
    character length, whether it was blank, whether it contained a
    Markdown fence, and (when applicable) an exception class name. Never
    includes prompts, hidden reasoning, API keys, or the raw SDK object.
    """
    parts = [
        f"finish_reason={_safe_reason_name(_response_finish_reason(response))}",
        f"text_length={len(raw_text)}",
        f"blank={not raw_text.strip()}",
        f"markdown_fence={had_fence}",
    ]
    block_reason = _safe_reason_name(_response_block_reason(response))
    if block_reason:
        parts.append(f"block_reason={block_reason}")
    if exception_class:
        parts.append(f"exception={exception_class}")
    return "; ".join(parts)


def _summarize_validation_error(exc: ValidationError) -> str:
    """Safe (no field values) diagnostic naming which fields failed schema."""
    fields = sorted({".".join(str(part) for part in err["loc"]) for err in exc.errors()})
    return f"{len(fields)} field(s) failed schema validation: {', '.join(fields)}"


_JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_markdown_fence(text: str) -> tuple[str, bool]:
    """Strips a single optional ```json / ``` fence wrapping the whole
    response. Returns (unwrapped_text, had_fence)."""
    stripped = text.strip()
    match = _JSON_FENCE_PATTERN.match(stripped)
    if match:
        return match.group(1).strip(), True
    return stripped, False


def _extract_balanced_json_object(text: str) -> str | None:
    """Finds the first complete, brace-balanced JSON object in `text`,
    tolerating harmless leading/trailing prose around it (e.g. "Here is
    the result: {...} Let me know if you need anything else."). Returns
    None if no balanced object is found. Braces inside JSON string
    literals are tracked so they never affect the balance count.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_parsed_detailed(
    response: object, model_cls: type
) -> tuple[Any | None, str | None, str | None]:
    """Extracts `model_cls` from a Gemini response with a precise failure
    stage. Returns (parsed_or_None, failure_stage, safe_detail); the last
    two are None on success.

    Order: (1) prefer `response.parsed` when it's already the expected
    type: (2) otherwise read `response.text` safely; (3) strip an optional
    Markdown fence; (4) extract the first balanced JSON object, tolerating
    harmless leading/trailing prose; (5) `json.loads`; (6) validate with
    `model_cls`.
    """
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, model_cls):
        return parsed, None, None

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        finish_reason = _response_finish_reason(response)
        block_reason = _response_block_reason(response)
        stage = (
            "safety_blocked"
            if _is_explicit_safety_block(finish_reason, block_reason)
            else "empty_model_response"
        )
        return None, stage, _response_diagnostics(response, raw_text)

    unwrapped, had_fence = _strip_markdown_fence(raw_text)
    candidate = _extract_balanced_json_object(unwrapped)
    if candidate is None:
        return (
            None,
            "json_parse_failed",
            _response_diagnostics(
                response, raw_text, had_fence=had_fence, exception_class="NoJsonObjectFound"
            ),
        )

    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        return (
            None,
            "json_parse_failed",
            _response_diagnostics(
                response, raw_text, had_fence=had_fence, exception_class=exc.__class__.__name__
            ),
        )

    try:
        return model_cls.model_validate(data), None, None
    except ValidationError as exc:
        detail = (
            f"{_summarize_validation_error(exc)}; text_length={len(raw_text)}; "
            f"markdown_fence={had_fence}"
        )
        return None, "schema_validation_failed", detail


# Failure stages that are worth one strict, low-temperature retry: a model
# occasionally wraps valid JSON in prose/Markdown or produces slightly
# malformed JSON, which is a formatting slip worth one clean retry - unlike
# an empty/blocked response or a transport failure, which a retry with the
# same input is unlikely to fix.
_RETRYABLE_STAGES = frozenset({"json_parse_failed", "schema_validation_failed"})

# The lowest temperature the Gemini API supports, for the single retry -
# maximizes determinism/instruction-following on the second attempt.
_RETRY_TEMPERATURE = 0.0

_STRICT_JSON_REMINDER = (
    "\n\nIMPORTANT: Your previous response could not be parsed. Respond "
    "with exactly one JSON object and nothing else - no prose, no "
    "explanation, and no Markdown code fences before, after, or around it."
)


def _request_analysis(client, model_name: str, transcript: str, types_module, *, strict: bool):
    """Single Gemma call for `analyze_patient_checkin`. No safety_settings
    override - Gemini's default upstream content-safety behavior is used
    unless a separately documented, tested reason justifies changing it
    (none currently exists; see module history/README)."""
    return client.models.generate_content(
        model=model_name,
        contents=_build_analysis_contents(transcript, strict=strict),
        config=types_module.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PatientCheckInAnalysis,
            temperature=_RETRY_TEMPERATURE if strict else 0.1,
        ),
    )


def analyze_patient_checkin(
    transcript: str,
) -> tuple[PatientCheckInAnalysis, InterpretationSource, str | None]:
    """Interprets a patient transcript, preferring Gemma with a safe fallback.

    Returns (analysis, source, note). This function has no side effects
    beyond the Gemma call(s) themselves - it never proposes, validates, or
    executes an application tool - so retrying the Gemma request internally
    (see below) can never cause a duplicate application action; the caller
    always receives exactly one final `PatientCheckInAnalysis`, and only
    that one result ever reaches `app.tools`.

    When `source == "fallback"`, `note` is `"<stage>:<safe detail>"` (or
    just `"<stage>"` when there's no extra detail), optionally suffixed
    with `";retry_attempted=true"`, identifying exactly which stage
    produced an unusable result - one of "gemma_disabled",
    "gemma_not_configured", "sdk_request_failed", "empty_model_response",
    "safety_blocked", "json_parse_failed", "schema_validation_failed", or
    "safety_scan_failed". `<safe detail>` never includes API keys, prompts,
    hidden reasoning, complete raw SDK objects, or the full patient
    transcript - only structural diagnostics (exception class names, which
    fields failed, Gemini's own block/finish reason codes, response text
    length, and whether it was blank/fenced).

    `json_parse_failed` and `schema_validation_failed` are retried exactly
    once with a stricter, lowest-temperature request before falling back
    (see `_RETRYABLE_STAGES`) - a model occasionally wraps valid JSON in
    Markdown or prose, which is a formatting slip, not proof the
    interpretation itself is untrustworthy.

    When `source == "gemma"`, `note` is `None` (Gemma's classification was
    used as-is on the first attempt), or a `;`-separated combination of
    `"retry_attempted=true"` (the result came from the one retry) and/or a
    `"semantic_correction:<reason>[,<reason>...]"` string describing which
    deterministic backend safety correction(s) from
    `_apply_deterministic_corrections` were applied - neither is a Gemma
    failure, so `source` stays "gemma" (see `describe_semantic_correction`).
    """
    client, model_name = _get_client_and_model()
    if client is None or model_name is None:
        reason = "gemma_disabled" if not _gemma_enabled() else "gemma_not_configured"
        analysis, _ = _apply_deterministic_corrections(
            transcript, _fallback_interpret(transcript)
        )
        return analysis, "fallback", reason

    try:
        from google.genai import types
    except Exception as exc:  # pragma: no cover - defensive; _get_client_and_model already imports this
        logger.warning("Gemma request failed, using fallback parser: %s", exc)
        analysis, _ = _apply_deterministic_corrections(
            transcript, _fallback_interpret(transcript)
        )
        return analysis, "fallback", f"sdk_request_failed:{exc.__class__.__name__}"

    retry_attempted = False
    analysis: PatientCheckInAnalysis | None = None
    stage: str | None = None
    detail: str | None = None

    for attempt_is_retry in (False, True):
        try:
            response = _request_analysis(
                client, model_name, transcript, types, strict=attempt_is_retry
            )
        except Exception as exc:
            logger.warning("Gemma request failed, using fallback parser: %s", exc)
            analysis, _ = _apply_deterministic_corrections(
                transcript, _fallback_interpret(transcript)
            )
            reason = f"sdk_request_failed:{exc.__class__.__name__}"
            if retry_attempted:
                reason += ";retry_attempted=true"
            return analysis, "fallback", reason

        analysis, stage, detail = _extract_parsed_detailed(response, PatientCheckInAnalysis)
        if analysis is not None:
            break

        if not attempt_is_retry and stage in _RETRYABLE_STAGES:
            retry_attempted = True
            logger.warning(
                "Gemma output failed (%s), retrying once with a stricter prompt", stage
            )
            continue
        break

    if analysis is None:
        logger.warning(
            "Gemma output could not be used (%s: %s, retry_attempted=%s), using fallback parser",
            stage, detail, retry_attempted,
        )
        analysis, _ = _apply_deterministic_corrections(
            transcript, _fallback_interpret(transcript)
        )
        reason = f"{stage}:{detail}" if detail else stage
        if retry_attempted:
            reason += ";retry_attempted=true"
        return analysis, "fallback", reason

    unsafe = _contains_unsafe_content(
        analysis.summary,
        analysis.english_summary,
        analysis.reported_condition,
        analysis.reported_action,
    )
    if unsafe:
        logger.warning("Gemma output failed safety re-scan (%s), using fallback", unsafe)
        analysis, _ = _apply_deterministic_corrections(
            transcript, _fallback_interpret(transcript)
        )
        reason = f"safety_scan_failed:{unsafe}"
        if retry_attempted:
            reason += ";retry_attempted=true"
        return analysis, "fallback", reason

    analysis, corrections = _apply_deterministic_corrections(transcript, analysis)
    note_parts = []
    if retry_attempted:
        note_parts.append("retry_attempted=true")
    if corrections:
        note_parts.append(f"semantic_correction:{','.join(corrections)}")
    note = ";".join(note_parts) if note_parts else None
    return analysis, "gemma", note


# --- Grounded caregiver handoff generation ---------------------------------

HANDOFF_SYSTEM_INSTRUCTION = """You are drafting a short handoff message for a \
caregiver in a Type 1 diabetes emergency-coordination prototype. You will be \
given ONLY verified backend facts. Use ONLY those facts.

STRICT RULES:
- Use only the facts supplied to you. Never infer or invent anything, \
including symptoms, locations, contacts, or causes.
- Always label the glucose reading as simulated.
- Never give treatment advice or recommend medication or a dosage.
- Never state or imply that the patient is unconscious.
- If a fact was not supplied (e.g. no location was shared), do not guess - \
list it under unknown_information instead.
- Keep the handoff concise (2-4 sentences) and factual, not dramatic or \
alarmist.
- Always write the handoff and headline in English, even if the original \
patient transcript was in another language - but preserve the detected \
original language in the detected_language field.
- Respond with structured JSON only, matching the provided schema.
"""


def _build_handoff_contents(facts: dict) -> str:
    facts_json = json.dumps(facts, indent=2, default=str)
    return (
        f"{HANDOFF_SYSTEM_INSTRUCTION}\n\nVerified facts (JSON):\n{facts_json}"
    )


def _verified_handoff_facts(event: EmergencyEvent, analysis: PatientCheckInAnalysis) -> dict:
    reading = event.latest_reading
    location_shared = (
        event.location_latitude is not None and event.location_longitude is not None
    )
    return {
        "simulated_glucose_reading_mg_dl": reading.value_mg_dl,
        "trend": reading.trend.value,
        "event_reason": event.reason,
        "event_status": event.status.value,
        "original_patient_transcript": event.patient_transcript,
        "patient_action_classified_as": analysis.action,
        "patient_responsive": analysis.responsive,
        "reported_condition": analysis.reported_condition,
        "reported_action": analysis.reported_action,
        "requested_contact": analysis.requested_contact,
        "supply_location": analysis.supply_location,
        "location_shared": location_shared,
        "check_in_deadline_expired": (
            event.check_in_deadline is not None
        ),
        "detected_language": analysis.detected_language,
    }


def get_default_caregiver_name() -> str:
    """The generic caregiver display name used when the patient did not
    explicitly name a contact. Never overrides an explicitly stated name.
    """
    return os.getenv("DEFAULT_CAREGIVER_NAME", "Helper").strip() or "Helper"


def _apply_default_caregiver_name(handoff: CaregiverHandoff) -> CaregiverHandoff:
    """Fills in the generic default caregiver name only when no contact was
    explicitly stated by the patient. Idempotent - safe to call on any
    handoff regardless of its source (Gemma or fallback).
    """
    if handoff.requested_contact:
        return handoff
    default_name = get_default_caregiver_name()
    unknown_information = list(handoff.unknown_information)
    if not any("contact" in item.lower() for item in unknown_information):
        unknown_information.append(
            f"The patient did not name a specific contact; defaulting to {default_name}."
        )
    return handoff.model_copy(
        update={"requested_contact": default_name, "unknown_information": unknown_information}
    )


def _fallback_handoff(event: EmergencyEvent, analysis: PatientCheckInAnalysis) -> CaregiverHandoff:
    """Deterministic handoff assembled only from verified fields - no Gemma call."""
    reading = event.latest_reading
    location_shared = (
        event.location_latitude is not None and event.location_longitude is not None
    )
    contact_display = analysis.requested_contact or get_default_caregiver_name()

    sentence_parts = [f"Simulated glucose reading: {reading.value_mg_dl} mg/dL ({reading.trend.value})."]
    if analysis.responsive:
        sentence_parts.append("The patient is responsive.")
    if analysis.reported_condition:
        sentence_parts.append(f"They reported: {analysis.reported_condition}.")
    if analysis.reported_action:
        sentence_parts.append(f"Reported action: {analysis.reported_action}.")
    if analysis.requested_contact:
        sentence_parts.append(f"They asked that {analysis.requested_contact} be contacted.")
    else:
        sentence_parts.append(
            f"No specific contact was named; notifying {contact_display}."
        )

    unknown_information: list[str] = []
    if analysis.reported_condition is None:
        unknown_information.append("Reported condition was not stated.")
    if analysis.supply_location is None:
        unknown_information.append("Supply location was not stated.")
    if not location_shared:
        unknown_information.append("Location was not shared.")
    if not analysis.requested_contact:
        unknown_information.append(
            f"The patient did not name a specific contact; defaulting to {contact_display}."
        )

    return CaregiverHandoff(
        headline="Patient requested caregiver assistance",
        handoff=" ".join(sentence_parts),
        patient_is_responsive=analysis.responsive,
        reported_condition=analysis.reported_condition,
        reported_action=analysis.reported_action,
        requested_contact=contact_display,
        supply_location=analysis.supply_location,
        location_shared=location_shared,
        unknown_information=unknown_information,
        detected_language=analysis.detected_language,
    )


def _handoff_enabled() -> bool:
    return os.getenv("ENABLE_GEMMA_HANDOFF", "true").strip().lower() == "true"


def generate_caregiver_handoff(
    event: EmergencyEvent,
    analysis: PatientCheckInAnalysis,
) -> tuple[CaregiverHandoff, HandoffSource, str | None]:
    """Generates a grounded caregiver handoff, preferring Gemma with a fallback.

    Only ever called after the backend has already validated an escalation
    (see app.main.voice_check_in) - never used to decide whether to escalate.
    """
    handoff, source, reason = _generate_caregiver_handoff_raw(event, analysis)
    return _apply_default_caregiver_name(handoff), source, reason


def _generate_caregiver_handoff_raw(
    event: EmergencyEvent,
    analysis: PatientCheckInAnalysis,
) -> tuple[CaregiverHandoff, HandoffSource, str | None]:
    if not _handoff_enabled():
        return _fallback_handoff(event, analysis), "fallback", "gemma_handoff_disabled"

    client, model_name = _get_client_and_model()
    if client is None or model_name is None:
        return _fallback_handoff(event, analysis), "fallback", "gemma_not_configured"

    try:
        from google.genai import types

        facts = _verified_handoff_facts(event, analysis)
        response = client.models.generate_content(
            model=model_name,
            contents=_build_handoff_contents(facts),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CaregiverHandoff,
                temperature=0.2,
            ),
        )
    except Exception as exc:
        logger.warning("Gemma handoff request failed, using fallback: %s", exc)
        return (
            _fallback_handoff(event, analysis),
            "fallback",
            f"gemma_handoff_sdk_request_failed:{exc.__class__.__name__}",
        )

    handoff, stage, detail = _extract_parsed_detailed(response, CaregiverHandoff)
    if handoff is None:
        logger.warning("Gemma handoff output could not be used (%s: %s), using fallback", stage, detail)
        reason = f"gemma_handoff_{stage}:{detail}" if detail else f"gemma_handoff_{stage}"
        return _fallback_handoff(event, analysis), "fallback", reason

    unsafe = _contains_unsafe_content(handoff.headline, handoff.handoff)
    if unsafe:
        logger.warning("Gemma handoff failed safety re-scan (%s), using fallback", unsafe)
        return (
            _fallback_handoff(event, analysis),
            "fallback",
            f"gemma_handoff_safety_scan_failed:{unsafe}",
        )

    return handoff, "gemma", None


def _extract_parsed(response, model_cls):
    parsed, _stage, _detail = _extract_parsed_detailed(response, model_cls)
    return parsed
