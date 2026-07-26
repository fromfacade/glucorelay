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
from typing import Literal

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
_UNSAFE_PHRASES = (
    "unconscious",
    "unresponsive and not breathing",
    "insulin",
    "dosage",
    " dose ",
    "milligram",
    "medication",
    "diagnos",
    "inject",
    "overdose",
)


def _contains_unsafe_content(*texts: str | None) -> str | None:
    for text in texts:
        if not text:
            continue
        lowered = f" {text.lower()} "
        for phrase in _UNSAFE_PHRASES:
            if phrase in lowered:
                return phrase.strip()
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
_OKAY_EVIDENCE = (
    "i'm okay", "im okay", "i am okay",
    "i'm ok", "im ok", "i am ok",
    "i'm fine", "im fine", "i am fine",
    "i feel okay", "i feel fine",
    "everything is okay", "everything's okay", "everything is fine",
    "i'm alright", "im alright", "i am alright",
    "feeling fine", "feeling okay",
    # Spanish equivalents (practical subset, not exhaustive).
    "estoy bien", "me siento bien", "todo está bien", "todo esta bien",
)

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

_NEED_HELP_EVIDENCE = (
    "need help", "i need help", "help me", "send help", "come get me",
    "call someone", "call ", "contact ", "please call", "please contact",
    "can't stand", "cant stand", "can't move", "cant move", "can't get up",
    "cant get up", "unable to stand", "unable to move", "i'm alone",
    "im alone", "i am alone", "too weak", "too dizzy", "feel faint",
    "about to pass out", "confus",
    # Spanish equivalents (practical subset, not exhaustive).
    "ayuda", "necesito ayuda", "llama", "confundid", "no puedo pararme",
    "no puedo levantarme", "estoy sola", "estoy solo",
)

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
    if not detected_language:
        return True
    normalized = detected_language.strip().lower()
    return (
        normalized in _ENGLISH_TAGS
        or normalized.startswith("en-")
        or normalized.startswith("en_")
    )


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
    # from an unrelated statement.
    if not _has_evidence(lowered, corrected.action):
        original_action = corrected.action
        corrected = corrected.model_copy(update={"action": "unknown"})
        reasons.append(f"unsupported_action_downgraded:{original_action}")

    # Never let a translation/rewrite attempt alter an already-English
    # summary - english_summary must be exactly the validated summary.
    if _is_english(corrected.detected_language) and corrected.english_summary != corrected.summary:
        corrected = corrected.model_copy(update={"english_summary": corrected.summary})
        reasons.append("english_summary_normalized")

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


def _build_analysis_contents(transcript: str) -> str:
    return f'{ANALYSIS_SYSTEM_INSTRUCTION}\n\nPatient transcript:\n"""{transcript}"""'


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


def analyze_patient_checkin(
    transcript: str,
) -> tuple[PatientCheckInAnalysis, InterpretationSource, str | None]:
    """Interprets a patient transcript, preferring Gemma with a safe fallback.

    Returns (analysis, source, note). When `source == "fallback"`, `note`
    explains why the fallback was used. When `source == "gemma"`, `note` is
    either None (Gemma's classification was used as-is) or a
    "semantic_correction:<reason>[,<reason>...]" string describing which
    deterministic backend safety correction(s) from
    `_apply_deterministic_corrections` were applied - this is NOT a Gemma
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

        response = client.models.generate_content(
            model=model_name,
            contents=_build_analysis_contents(transcript),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PatientCheckInAnalysis,
                temperature=0.1,
            ),
        )
    except Exception as exc:
        logger.warning("Gemma request failed, using fallback parser: %s", exc)
        analysis, _ = _apply_deterministic_corrections(
            transcript, _fallback_interpret(transcript)
        )
        return analysis, "fallback", "gemma_request_failed"

    analysis = _extract_parsed(response, PatientCheckInAnalysis)
    if analysis is None:
        logger.warning("Gemma returned unusable output, using fallback parser")
        analysis, _ = _apply_deterministic_corrections(
            transcript, _fallback_interpret(transcript)
        )
        return analysis, "fallback", "gemma_invalid_output"

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
        return (
            analysis,
            "fallback",
            f"gemma_unsafe_content:{unsafe}",
        )

    analysis, corrections = _apply_deterministic_corrections(transcript, analysis)
    note = f"semantic_correction:{','.join(corrections)}" if corrections else None
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
        return _fallback_handoff(event, analysis), "fallback", "gemma_handoff_request_failed"

    handoff = _extract_parsed(response, CaregiverHandoff)
    if handoff is None:
        logger.warning("Gemma handoff output was unusable, using fallback")
        return _fallback_handoff(event, analysis), "fallback", "gemma_handoff_invalid_output"

    unsafe = _contains_unsafe_content(handoff.headline, handoff.handoff)
    if unsafe:
        logger.warning("Gemma handoff failed safety re-scan (%s), using fallback", unsafe)
        return (
            _fallback_handoff(event, analysis),
            "fallback",
            f"gemma_handoff_unsafe_content:{unsafe}",
        )

    return handoff, "gemma", None


def _extract_parsed(response, model_cls):
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, model_cls):
        return parsed

    raw_text = getattr(response, "text", None)
    if raw_text:
        try:
            data = json.loads(raw_text)
            return model_cls.model_validate(data)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return None
    return None
