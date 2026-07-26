"""Deterministic proposal, validation, and execution of Gemma-driven tools.

WHY THIS MODULE EXISTS (safety boundary): Gemma is never allowed to execute
anything. Its job ends at producing a `PatientCheckInAnalysis` (see
`app.gemma_service`); everything from there on - which tool that analysis
implies, whether it's actually permitted right now, and what effect it has
on the event - is deterministic Python.

ON "TOOL SELECTION": Gemma is accessed here through the Gemini API's
structured-output mechanism (`response_schema`), not native function
calling. Google's own docs for Gemma-family models describe function
calling only for locally-hosted Gemma via `transformers` +
`apply_chat_template`, with the caller manually parsing the model's text
output - not the Gemini API's `tools`/`function_declarations` parameter,
which is documented for Gemini reasoning models. Since GEMMA_MODEL here is
called through the Gemini API, this project does not claim native
function calling. Instead, `propose_tool` below deterministically derives
a `ProposedToolCall` from Gemma's `action` field, which is itself
constrained to one of six literal values by the JSON schema Gemma must
follow - that schema constraint is the "equivalent constrained tool
selection" mechanism in place of native tool calling.
"""

import re
from datetime import datetime, timezone
from typing import Any

from app.models import EmergencyEvent, EventStatus, PatientCheckInAnalysis, ProposedToolCall
from app.transitions import InvalidTransitionError, ensure_transition_allowed

ValidatedToolCall = ProposedToolCall


class ToolValidationError(Exception):
    """Raised when a proposed tool call fails backend validation."""


ACTION_TO_TOOL: dict[str, str] = {
    "okay": "record_patient_okay",
    "treating": "record_patient_treating",
    "need_help": "request_caregiver_help",
    "schedule_recheck": "schedule_patient_recheck",
    "false_alarm": "resolve_false_alarm",
    "unknown": "report_unclear_response",
}

# The status a tool moves an event to. Tools absent from this map (only
# `report_unclear_response`) never change status.
TOOL_TO_STATUS: dict[str, EventStatus] = {
    "record_patient_okay": EventStatus.MONITORING,
    "record_patient_treating": EventStatus.MONITORING,
    "request_caregiver_help": EventStatus.CONTACTING,
    "schedule_patient_recheck": EventStatus.MONITORING,
    "resolve_false_alarm": EventStatus.RESOLVED,
}

# Which tool-argument keys each tool may carry, and (when a tool argument's
# name differs from the PatientCheckInAnalysis field it must be verified
# against) the analysis field it's sourced from/checked against. Keeping
# this minimal (rather than mirroring every analysis field) makes
# "unsupported field" and "argument must match verified analysis" checks
# meaningful rather than vacuous.
TOOL_ALLOWED_ARGUMENTS: dict[str, set[str]] = {
    "record_patient_okay": set(),
    "record_patient_treating": set(),
    "request_caregiver_help": set(),
    "schedule_patient_recheck": {"minutes"},
    "resolve_false_alarm": set(),
    "report_unclear_response": set(),
}

# `schedule_patient_recheck`'s "minutes" argument is sourced from/verified
# against `PatientCheckInAnalysis.follow_up_minutes` - the tool argument
# name intentionally differs from the analysis field name.
TOOL_ARGUMENT_SOURCE_FIELD: dict[str, dict[str, str]] = {
    "schedule_patient_recheck": {"minutes": "follow_up_minutes"},
}

MIN_FOLLOW_UP_MINUTES = 1
MAX_FOLLOW_UP_MINUTES = 180


def _analysis_field_for_argument(tool_name: str, argument_key: str) -> str:
    return TOOL_ARGUMENT_SOURCE_FIELD.get(tool_name, {}).get(argument_key, argument_key)

# Defense-in-depth: even though tool arguments are drawn from a small
# allow-list of factual fields, scan any string argument for language that
# would indicate a treatment/medication recommendation slipped through.
#
# SCOPE: these entries target generated MEDICAL INSTRUCTIONS (a treatment
# command, a dosage, an insulin recommendation) - never a patient-reported
# physical condition. Phrases like "can't stand", "unable to move", or
# "alone" must NEVER be added here - they are exactly the kind of explicit
# symptom `need_help` exists to capture. Matched on word boundaries (see
# `_scan_for_treatment_language`) so an entry can never fire on part of an
# unrelated word.
_TREATMENT_DENYLIST = (
    "insulin",
    "dose",
    "dosage",
    "milligram",
    "mg of",
    "unit of",
    "units of",
    "medication",
    "inject",
    "administer",
)

_TREATMENT_DENYLIST_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _TREATMENT_DENYLIST) + r")\w*\b"
)


def propose_tool(analysis: PatientCheckInAnalysis) -> ProposedToolCall:
    """Deterministically derives a tool proposal from Gemma's constrained action.

    This is pure Python - it is not a second call to Gemma.
    """
    name = ACTION_TO_TOOL[analysis.action]
    arguments: dict[str, Any] = {}
    for argument_key in TOOL_ALLOWED_ARGUMENTS[name]:
        field_name = _analysis_field_for_argument(name, argument_key)
        value = getattr(analysis, field_name, None)
        if value is not None:
            arguments[argument_key] = value
    return ProposedToolCall(name=name, arguments=arguments)


def _scan_for_treatment_language(arguments: dict[str, Any]) -> str | None:
    for value in arguments.values():
        if isinstance(value, str):
            match = _TREATMENT_DENYLIST_PATTERN.search(value.lower())
            if match:
                return match.group(0)
    return None


def validate_tool_call(
    event: EmergencyEvent,
    analysis: PatientCheckInAnalysis,
    proposed_tool: ProposedToolCall,
) -> ValidatedToolCall:
    """Validates a proposed tool call. Raises ToolValidationError if unsafe/illegal.

    Checks (per spec): supported tool name, current status permits it,
    every argument matches a fact Gemma actually extracted (never
    invented), no unsupported argument fields, numeric ranges are sane,
    no treatment/medication language, and the implied transition is legal.
    """
    name = proposed_tool.name
    if name not in TOOL_ALLOWED_ARGUMENTS:
        raise ToolValidationError(f"Unsupported tool '{name}'.")

    if event.status != EventStatus.CHECK_IN_REQUIRED:
        raise ToolValidationError(
            f"Tool '{name}' is not permitted while the event is '{event.status.value}'."
        )

    allowed_keys = TOOL_ALLOWED_ARGUMENTS[name]
    unexpected = set(proposed_tool.arguments) - allowed_keys
    if unexpected:
        raise ToolValidationError(
            f"Tool '{name}' included unsupported argument fields: {sorted(unexpected)}."
        )

    for key, value in proposed_tool.arguments.items():
        field_name = _analysis_field_for_argument(name, key)
        verified_value = getattr(analysis, field_name, None)
        if verified_value is None:
            raise ToolValidationError(
                f"Argument '{key}' was not actually stated by the patient."
            )
        if verified_value != value:
            raise ToolValidationError(
                f"Argument '{key}' does not match the verified analysis."
            )

    if "minutes" in proposed_tool.arguments:
        minutes = proposed_tool.arguments["minutes"]
        if not isinstance(minutes, int) or not (
            MIN_FOLLOW_UP_MINUTES <= minutes <= MAX_FOLLOW_UP_MINUTES
        ):
            raise ToolValidationError(
                "minutes must be an integer between "
                f"{MIN_FOLLOW_UP_MINUTES} and {MAX_FOLLOW_UP_MINUTES}."
            )

    banned = _scan_for_treatment_language(proposed_tool.arguments)
    if banned:
        raise ToolValidationError(
            f"Tool arguments referenced medical treatment ('{banned}')."
        )

    target = TOOL_TO_STATUS.get(name)
    if target is not None:
        try:
            ensure_transition_allowed(event.status, target)
        except InvalidTransitionError as exc:
            raise ToolValidationError(str(exc)) from exc

    return ProposedToolCall(name=name, arguments=dict(proposed_tool.arguments))


def execute_validated_tool(event: EmergencyEvent, tool: ValidatedToolCall) -> bool:
    """Applies a validated tool's state effect. Returns True if status changed.

    Timeline entries, timer cancellation, and notifications are the
    caller's responsibility (see app.main) since they depend on services
    this module intentionally has no knowledge of.
    """
    target = TOOL_TO_STATUS.get(tool.name)
    if target is None or event.status == target:
        return False
    event.status = target
    event.updated_at = datetime.now(timezone.utc)
    return True
