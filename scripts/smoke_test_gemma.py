"""Manual, real-network smoke test for the configured Gemma model.

This is NOT part of the automated pytest suite (it requires real
credentials and network access). Run it by hand after configuring `.env`
with a real GEMINI_API_KEY and GEMMA_MODEL:

    python scripts/smoke_test_gemma.py

It refuses to run if configuration is missing, sends several representative
transcripts through the real `analyze_patient_checkin` pipeline, prints the
validated structured result and proposed tool for each, and - critically -
asserts the expected *semantic* outcome for each transcript, not just that
the response was schema-valid. It exits nonzero if any transcript:

- Falls back to the deterministic parser (meaning Gemma was not actually
  exercised), or
- Fails schema/safety validation, or
- Produces a semantically wrong result even though it was schema-valid -
  e.g. "My backpack is red." classified as "okay", a follow-up recheck
  request not taking precedence over a general "okay" statement, an
  English transcript's `english_summary` being needlessly reworded, or a
  "Helper" contact request not being extracted.

Never prints the API key.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.gemma_service import analyze_patient_checkin  # noqa: E402
from app.tools import propose_tool  # noqa: E402

# Each case documents the transcript plus the semantic outcome the
# deterministic backend (Gemma output + `_apply_deterministic_corrections`)
# must produce. `expect_*` keys are optional; only the ones present are
# checked.
CASES = [
    {
        "transcript": "I already drank some juice and I'm handling it.",
        "expect_action": "treating",
    },
    {
        "transcript": "I feel confused and I need Helper to help me.",
        "expect_action": "need_help",
        "expect_requested_contact": "Helper",
    },
    {
        "transcript": "I'm awake, but I'm alone and I can't stand up.",
        "expect_action": "need_help",
        "expect_english_summary_equals_summary": True,
    },
    {
        "transcript": "Everything is okay, but check on me again in ten minutes.",
        "expect_action": "schedule_recheck",
        "expect_follow_up_minutes": 10,
        "expect_tool_name": "schedule_patient_recheck",
        "expect_tool_arguments": {"minutes": 10},
    },
    {
        "transcript": "My backpack is red.",
        "expect_action": "unknown",
    },
    {
        "transcript": "Call Helper-actually, never mind, I think I'm okay.",
        "expect_action": "okay",
    },
    {
        "transcript": "Me siento confundida. Por favor llama a Helper.",
        "expect_action": "need_help",
        "expect_requested_contact": "Helper",
        "expect_not_english": True,
    },
]


def _check_case(case: dict, analysis, tool) -> list[str]:
    """Returns a list of human-readable problems (empty if the case passed)."""
    problems: list[str] = []

    expected_action = case.get("expect_action")
    if expected_action is not None and analysis.action != expected_action:
        problems.append(
            f"expected action='{expected_action}', got '{analysis.action}'"
        )

    expected_contact = case.get("expect_requested_contact")
    if expected_contact is not None and analysis.requested_contact != expected_contact:
        problems.append(
            f"expected requested_contact='{expected_contact}', "
            f"got {analysis.requested_contact!r}"
        )

    expected_minutes = case.get("expect_follow_up_minutes")
    if expected_minutes is not None and analysis.follow_up_minutes != expected_minutes:
        problems.append(
            f"expected follow_up_minutes={expected_minutes}, "
            f"got {analysis.follow_up_minutes!r}"
        )

    if case.get("expect_english_summary_equals_summary"):
        if analysis.english_summary != analysis.summary:
            problems.append(
                "expected english_summary to equal summary for an English "
                f"transcript, got summary={analysis.summary!r} "
                f"english_summary={analysis.english_summary!r}"
            )

    if case.get("expect_not_english"):
        if not analysis.english_summary:
            problems.append("expected a populated english_summary for a non-English transcript")
        detected = (analysis.detected_language or "").strip().lower()
        if detected in ("en", "english") or detected.startswith("en-"):
            problems.append(f"expected a non-English detected_language, got {detected!r}")

    expected_tool_name = case.get("expect_tool_name")
    if expected_tool_name is not None and tool.name != expected_tool_name:
        problems.append(f"expected tool '{expected_tool_name}', got '{tool.name}'")

    expected_tool_arguments = case.get("expect_tool_arguments")
    if expected_tool_arguments is not None and tool.arguments != expected_tool_arguments:
        problems.append(
            f"expected tool arguments {expected_tool_arguments!r}, got {tool.arguments!r}"
        )

    return problems


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMMA_MODEL", "").strip()
    if not api_key or not model:
        print(
            "GEMINI_API_KEY and GEMMA_MODEL must both be set in the environment "
            "to run this smoke test. Refusing to run."
        )
        return 1

    print(f"Using model: {model}")
    print("(A GEMINI_API_KEY is configured; it will never be printed.)\n")

    failures = 0
    for case in CASES:
        transcript = case["transcript"]
        print(f"--- Transcript: {transcript!r}")
        try:
            analysis, source, note = analyze_patient_checkin(transcript)
        except Exception as exc:  # pragma: no cover - manual diagnostic script
            print(f"    ERROR calling Gemma: {exc}")
            failures += 1
            continue

        if source != "gemma":
            print(
                f"    FAILED: fell back to the '{source}' parser "
                f"(reason: {note}). Gemma was not actually exercised."
            )
            failures += 1
            continue

        tool = propose_tool(analysis)
        print(f"    action:             {analysis.action}")
        print(f"    summary:            {analysis.summary}")
        print(f"    english_summary:    {analysis.english_summary}")
        print(f"    detected_language:  {analysis.detected_language}")
        print(f"    requested_contact:  {analysis.requested_contact}")
        print(f"    reported_condition: {analysis.reported_condition}")
        print(f"    reported_action:    {analysis.reported_action}")
        print(f"    supply_location:    {analysis.supply_location}")
        print(f"    follow_up_minutes:  {analysis.follow_up_minutes}")
        print(f"    proposed tool:      {tool.name} {tool.arguments}")
        if note:
            print(f"    note:               {note}")

        problems = _check_case(case, analysis, tool)
        if problems:
            print("    SEMANTIC CHECK FAILED:")
            for problem in problems:
                print(f"      - {problem}")
            failures += 1
        else:
            print("    semantic check:     OK")
        print()

    if failures:
        print(
            f"\n{failures} of {len(CASES)} transcript(s) failed validation or "
            "produced the wrong semantic result."
        )
        return 1

    print(f"\nAll {len(CASES)} transcripts produced correct, validated Gemma responses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
