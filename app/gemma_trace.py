"""Reusable helper for building a safe, presentation-ready Gemma trace.

A GemmaTrace records *that* something happened (input received, Gemma
called, a tool proposed/validated/executed, a handoff generated) - never
API keys, raw prompts, hidden model reasoning, full SDK responses, or
stack traces.
"""

from typing import Any

from app.models import GemmaTrace, GemmaTraceStage, GemmaTraceStep


def new_trace() -> GemmaTrace:
    return GemmaTrace()


def add_trace_step(
    trace: GemmaTrace,
    stage: GemmaTraceStage,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> GemmaTraceStep:
    step = GemmaTraceStep(stage=stage, message=message, metadata=metadata)
    trace.steps.append(step)
    return step
