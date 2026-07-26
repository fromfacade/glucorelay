"""Reusable helper for appending factual incident-timeline entries.

Every state change, notification attempt, and voice interpretation should be
recorded here so the full incident can be reconstructed later. Messages must
describe application events only - never invented medical claims.
"""

from typing import Any

from app.models import EmergencyEvent, TimelineEntry


def add_timeline_entry(
    event: EmergencyEvent,
    event_type: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> TimelineEntry:
    entry = TimelineEntry(event_type=event_type, message=message, metadata=metadata)
    event.timeline.append(entry)
    return entry
