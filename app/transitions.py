"""Single source of truth for EmergencyEvent state transitions.

Every route that changes an event's status must go through
`ensure_transition_allowed` (or `is_transition_allowed`) so the rules below
are never duplicated or drift apart between endpoints.
"""

from app.models import EventStatus

ALLOWED_TRANSITIONS: dict[EventStatus, set[EventStatus]] = {
    EventStatus.CHECK_IN_REQUIRED: {
        EventStatus.MONITORING,
        EventStatus.CONTACTING,
        EventStatus.RESOLVED,
    },
    EventStatus.MONITORING: {
        EventStatus.CONTACTING,
        EventStatus.RESOLVED,
    },
    EventStatus.CONTACTING: {
        EventStatus.ACKNOWLEDGED,
        EventStatus.RESOLVED,
    },
    EventStatus.ACKNOWLEDGED: {
        EventStatus.RESOLVED,
    },
    EventStatus.RESOLVED: set(),
}


class InvalidTransitionError(Exception):
    """Raised when an event cannot move from its current status to a target one."""

    def __init__(self, current: EventStatus, target: EventStatus) -> None:
        self.current = current
        self.target = target
        self.allowed = sorted(s.value for s in allowed_next_statuses(current))
        super().__init__(
            f"Cannot transition event from '{current.value}' to '{target.value}'"
        )


def allowed_next_statuses(current: EventStatus) -> set[EventStatus]:
    return ALLOWED_TRANSITIONS.get(current, set())


def is_transition_allowed(current: EventStatus, target: EventStatus) -> bool:
    return target in allowed_next_statuses(current)


def ensure_transition_allowed(current: EventStatus, target: EventStatus) -> None:
    if not is_transition_allowed(current, target):
        raise InvalidTransitionError(current, target)
