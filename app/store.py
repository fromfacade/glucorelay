import asyncio
from copy import deepcopy

from app.models import EmergencyEvent, Reading


class InMemoryStore:
    """In-memory persistence for the hackathon prototype.

    NOTE: this is intentionally simple (single active event, list history,
    in-process asyncio timers). It is appropriate for a demo but would need
    to be replaced with a durable database and job queue in production.
    Services in this codebase are written against this store's small
    interface so that swap can happen without touching route logic.
    """

    def __init__(self) -> None:
        self.latest_reading: Reading | None = None
        self.active_event: EmergencyEvent | None = None
        self.history: list[EmergencyEvent] = []
        self._timeout_tasks: dict[str, asyncio.Task] = {}

    def save_reading(self, reading: Reading) -> None:
        self.latest_reading = reading

    def save_event(self, event: EmergencyEvent) -> None:
        self.active_event = event

    def archive_active_event(self) -> None:
        if self.active_event:
            self.history.insert(0, deepcopy(self.active_event))
            self.history = self.history[:20]
            self.active_event = None
        self.cancel_timeout_task_for_all()

    def get_by_public_token(self, public_token: str) -> EmergencyEvent | None:
        if self.active_event and self.active_event.public_token == public_token:
            return self.active_event
        for event in self.history:
            if event.public_token == public_token:
                return event
        return None

    # --- Deadline timer bookkeeping -------------------------------------
    #
    # Avoids scheduling more than one pending timeout task per event id.
    # A durable job queue (e.g. Celery, RQ, cloud scheduler) would replace
    # this in a production deployment.

    def register_timeout_task(self, event_id: str, task: asyncio.Task) -> None:
        self.cancel_timeout_task(event_id)
        self._timeout_tasks[event_id] = task

    def cancel_timeout_task(self, event_id: str) -> None:
        existing = self._timeout_tasks.pop(event_id, None)
        if existing and not existing.done():
            existing.cancel()

    def cancel_timeout_task_for_all(self) -> None:
        for event_id in list(self._timeout_tasks.keys()):
            self.cancel_timeout_task(event_id)

    def reset(self) -> None:
        self.latest_reading = None
        self.active_event = None
        self.history = []
        self.cancel_timeout_task_for_all()


store = InMemoryStore()
