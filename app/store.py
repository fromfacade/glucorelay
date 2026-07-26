from copy import deepcopy

from app.models import EmergencyEvent, Reading


class InMemoryStore:
    def __init__(self) -> None:
        self.latest_reading: Reading | None = None
        self.active_event: EmergencyEvent | None = None
        self.history: list[EmergencyEvent] = []

    def save_reading(self, reading: Reading) -> None:
        self.latest_reading = reading

    def save_event(self, event: EmergencyEvent) -> None:
        self.active_event = event

    def archive_active_event(self) -> None:
        if self.active_event:
            self.history.insert(0, deepcopy(self.active_event))
            self.history = self.history[:20]
            self.active_event = None


store = InMemoryStore()
