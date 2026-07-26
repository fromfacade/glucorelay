from dataclasses import dataclass
from enum import Enum

from app.models import Reading


class Decision(str, Enum):
    NONE = "none"
    CHECK_IN = "check_in"
    CONTACT_NOW = "contact_now"


@dataclass(frozen=True)
class DemoThresholds:
    low: int
    urgent_low: int
    high: int


def evaluate_reading(
    reading: Reading,
    thresholds: DemoThresholds,
) -> tuple[Decision, str]:
    # Deterministic rules only. AI must not make this decision.
    if reading.value_mg_dl <= thresholds.urgent_low:
        return (
            Decision.CONTACT_NOW,
            "Demo urgent-low threshold reached",
        )

    if reading.value_mg_dl <= thresholds.low:
        return (
            Decision.CHECK_IN,
            "Demo low threshold reached",
        )

    if reading.value_mg_dl >= thresholds.high:
        return (
            Decision.CHECK_IN,
            "Demo high threshold reached",
        )

    return Decision.NONE, "Reading within configured demo range"
