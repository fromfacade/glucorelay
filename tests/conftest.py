import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import store


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Resets the in-memory store and keeps every test hermetic.

    Deleting the Gemma env vars means `interpret_voice_checkin` always uses
    the deterministic fallback parser unless a test explicitly stubs it -
    no test needs a real API key or network access.
    """
    store.reset()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMMA_MODEL", raising=False)
    monkeypatch.setenv("ENABLE_SMS", "false")
    yield
    store.reset()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
