"""Access protection for the home-network setup (roadmap 3.7/4.1).

With LOREKEEPER_ADMIN_PASSWORD set, admin pages and the provider/Gemini-key controls
require a login per browser session; the chat stays usable. Without a password
everything is open (local single-user mode) — covered by test_backend_calls.py.
"""
import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import PAGES, open_page

ADMIN_PAGES = ["sources", "evaluation", "prompts", "settings"]


@pytest.fixture
def password(monkeypatch):
    monkeypatch.setenv("LOREKEEPER_ADMIN_PASSWORD", "geheim")
    return "geheim"


def _login(at: AppTest, entered: str) -> AppTest:
    at.text_input[0].input(entered)
    at.button[0].click()
    return at.run()


@pytest.mark.parametrize("page", ADMIN_PAGES)
def test_admin_page_shows_login_and_loads_nothing(page, password, fake_backend):
    """A locked admin page renders only the login form and makes no backend calls."""
    at = AppTest.from_file(str(PAGES[page]), default_timeout=60).run()

    assert not at.exception
    assert at.title[0].value == "🔒 Admin-Bereich"
    assert fake_backend.calls == []


def test_correct_password_unlocks_page(password, fake_backend):
    at = _login(AppTest.from_file(str(PAGES["settings"]), default_timeout=60).run(), password)

    assert not at.exception
    assert at.title[0].value != "🔒 Admin-Bereich"
    assert "GET /config" in fake_backend.calls


def test_wrong_password_stays_locked(password, fake_backend):
    at = _login(AppTest.from_file(str(PAGES["settings"]), default_timeout=60).run(), "falsch")

    assert at.title[0].value == "🔒 Admin-Bereich"
    assert any("Falsches Passwort" in e.value for e in at.error)
    assert fake_backend.calls == []


def test_chat_stays_open_but_hides_provider_controls(password, fake_backend):
    """Guests can ask questions; switching to Gemini (costs money) or setting a key needs the login."""
    at = open_page("chat").run()

    assert not at.exception
    assert not [s for s in at.selectbox if s.label == "Provider"]
    at.chat_input[0].set_value("Was ist eine Marke?").run()
    assert "POST /query/stream" in fake_backend.calls


def test_api_url_is_not_editable_in_the_browser(fake_backend):
    """The backend URL comes from LOREKEEPER_API_URL — no sidebar field to point the server elsewhere."""
    at = open_page("chat").run()

    assert not [t for t in at.sidebar.text_input if t.label == "API URL"]


def test_api_url_from_environment(monkeypatch, fake_backend):
    monkeypatch.setenv("LOREKEEPER_API_URL", "http://127.0.0.1:9000/")
    import admin_gate

    assert admin_gate.api_url() == "http://127.0.0.1:9000"


def test_failed_logins_and_the_lock_are_logged(monkeypatch, caplog):
    """Guessing the password in the home network must leave a trace (Streamlit console)."""
    import admin_gate
    monkeypatch.setenv("LOREKEEPER_ADMIN_PASSWORD", "geheim")
    for _ in range(admin_gate.MAX_FAILURES):
        assert not admin_gate.check_password("falsch")

    warnings = [r.getMessage() for r in caplog.records if r.name == "admin_gate"]
    assert warnings[0] == f"Admin login failed (1 of {admin_gate.MAX_FAILURES} before the lock)"
    assert warnings[-1].startswith(f"Admin login failed — {admin_gate.MAX_FAILURES} failures, login locked")
