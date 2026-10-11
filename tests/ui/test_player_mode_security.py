"""Security tests for the UI side of player mode and the admin login (roadmap 3.11, 3.7).

The retrieval side (filters can't leak game-master chunks) is covered in
tests/integration/test_player_mode_security.py. Here: what the Streamlit process
sends for a player, how it behaves with tampered or stale state and when the
backend fails (it must fail closed), and the brute-force protection of the login.
"""
import admin_gate
import pytest
import roles

from tests.ui import conftest
from tests.ui.conftest import PAGES, open_page

PLAYER_FILTER = {"source_id": {"$in": ["regelbuch"]}}


def _ask(at, question="Was ist eine Marke?"):
    at.chat_input[0].set_value(question).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


# ─── Tampered / leftover session state ─────────────────────────────────────────

@pytest.mark.parametrize("role", ["admin", "", None, "GAME_MASTER"])
def test_unknown_role_value_falls_back_to_picker(role, fake_backend):
    """Anything but a known role shows the picker — no chat, no backend calls."""
    at = open_page("chat", role=None)
    at.session_state["_role"] = role
    at.run()

    assert len(at.chat_input) == 0
    assert fake_backend.calls == []


def test_leftover_game_master_filters_cannot_widen_player_search(fake_backend):
    """Filters a game master set earlier in the session are ignored for players — not merged, not blocking."""
    at = open_page("chat", role=roles.PLAYER)
    at.session_state["_category_filter"] = {"$in": ["lore", "adventure"]}
    at.session_state["_content_category_filter"] = {"$in": ["npc"]}
    _ask(at.run())

    assert fake_backend.bodies["POST /query/stream"][-1]["metadata_filters"] == PLAYER_FILTER


def test_blocked_game_master_filter_does_not_block_players(fake_backend):
    at = open_page("chat", role=roles.PLAYER)
    at.session_state["_category_filter"] = "__BLOCKED__"
    _ask(at.run())

    assert fake_backend.bodies["POST /query/stream"][-1]["metadata_filters"] == PLAYER_FILTER


def test_every_player_question_carries_the_filter(fake_backend):
    """Follow-ups in the same session are filtered too — not just the first question."""
    at = open_page("chat", role=roles.PLAYER).run()
    for q in ("Was ist eine Marke?", "Und wie viele kann ich haben?", "Was ist Überschaden?"):
        _ask(at, q)

    bodies = fake_backend.bodies["POST /query/stream"]
    assert len(bodies) == 3
    assert all(b["metadata_filters"] == PLAYER_FILTER for b in bodies)


# ─── Fail closed ───────────────────────────────────────────────────────────────

def test_backend_down_locks_player_chat(fake_backend):
    """Without the list of released sources a player can't ask — never an unfiltered fallback."""
    fake_backend.fail_paths.add("/sidebar-state")

    at = open_page("chat", role=roles.PLAYER).run()

    assert not at.exception
    assert at.chat_input[0].disabled
    assert "POST /query/stream" not in fake_backend.calls


def test_missing_player_list_in_response_locks_player_chat(fake_backend, monkeypatch):
    """An older backend without player_source_ids also fails closed."""
    state = {k: v for k, v in conftest.PAYLOADS["/sidebar-state"].items() if k != "player_source_ids"}
    monkeypatch.setitem(conftest.PAYLOADS, "/sidebar-state", state)

    at = open_page("chat", role=roles.PLAYER).run()

    assert at.chat_input[0].disabled


def test_revoking_a_source_takes_effect_on_the_next_question(fake_backend, monkeypatch):
    """Saving on the Sources page clears the cached release list for all sessions (was up to 30 s stale)."""
    player = open_page("chat", role=roles.PLAYER).run()
    assert not player.chat_input[0].disabled            # list cached with "regelbuch"

    monkeypatch.setitem(conftest.PAYLOADS["/sidebar-state"], "player_source_ids", [])
    sources = open_page("sources").run()
    next(b for b in sources.button if b.label == "💾 Änderungen speichern").click().run()
    assert "PUT /sources" in fake_backend.calls

    player.run()
    assert player.chat_input[0].disabled


# ─── Admin login: brute-force protection ───────────────────────────────────────

@pytest.fixture
def password(monkeypatch):
    monkeypatch.setenv("LOREKEEPER_ADMIN_PASSWORD", "geheim")
    return "geheim"


def _try_login(at, entered):
    at.text_input[0].input(entered)
    at.button[0].click()
    return at.run()


def _settings_page():
    return open_page("settings").run()


def test_lockout_after_too_many_wrong_passwords(password, fake_backend):
    """After MAX_FAILURES wrong attempts even the correct password is refused."""
    at = _settings_page()
    for _ in range(admin_gate.MAX_FAILURES):
        _try_login(at, "falsch")

    _try_login(at, password)

    assert at.title[0].value == "🔒 Admin-Bereich"
    assert any("Zu viele Fehlversuche" in e.value for e in at.error)
    assert fake_backend.calls == []


def test_lockout_applies_to_new_sessions(password, fake_backend):
    """Opening a fresh browser session doesn't reset the counter — it is server-wide."""
    attacker = _settings_page()
    for _ in range(admin_gate.MAX_FAILURES):
        _try_login(attacker, "falsch")

    other = _try_login(_settings_page(), password)

    assert other.title[0].value == "🔒 Admin-Bereich"


def test_lockout_expires(password, fake_backend, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(admin_gate, "_now", lambda: clock[0])
    at = _settings_page()
    for _ in range(admin_gate.MAX_FAILURES):
        _try_login(at, "falsch")

    clock[0] += admin_gate.LOCKOUT_S + 1
    _try_login(at, password)

    assert at.title[0].value != "🔒 Admin-Bereich"


def test_logged_in_session_is_not_locked_out(password, fake_backend):
    """A game master who is already logged in keeps working while someone guesses elsewhere."""
    gm = _try_login(_settings_page(), password)
    attacker = _settings_page()
    for _ in range(admin_gate.MAX_FAILURES):
        _try_login(attacker, "falsch")

    gm.run()

    assert gm.title[0].value != "🔒 Admin-Bereich"


def test_successful_login_resets_the_counter(password, fake_backend):
    at = _settings_page()
    for _ in range(admin_gate.MAX_FAILURES - 1):
        _try_login(at, "falsch")
    _try_login(at, password)

    assert admin_gate.lockout_remaining() == 0
    assert admin_gate._failures == []


def test_check_password_itself_enforces_the_lockout(password):
    """Not only the login form: the check refuses the correct password while locked."""
    for _ in range(admin_gate.MAX_FAILURES):
        assert admin_gate.check_password("falsch") is False

    assert admin_gate.check_password(password) is False


def test_wrong_password_waits(password, fake_backend, monkeypatch):
    """Each wrong attempt is slowed down (the delay is disabled in the other tests)."""
    slept = []
    monkeypatch.setattr(admin_gate, "FAILURE_DELAY_S", 1.0)
    monkeypatch.setattr(admin_gate.time, "sleep", slept.append)

    assert admin_gate.check_password("falsch") is False
    assert slept == [1.0]


def test_chat_sidebar_login_shares_the_lockout(password, fake_backend):
    """The second login form (chat sidebar) is no way around the limit."""
    for _ in range(admin_gate.MAX_FAILURES):
        admin_gate.check_password("falsch")

    at = open_page("chat").run()
    form_input = next(t for t in at.sidebar.text_input if t.label == "Passwort")
    form_input.input(password)
    next(b for b in at.sidebar.button if b.label == "Anmelden").click().run()

    assert "_is_admin" not in at.session_state or not at.session_state["_is_admin"]
    assert [s for s in at.selectbox if s.label == "Provider"] == []


def test_pages_are_listed_in_conftest():
    """Guard: every admin page is covered by the gate tests."""
    assert {"sources", "evaluation", "prompts", "settings"} <= set(PAGES)
