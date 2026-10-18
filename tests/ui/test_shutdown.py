"""The "LoreKeeper beenden" / "neu starten" buttons (ui/shutdown.py): admin only, confirmation step, refused
during an ingest.

stop.ps1 itself is never started here — ``_launch_stop_script`` is replaced by a recorder; two tests check
the command it would run with ``subprocess.Popen`` mocked.
"""
import subprocess

import pytest
import shutdown

from tests.ui.conftest import PAGES, PAYLOADS, open_page

ASK, RESTART, YES, NO = "_stop_ask", "_restart_ask", "_stop_yes", "_stop_no"


@pytest.fixture
def launches(monkeypatch):
    """Record stop-script launches instead of running stop.ps1; no ingest is running by default."""
    calls: list[str] = []
    monkeypatch.setattr(shutdown, "_launch_stop_script",
                        lambda restart=False: calls.append("restart" if restart else "stop"))
    monkeypatch.setitem(PAYLOADS, "/ingest/active", None)
    return calls


def _keys(at) -> set[str]:
    return {b.key for b in at.button}


def _confirm(at, ask=ASK):
    at.button(key=ask).click().run()
    return at.button(key=YES).click().run()


@pytest.mark.parametrize("page", list(PAGES))
def test_every_page_has_the_stop_and_restart_buttons(page, fake_backend, launches):
    at = open_page(page).run()

    assert not at.exception
    assert {ASK, RESTART} <= _keys(at)


def test_guest_on_the_chat_sees_no_stop_or_restart_button(monkeypatch, fake_backend, launches):
    """With a password, a visitor in the LAN who is not logged in must not be able to stop the server."""
    monkeypatch.setenv("LOREKEEPER_ADMIN_PASSWORD", "geheim")
    at = open_page("chat").run()

    assert not at.exception
    assert not {ASK, RESTART} & _keys(at)


def test_logged_in_admin_sees_the_stop_button(monkeypatch, fake_backend, launches):
    monkeypatch.setenv("LOREKEEPER_ADMIN_PASSWORD", "geheim")
    at = open_page("chat")
    at.session_state["_is_admin"] = True
    at.run()

    assert ASK in _keys(at)


def test_first_click_only_asks(fake_backend, launches):
    at = open_page("settings").run()
    at.button(key=ASK).click().run()

    assert launches == []
    assert {YES, NO} <= _keys(at)
    assert any("alle Nutzer verlieren die Verbindung" in w.value for w in at.warning)


def test_confirm_launches_the_stop_script(fake_backend, launches):
    at = _confirm(open_page("settings").run())

    assert launches == ["stop"]
    assert "GET /ingest/active" in fake_backend.calls
    assert any("wird beendet" in s.value for s in at.success)


def test_confirm_in_the_chat_sidebar_launches_the_stop_script(fake_backend, launches):
    """The chat sidebar is a fragment — the confirmation step must survive its reruns."""
    at = _confirm(open_page("chat").run())

    assert not at.exception
    assert launches == ["stop"]


def test_cancel_does_not_stop(fake_backend, launches):
    at = open_page("settings").run()
    at.button(key=ASK).click().run()
    at.button(key=NO).click().run()

    assert launches == []
    assert ASK in _keys(at)


@pytest.mark.parametrize("ask", [ASK, RESTART])
def test_refused_while_an_ingest_runs(ask, monkeypatch, fake_backend, launches):
    """A kill mid-ingest can leave a half-written index — a restart kills just the same."""
    monkeypatch.setitem(PAYLOADS, "/ingest/active", {"job_id": "j1", "status": "running"})
    at = _confirm(open_page("settings").run(), ask)

    assert launches == []
    assert any("Indizierung läuft" in e.value for e in at.error)


def test_restart_asks_first_then_launches_a_restart(fake_backend, launches):
    at = open_page("settings").run()
    at.button(key=RESTART).click().run()

    assert launches == []
    assert any("starten neu" in w.value for w in at.warning)
    at.button(key=YES).click().run()
    assert launches == ["restart"]
    assert any("startet neu" in s.value for s in at.success)


def test_restart_in_the_chat_sidebar(fake_backend, launches):
    """The chat sidebar is a fragment — the restart question must survive its reruns like the stop one."""
    at = _confirm(open_page("chat").run(), RESTART)

    assert not at.exception
    assert launches == ["restart"]


def test_unreachable_backend_still_stops(fake_backend, launches):
    """Nothing to protect when the backend is down — the UI (and whatever hangs) is still stopped."""
    fake_backend.fail_paths.add("/ingest/active")
    _confirm(open_page("settings").run())

    assert launches == ["stop"]


def test_launch_runs_stop_ps1_detached(monkeypatch, tmp_path):
    """Detached and without a window: the script must outlive the UI process it stops."""
    popen: list[tuple] = []
    monkeypatch.setattr(shutdown.subprocess, "Popen", lambda *a, **kw: popen.append((a, kw)))
    monkeypatch.setattr(shutdown, "STOP_LOG", tmp_path / "logs" / "stop.log")

    shutdown._launch_stop_script()

    (args,), kwargs = popen[0]
    assert args[0] == "powershell.exe"
    assert args[-2:] == ["-File", str(shutdown.STOP_SCRIPT)]
    assert shutdown.STOP_SCRIPT.is_file()
    assert kwargs["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert (tmp_path / "logs" / "stop.log").exists()


def test_restart_launch_adds_the_restart_switch(monkeypatch, tmp_path):
    popen: list[tuple] = []
    monkeypatch.setattr(shutdown.subprocess, "Popen", lambda *a, **kw: popen.append((a, kw)))
    monkeypatch.setattr(shutdown, "STOP_LOG", tmp_path / "logs" / "stop.log")

    shutdown._launch_stop_script(restart=True)

    (args,), _ = popen[0]
    assert args[-3:] == ["-File", str(shutdown.STOP_SCRIPT), "-Restart"]
