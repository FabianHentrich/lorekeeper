"""Access settings for the Streamlit UI: backend URL and the admin password gate.

The backend stays on 127.0.0.1 — only the UI is reachable in the home network
(roadmap 4.1), so access control happens here (roadmap 3.7):

- ``LOREKEEPER_ADMIN_PASSWORD`` (``.env``) protects the admin pages (Sources,
  Evaluation, Prompts, Settings) and the provider/Gemini-key controls in the chat
  sidebar. The chat itself stays open. Without a password everything is open,
  as before (local single-user setup).
- ``LOREKEEPER_API_URL`` (``.env``, default ``http://127.0.0.1:8000``) replaces the
  former "API URL" sidebar field, which let any visitor point the UI server at
  arbitrary URLs.

Password guessing is slowed down server-wide (not per browser session, which an
attacker could simply renew): every wrong attempt waits ``FAILURE_DELAY_S``, and
``MAX_FAILURES`` wrong attempts within ``LOCKOUT_S`` lock the login for everyone
for ``LOCKOUT_S``. Sessions that are already logged in are not affected.
"""
import hmac
import os
import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# 127.0.0.1, not localhost: on Windows localhost resolves to IPv6 first and costs ~2 s per connection
DEFAULT_API_URL = "http://127.0.0.1:8000"

MAX_FAILURES = 5
LOCKOUT_S = 300.0
FAILURE_DELAY_S = 1.0

# Module state lives as long as the Streamlit server process: shared by all browser sessions
_failures: list[float] = []
_lock = threading.Lock()
_now = time.monotonic


def api_url() -> str:
    """Backend URL for this installation."""
    return os.environ.get("LOREKEEPER_API_URL", DEFAULT_API_URL).rstrip("/")


def _password() -> str:
    """The admin password from the environment; empty = no protection (read per call, not cached)."""
    return os.environ.get("LOREKEEPER_ADMIN_PASSWORD", "")


def is_admin() -> bool:
    """True when no password is configured, or after a successful login in this browser session."""
    return not _password() or bool(st.session_state.get("_is_admin", False))


def lockout_remaining() -> float:
    """Seconds until login attempts are accepted again (0 = not locked)."""
    with _lock:
        now = _now()
        _failures[:] = [t for t in _failures if now - t < LOCKOUT_S]
        if len(_failures) < MAX_FAILURES:
            return 0.0
        return LOCKOUT_S - (now - _failures[-MAX_FAILURES])


def check_password(entered: str) -> bool:
    """Constant-time check with server-wide rate limiting; False while locked out."""
    if lockout_remaining() > 0:
        return False
    if hmac.compare_digest(entered.encode(), _password().encode()):
        with _lock:
            _failures.clear()
        return True
    with _lock:
        _failures.append(_now())
    time.sleep(FAILURE_DELAY_S)
    return False


def admin_login_form(key: str) -> None:
    """Password form; a correct password unlocks the admin areas for this browser session."""
    with st.form(key, clear_on_submit=True):
        entered = st.text_input("Passwort", type="password")
        if st.form_submit_button("Anmelden"):
            if lockout_remaining() > 0:
                st.error(f"Zu viele Fehlversuche — Anmeldung für {lockout_remaining() / 60:.0f} Minuten gesperrt.")
                return
            if check_password(entered):
                st.session_state["_is_admin"] = True
                st.rerun()
            if lockout_remaining() > 0:
                st.error(f"Zu viele Fehlversuche — Anmeldung für {lockout_remaining() / 60:.0f} Minuten gesperrt.")
            else:
                st.error("Falsches Passwort.")


def require_admin() -> None:
    """Show a login form and stop the page unless this session is admin."""
    if is_admin():
        return
    st.title("🔒 Admin-Bereich")
    st.caption("Diese Seite ist nur für die Spielleitung.")
    admin_login_form("admin_login_page")
    st.stop()
