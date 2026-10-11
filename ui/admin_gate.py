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
"""
import hmac
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# 127.0.0.1, not localhost: on Windows localhost resolves to IPv6 first and costs ~2 s per connection
DEFAULT_API_URL = "http://127.0.0.1:8000"


def api_url() -> str:
    """Backend URL for this installation."""
    return os.environ.get("LOREKEEPER_API_URL", DEFAULT_API_URL).rstrip("/")


def _password() -> str:
    return os.environ.get("LOREKEEPER_ADMIN_PASSWORD", "")


def is_admin() -> bool:
    """True when no password is configured, or after a successful login in this browser session."""
    return not _password() or bool(st.session_state.get("_is_admin", False))


def admin_login_form(key: str) -> None:
    """Password form; a correct password unlocks the admin areas for this browser session."""
    with st.form(key, clear_on_submit=True):
        entered = st.text_input("Passwort", type="password")
        if st.form_submit_button("Anmelden"):
            if hmac.compare_digest(entered.encode(), _password().encode()):
                st.session_state["_is_admin"] = True
                st.rerun()
            st.error("Falsches Passwort.")


def require_admin() -> None:
    """Show a login form and stop the page unless this session is admin."""
    if is_admin():
        return
    st.title("🔒 Admin-Bereich")
    st.caption("Diese Seite ist nur für die Spielleitung.")
    admin_login_form("admin_login_page")
    st.stop()
