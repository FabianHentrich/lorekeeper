"""The "LoreKeeper beenden" button at the bottom of the sidebar on every page (admin only).

Stopping goes through ``stop.ps1``, the same path as from a terminal: it sends Ctrl+C to the backend
and UI windows (a clean shutdown — the backend stops the Ollama it started) and kills whatever still
runs after its timeout. The UI launches the script itself instead of calling a backend endpoint: UI
server and backend always run on the same PC (the backend only listens on 127.0.0.1), so the stop
stays behind the admin gate without a new API route. The script runs detached and outlives this
process, which it stops as well; its output goes to ``logs/stop.log``.

Callers place it in the sidebar themselves — the chat's sidebar is a fragment, and ``st.sidebar``
cannot be opened from inside one.
"""
import subprocess
from pathlib import Path

import streamlit as st
from admin_gate import api_url, is_admin
from backend import get_json

ROOT = Path(__file__).resolve().parents[1]
STOP_SCRIPT = ROOT / "stop.ps1"
STOP_LOG = ROOT / "logs" / "stop.log"


def _ingest_running() -> bool:
    """True while an ingest job runs; an unreachable backend has nothing to protect (False)."""
    try:
        return get_json(f"{api_url()}/ingest/active") is not None
    except Exception:
        return False


def _launch_stop_script() -> None:
    """Start stop.ps1 detached: no window, own process group, so its Ctrl+C does not hit this process
    directly and the script survives when this process ends."""
    STOP_LOG.parent.mkdir(parents=True, exist_ok=True)
    # The child keeps its own handle — ours can be closed right after the start
    with STOP_LOG.open("a", encoding="utf-8") as out:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(STOP_SCRIPT)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )


def render_stop_button() -> None:
    """Stop button with a confirmation step; refused while an ingest runs (a kill mid-run can leave a
    half-written index — stop.ps1 refuses too, ``-Force`` overrides it there)."""
    if not is_admin():
        return
    st.divider()
    if not st.session_state.get("_confirm_stop"):
        if st.button("⏻ LoreKeeper beenden", key="_stop_ask",
                     help="Beendet Backend und UI sauber (wie Strg+C in beiden Fenstern)."):
            st.session_state["_confirm_stop"] = True
            st.rerun()
        return

    st.warning("Backend und UI werden beendet — alle Nutzer verlieren die Verbindung. "
               "Neu starten mit `.\\start.ps1`.")
    yes, no = st.columns(2)
    if no.button("Abbrechen", key="_stop_no"):
        st.session_state.pop("_confirm_stop", None)
        st.rerun()
    if yes.button("Beenden", key="_stop_yes", type="primary"):
        st.session_state.pop("_confirm_stop", None)
        if _ingest_running():
            st.error("Eine Indizierung läuft — erst abwarten (abbrechen nur mit `.\\stop.ps1 -Force`).")
            return
        _launch_stop_script()
        st.success("LoreKeeper wird beendet …")
