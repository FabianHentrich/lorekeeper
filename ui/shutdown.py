"""The "LoreKeeper beenden" and "neu starten" buttons at the bottom of the sidebar on every page (admin only).

Stopping goes through ``stop.ps1``, the same path as from a terminal: it sends Ctrl+C to the backend
and UI windows (a clean shutdown — the backend stops the Ollama it started) and kills whatever still
runs after its timeout. The UI launches the script itself instead of calling a backend endpoint: UI
server and backend always run on the same PC (the backend only listens on 127.0.0.1), so the stop
stays behind the admin gate without a new API route. The script runs detached and outlives this
process, which it stops as well; its output goes to ``logs/stop.log``. A restart is ``stop.ps1 -Restart``:
the same stop, then ``start.ps1`` in the mode that ran — it also frees the VRAM (Ollama is stopped), the
way out when the model landed in shared GPU memory (roadmap 16).

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


def _launch_stop_script(restart: bool = False) -> None:
    """Start stop.ps1 (with ``-Restart`` when asked) detached: no window, own process group, so its Ctrl+C
    does not hit this process directly and the script survives when this process ends. The process group
    switches Ctrl+C off for the script and its children — stop.ps1 switches it back on before a restart."""
    STOP_LOG.parent.mkdir(parents=True, exist_ok=True)
    args = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(STOP_SCRIPT)]
    if restart:
        args.append("-Restart")
    # The child keeps its own handle — ours can be closed right after the start
    with STOP_LOG.open("a", encoding="utf-8") as out:
        subprocess.Popen(
            args,
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )


def render_stop_button() -> None:
    """Stop and restart buttons with one shared confirmation step; both refused while an ingest runs (a
    kill mid-run can leave a half-written index — stop.ps1 refuses too, ``-Force`` overrides it there)."""
    if not is_admin():
        return
    st.divider()
    action = st.session_state.get("_confirm_power")  # "stop" | "restart" while the question is open
    if not action:
        if st.button("⏻ LoreKeeper beenden", key="_stop_ask",
                     help="Beendet Backend und UI sauber (wie Strg+C in beiden Fenstern)."):
            st.session_state["_confirm_power"] = "stop"
            st.rerun()
        if st.button("🔄 LoreKeeper neu starten", key="_restart_ask",
                     help="Beendet Backend und UI sauber und startet sie im selben Modus neu — "
                          "räumt dabei auch den Grafikspeicher (Ollama wird mit beendet)."):
            st.session_state["_confirm_power"] = "restart"
            st.rerun()
        return

    restart = action == "restart"
    if restart:
        st.warning("Backend und UI starten neu — alle Nutzer verlieren für ~30 s die Verbindung.")
    else:
        st.warning("Backend und UI werden beendet — alle Nutzer verlieren die Verbindung. "
                   "Neu starten mit `.\\start.ps1`.")
    yes, no = st.columns(2)
    if no.button("Abbrechen", key="_stop_no"):
        st.session_state.pop("_confirm_power", None)
        st.rerun()
    if yes.button("Neu starten" if restart else "Beenden", key="_stop_yes", type="primary"):
        st.session_state.pop("_confirm_power", None)
        if _ingest_running():
            st.error("Eine Indizierung läuft — erst abwarten (abbrechen nur mit `.\\stop.ps1 -Force`).")
            return
        _launch_stop_script(restart)
        if restart:
            st.success("LoreKeeper startet neu … die Seite verbindet sich danach von selbst wieder.")
        else:
            st.success("LoreKeeper wird beendet …")
