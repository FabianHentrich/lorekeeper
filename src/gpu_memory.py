"""How much of the local LLM sits in shared GPU memory (system RAM behind PCIe) — roadmap 7b.

On 17./18.10. ``dwm.exe`` held 7.85 GB of VRAM after another program, and Ollama put 2.65 GB of qwen3:14b into
shared memory: answers ran at ~17 instead of ~54 tok/s while ``ollama ps`` still said "100% GPU" (no layer
on the CPU). Reloading the model doesn't help while the VRAM is taken, so the health loop only reports it.

Read from the Windows performance counter ``\\GPU Process Memory(*)\\Shared Usage`` for Ollama's processes
(``llama-server`` runner and ``ollama``), through one PowerShell call (~1–2 s, run in the threadpool by the
caller) — no extra dependency. Windows only, like LoreKeeper; anything that goes wrong reads as "unknown".
"""
import logging
import subprocess

logger = logging.getLogger(__name__)

WARN_BYTES = 1024 ** 3  # 2.65 GB slowed answers 3×; a few hundred MB of driver bookkeeping are harmless
_TIMEOUT_S = 20
_CREATE_NO_WINDOW = 0x08000000  # no console window flashing up every few minutes
# Sum of "shared usage" over the counter instances of Ollama's processes; prints nothing without a runner
_SCRIPT = (
    "$ids = @(Get-Process llama-server, ollama -ErrorAction SilentlyContinue | ForEach-Object Id); "
    "if (-not $ids) { exit 0 }; $sum = 0; "
    "(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples | ForEach-Object { "
    "if ($_.InstanceName -match 'pid_(\\d+)_' -and $ids -contains [int]$matches[1]) { $sum += $_.CookedValue } }; "
    "[int64]$sum"
)


def llm_shared_bytes() -> int | None:
    """Shared GPU memory of Ollama's processes in bytes; None when unknown (no Ollama process, no counters,
    PowerShell failed or timed out). Blocks ~1–2 s — call it through ``run_in_threadpool``."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _SCRIPT],
            capture_output=True, text=True, timeout=_TIMEOUT_S, creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("GPU memory check failed: %s", e)
        return None
    out = result.stdout.strip()
    if result.returncode != 0 or not out:
        logger.debug("GPU memory check: no value (exit %s, %s)", result.returncode, result.stderr.strip()[:200])
        return None
    try:
        return int(out.splitlines()[-1])
    except ValueError:
        logger.debug("GPU memory check: unexpected output %r", out[:200])
        return None
