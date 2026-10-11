"""Baseline handling for the timing-based performance checks (roadmap 5.2).

Timings depend on the machine, so the checks compare against a checked-in baseline
with a tolerance instead of absolute limits. To accept new values on purpose (new
hardware, an intended trade-off), run once with ``PERF_UPDATE_BASELINE=1``:

    PERF_UPDATE_BASELINE=1 pytest -m perf
"""
import json
import os
from pathlib import Path

import pytest

BASELINE_FILE = Path(__file__).with_name("baseline.json")
UPDATE = os.environ.get("PERF_UPDATE_BASELINE") == "1"


class Baseline:
    """Read/compare/update one section of ``baseline.json``."""

    def __init__(self):
        self._data = json.loads(BASELINE_FILE.read_text(encoding="utf-8")) if BASELINE_FILE.exists() else {}
        self._dirty = False

    def check(self, section: str, key: str, value: float, *, factor: float = 1.5, slack: float = 0.0,
              higher_is_better: bool = False) -> None:
        """Fail if ``value`` is worse than the baseline beyond ``factor`` (+ absolute ``slack``).

        Without a baseline entry (or in update mode) the value is recorded instead.
        """
        stored = self._data.get(section, {}).get(key)
        if UPDATE or stored is None:
            self._data.setdefault(section, {})[key] = round(value, 3)
            self._dirty = True
            return
        if higher_is_better:
            limit = stored / factor - slack
            assert value >= limit, f"{section}.{key}: {value:.3f} < {limit:.3f} (baseline {stored})"
        else:
            limit = stored * factor + slack
            assert value <= limit, f"{section}.{key}: {value:.3f} > {limit:.3f} (baseline {stored})"

    def save(self) -> None:
        if self._dirty:
            BASELINE_FILE.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def baseline():
    """Session-wide baseline; new or updated values are written back at the end."""
    b = Baseline()
    yield b
    b.save()
