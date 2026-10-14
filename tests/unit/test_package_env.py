"""src/__init__.py: environment defaults that must be in place before the first Hugging Face import."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _offline_value(env_value):
    """HF_HUB_OFFLINE as a fresh interpreter sees it after ``import src`` (fresh: setdefault runs once)."""
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    if env_value is not None:
        env["HF_HUB_OFFLINE"] = env_value
    out = subprocess.run([sys.executable, "-c", "import os, src; print(os.environ['HF_HUB_OFFLINE'])"],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_models_load_offline_by_default():
    assert _offline_value(None) == "1"


def test_environment_can_switch_offline_mode_off():
    """HF_HUB_OFFLINE=0 is the documented way to download a new model — it must win over the default."""
    assert _offline_value("0") == "0"
