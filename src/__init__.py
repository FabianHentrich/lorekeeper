"""LoreKeeper backend package.

Importing it sets environment defaults for the Hugging Face / safetensors stack: quiet output and offline
model loading. They only work when set before the first HF import — which is why they live here and not
in main.py, and why the backend's ``.env`` (read only into the Pydantic settings) can't set them.
"""
import os

# Suppress noisy HF/safetensors console output. Must be set before any HF import.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("SAFETENSORS_LOG_LEVEL", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Models load from the local cache only — otherwise every start asks Hugging Face for updates, and a slow
# or failing hub (429/500) delays the start. A new or changed model needs one download: HF_HUB_OFFLINE=0
# in the environment, or the preload command in the README (docs/operations.md → "Modelle fehlen").
os.environ.setdefault("HF_HUB_OFFLINE", "1")
