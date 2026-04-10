import os

# Suppress noisy HF/safetensors console output. Must be set before any HF import.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("SAFETENSORS_LOG_LEVEL", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
