"""LoreKeeper backend package.

Importing it sets environment defaults that keep the Hugging Face / safetensors stack quiet. They
only work when set before the first HF import — which is why they live here and not in main.py.
"""
import os

# Suppress noisy HF/safetensors console output. Must be set before any HF import.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("SAFETENSORS_LOG_LEVEL", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
