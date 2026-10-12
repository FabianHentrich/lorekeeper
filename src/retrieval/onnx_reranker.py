"""Cross-encoder reranker on ONNX Runtime + DirectML — runs on any DirectX 12 GPU, including AMD.

Same model and tokenizer as the torch ``CrossEncoder``, so the scores match to rounding
(raw logits, no activation — like ``CrossEncoder.predict`` for this model). Measured on the
RX 9070 XT: 15 pairs in ~50 ms instead of ~840 ms on the CPU, and ~70 ms instead of ~14 s
while the CPU is fully loaded (roadmap 5.4).
"""
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

DML_PROVIDER = "DmlExecutionProvider"


class DirectMLUnavailable(RuntimeError):
    """DirectML can't run the model — the caller falls back to the torch reranker on the CPU."""


class DirectMLCrossEncoder:
    """Drop-in for ``sentence_transformers.CrossEncoder.predict`` on the GPU via DirectML."""

    def __init__(self, model_name: str, max_length: int = 512):
        """Load the model's published ONNX export and tokenizer from the Hugging Face cache.

        Raises ``DirectMLUnavailable`` when the DirectML provider is missing — e.g. the plain
        ``onnxruntime`` package was installed over ``onnxruntime-directml`` — or the model has
        no ONNX export.
        """
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        if DML_PROVIDER not in ort.get_available_providers():
            raise DirectMLUnavailable(
                f"{DML_PROVIDER} missing (available: {ort.get_available_providers()}) — "
                "is onnxruntime-directml installed? See docs/configuration.md → reranking.backend"
            )
        try:
            model_path = hf_hub_download(model_name, "onnx/model.onnx")
            tokenizer_path = hf_hub_download(model_name, "tokenizer.json")
        except Exception as e:
            raise DirectMLUnavailable(f"no ONNX export for {model_name}: {e}") from e

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length)
        self._tokenizer.enable_padding()

        options = ort.SessionOptions()
        options.enable_mem_pattern = False  # required by the DirectML provider
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(model_path, options, providers=[DML_PROVIDER, "CPUExecutionProvider"])
        if self._session.get_providers()[0] != DML_PROVIDER:
            raise DirectMLUnavailable(f"session fell back to {self._session.get_providers()}")
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._lock = threading.Lock()  # one GPU queue — concurrent requests run one after another

    def predict(self, pairs: list[list[str]]) -> list[float]:
        """Relevance score per (query, passage) pair, higher = more relevant."""
        if not pairs:
            return []
        encoded = self._tokenizer.encode_batch([(q, p) for q, p in pairs])
        feed = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
        }
        with self._lock:
            logits = self._session.run(None, {k: v for k, v in feed.items() if k in self._input_names})[0]
        return [float(s) for s in np.asarray(logits).reshape(-1)]
