"""Cross-encoder reranker on ONNX Runtime — on the GPU via DirectML (any DirectX 12 GPU, including AMD),
or on the CPU as the fallback.

Loads the model's published ONNX export and tokenizer from the Hugging Face cache and returns raw logits
(no activation). No torch: the model in use (jina-reranker-v2) only loads in torch with
``trust_remote_code`` and an extra package, ONNX needs neither. Measured on the RX 9070 XT (14.10.,
roadmap 3.10): jina fp16 20 pairs in ~26 ms on the GPU; the fp32 export on the CPU ~2.1 s, scores
within 0.004 of the GPU. The fp16 export does not load on ONNX Runtime's CPU provider, so the CPU always
uses ``CPU_ONNX_FILE``.
"""
import threading

import numpy as np

DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
CPU_ONNX_FILE = "onnx/model.onnx"  # fp32 export — fp16 graphs fail to initialise on the CPU provider


class DirectMLUnavailable(RuntimeError):
    """DirectML can't run the model — the caller falls back to the CPU."""


class OnnxCrossEncoder:
    """Drop-in for ``sentence_transformers.CrossEncoder.predict`` on ONNX Runtime."""

    def __init__(self, model_name: str, onnx_file: str = CPU_ONNX_FILE, max_length: int = 512, gpu: bool = True):
        """Load ``onnx_file`` of ``model_name`` and its tokenizer, on the GPU (DirectML) or the CPU.

        With ``gpu=True`` it raises ``DirectMLUnavailable`` when the DirectML provider is missing — e.g. the
        plain ``onnxruntime`` package was installed over ``onnxruntime-directml`` —, the session ends up on
        the CPU, or the file can't be downloaded. On the CPU, download errors propagate unchanged.
        """
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        if gpu and DML_PROVIDER not in ort.get_available_providers():
            raise DirectMLUnavailable(
                f"{DML_PROVIDER} missing (available: {ort.get_available_providers()}) — "
                "is onnxruntime-directml installed? See docs/configuration.md → reranking.backend"
            )
        try:
            model_path = hf_hub_download(model_name, onnx_file)
            tokenizer_path = hf_hub_download(model_name, "tokenizer.json")
        except Exception as e:
            if not gpu:
                raise
            raise DirectMLUnavailable(f"no ONNX export {onnx_file} for {model_name}: {e}") from e

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length)
        # tokenizer.json has no padding preset; the default pad id 0 is "<s>" in XLM-R vocabularies. The
        # attention mask hides padding either way, the model's own pad token keeps it like the reference.
        pad = next((t for t in ("<pad>", "[PAD]") if self._tokenizer.token_to_id(t) is not None), None)
        if pad:
            self._tokenizer.enable_padding(pad_id=self._tokenizer.token_to_id(pad), pad_token=pad)
        else:
            self._tokenizer.enable_padding()

        options = ort.SessionOptions()
        options.enable_mem_pattern = False  # required by the DirectML provider
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [DML_PROVIDER, CPU_PROVIDER] if gpu else [CPU_PROVIDER]
        self._session = ort.InferenceSession(model_path, options, providers=providers)
        if gpu and self._session.get_providers()[0] != DML_PROVIDER:
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
