"""ONNX reranker: refuses the GPU without DirectML, runs on the CPU on request, feeds the session correctly."""
from types import SimpleNamespace

import numpy as np
import onnxruntime
import pytest

from src.retrieval import onnx_reranker
from src.retrieval.onnx_reranker import CPU_PROVIDER, DML_PROVIDER, DirectMLUnavailable, OnnxCrossEncoder


class FakeTokenizer:
    def encode_batch(self, pairs):
        self.pairs = pairs
        return [SimpleNamespace(ids=[1, 2, 3], attention_mask=[1, 1, 1], type_ids=[0, 0, 1]) for _ in pairs]


class FakeSession:
    def __init__(self, input_names=("input_ids", "attention_mask"), providers=(DML_PROVIDER, "CPUExecutionProvider")):
        self._inputs = [SimpleNamespace(name=n) for n in input_names]
        self._providers = list(providers)

    def get_inputs(self):
        return self._inputs

    def get_providers(self):
        return self._providers

    def run(self, _outputs, feed):
        self.feed = feed
        return [np.array([[0.5 + i] for i in range(len(feed["input_ids"]))], dtype=np.float32)]


class FakeLoadedTokenizer:
    """What Tokenizer.from_file returns — records the padding setup; vocabulary with or without <pad>."""
    def __init__(self, vocab=None):
        self.vocab = {"<s>": 0, "<pad>": 1} if vocab is None else vocab
        self.padding = None

    def token_to_id(self, token):
        return self.vocab.get(token)

    def enable_truncation(self, n):
        pass

    def enable_padding(self, **kwargs):
        self.padding = kwargs


def _load(monkeypatch, *, providers=(DML_PROVIDER, CPU_PROVIDER), session_providers=None, tokenizer=None,
          download=None):
    """Patch onnxruntime / hub / tokenizers; returns the dict that records the InferenceSession call."""
    import huggingface_hub
    import tokenizers

    created = {}

    def _session(path, options, providers):
        created.update(path=path, providers=providers)
        return FakeSession(providers=session_providers or providers)

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: list(providers))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download or (lambda repo, name: f"/cache/{name}"))
    tok = tokenizer or FakeLoadedTokenizer()
    monkeypatch.setattr(tokenizers.Tokenizer, "from_file", staticmethod(lambda path: tok))
    monkeypatch.setattr(onnxruntime, "InferenceSession", _session)
    created["tokenizer"] = tok
    return created


def _encoder(session):
    """An OnnxCrossEncoder without loading anything — for predict()."""
    import threading
    enc = OnnxCrossEncoder.__new__(OnnxCrossEncoder)
    enc._tokenizer = FakeTokenizer()
    enc._session = session
    enc._input_names = {i.name for i in session.get_inputs()}
    enc._lock = threading.Lock()
    return enc


def test_missing_directml_provider_refuses(monkeypatch):
    """The plain onnxruntime package has no DirectML — loading must fail loudly, not run on the CPU."""
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(DirectMLUnavailable, match="onnxruntime-directml"):
        OnnxCrossEncoder("some/model")


def test_session_falling_back_to_cpu_refuses(monkeypatch):
    """Provider listed, but the session ends up on the CPU (e.g. no DX12 device) → refuse."""
    import huggingface_hub
    import tokenizers

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: [DML_PROVIDER, "CPUExecutionProvider"])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: f"/cache/{name}")
    monkeypatch.setattr(tokenizers.Tokenizer, "from_file", staticmethod(lambda path: FakeLoadedTokenizer()))
    monkeypatch.setattr(onnxruntime, "InferenceSession",
                        lambda *a, **k: FakeSession(providers=["CPUExecutionProvider"]))
    with pytest.raises(DirectMLUnavailable, match="fell back"):
        OnnxCrossEncoder("some/model")


def test_missing_onnx_export_refuses(monkeypatch):
    import huggingface_hub

    def _missing(repo, name):
        raise FileNotFoundError(name)

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: [DML_PROVIDER, "CPUExecutionProvider"])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _missing)
    with pytest.raises(DirectMLUnavailable, match="no ONNX export"):
        OnnxCrossEncoder("some/model")


def test_predict_scores_each_pair_in_order():
    session = FakeSession()
    enc = _encoder(session)

    scores = enc.predict([["q", "a"], ["q", "b"], ["q", "c"]])

    assert scores == [0.5, 1.5, 2.5]
    assert enc._tokenizer.pairs == [("q", "a"), ("q", "b"), ("q", "c")]
    assert set(session.feed) == {"input_ids", "attention_mask"}  # only what the model declares
    assert session.feed["input_ids"].dtype == np.int64


def test_predict_passes_token_type_ids_when_the_model_wants_them():
    session = FakeSession(input_names=("input_ids", "attention_mask", "token_type_ids"))
    _encoder(session).predict([["q", "a"]])
    assert session.feed["token_type_ids"].tolist() == [[0, 0, 1]]


def test_predict_empty():
    assert _encoder(FakeSession()).predict([]) == []


def test_module_has_no_torch_import():
    """The GPU path must not pull in torch — that's the point of using ONNX Runtime."""
    assert "torch" not in vars(onnx_reranker)


def test_gpu_loads_the_configured_onnx_file_on_directml(monkeypatch):
    created = _load(monkeypatch)
    OnnxCrossEncoder("some/model", "onnx/model_fp16.onnx", gpu=True)
    assert created["path"] == "/cache/onnx/model_fp16.onnx"
    assert created["providers"][0] == DML_PROVIDER


def test_cpu_runs_without_directml(monkeypatch):
    """The CPU fallback must work exactly when DirectML is missing — and never ask for it."""
    created = _load(monkeypatch, providers=(CPU_PROVIDER,))
    OnnxCrossEncoder("some/model", "onnx/model.onnx", gpu=False)
    assert created["providers"] == [CPU_PROVIDER]


def test_cpu_download_error_is_not_disguised_as_directml(monkeypatch):
    """On the CPU there is no further fallback — a missing file is a real error, not DirectMLUnavailable."""
    def _missing(repo, name):
        raise FileNotFoundError(name)

    _load(monkeypatch, download=_missing)
    with pytest.raises(FileNotFoundError):
        OnnxCrossEncoder("some/model", gpu=False)


def test_pads_with_the_models_pad_token(monkeypatch):
    """tokenizer.json has no padding preset; the default id 0 is <s> in XLM-R vocabularies."""
    created = _load(monkeypatch)
    OnnxCrossEncoder("some/model", gpu=True)
    assert created["tokenizer"].padding == {"pad_id": 1, "pad_token": "<pad>"}


def test_pads_with_the_default_without_a_pad_token(monkeypatch):
    created = _load(monkeypatch, tokenizer=FakeLoadedTokenizer(vocab={}))
    OnnxCrossEncoder("some/model", gpu=True)
    assert created["tokenizer"].padding == {}


def test_predict_splits_large_pools_into_batches(monkeypatch):
    """top_k is editable in the UI — 200 candidates in one run filled the GPU next to the LLM (14.10.)."""
    monkeypatch.setattr(onnx_reranker, "BATCH_SIZE", 4)

    class CountingSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def run(self, _outputs, feed):
            n = len(feed["input_ids"])
            start = sum(self.batch_sizes)
            self.batch_sizes.append(n)
            return [np.array([[float(start + i)] for i in range(n)], dtype=np.float32)]

    session = CountingSession()
    scores = _encoder(session).predict([["q", str(i)] for i in range(10)])

    assert session.batch_sizes == [4, 4, 2]
    assert scores == [float(i) for i in range(10)]  # one score per pair, in input order
