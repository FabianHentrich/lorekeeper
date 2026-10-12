"""DirectML reranker: refuses to run without the GPU provider, feeds the session correctly."""
from types import SimpleNamespace

import numpy as np
import onnxruntime
import pytest

from src.retrieval import onnx_reranker
from src.retrieval.onnx_reranker import DML_PROVIDER, DirectMLCrossEncoder, DirectMLUnavailable


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


def _encoder(session):
    """A DirectMLCrossEncoder without loading anything — for predict()."""
    import threading
    enc = DirectMLCrossEncoder.__new__(DirectMLCrossEncoder)
    enc._tokenizer = FakeTokenizer()
    enc._session = session
    enc._input_names = {i.name for i in session.get_inputs()}
    enc._lock = threading.Lock()
    return enc


def test_missing_directml_provider_refuses(monkeypatch):
    """The plain onnxruntime package has no DirectML — loading must fail loudly, not run on the CPU."""
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(DirectMLUnavailable, match="onnxruntime-directml"):
        DirectMLCrossEncoder("some/model")


def test_session_falling_back_to_cpu_refuses(monkeypatch):
    """Provider listed, but the session ends up on the CPU (e.g. no DX12 device) → refuse."""
    import huggingface_hub
    import tokenizers

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: [DML_PROVIDER, "CPUExecutionProvider"])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: f"/cache/{name}")
    monkeypatch.setattr(tokenizers.Tokenizer, "from_file", staticmethod(lambda path: SimpleNamespace(
        enable_truncation=lambda n: None, enable_padding=lambda: None)))
    monkeypatch.setattr(onnxruntime, "InferenceSession",
                        lambda *a, **k: FakeSession(providers=["CPUExecutionProvider"]))
    with pytest.raises(DirectMLUnavailable, match="fell back"):
        DirectMLCrossEncoder("some/model")


def test_missing_onnx_export_refuses(monkeypatch):
    import huggingface_hub

    def _missing(repo, name):
        raise FileNotFoundError(name)

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: [DML_PROVIDER, "CPUExecutionProvider"])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _missing)
    with pytest.raises(DirectMLUnavailable, match="no ONNX export"):
        DirectMLCrossEncoder("some/model")


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
