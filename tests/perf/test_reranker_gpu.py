"""DirectML reranker vs. torch on the CPU (roadmap 5.4) — real model, real GPU.

Skipped unless onnxruntime-directml is installed. Checks that the GPU path ranks exactly
like today's CPU path and is clearly faster (measured: ~50 ms vs. ~840 ms for 15 pairs).
"""
import statistics
import time

import pytest

pytestmark = pytest.mark.perf

MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
QUERY = "Wie funktioniert die Section Expansion und wann wird eine ganze Sektion an das LLM gegeben?"
PASSAGES = [
    "Section Expansion gibt dem LLM die ganze Sektion, wenn mindestens zwei Treffer unter derselben Überschrift liegen.",
    "Der Spieler-Modus beschränkt die Suche auf Quellen mit player_visible.",
    "Arkenfeld ist ein Zeitmagier der Akademie und hütet das Archiv der Stunden.",
    "Die Bonusaktion darf einmal pro Runde zusätzlich zur Hauptaktion eingesetzt werden.",
    "Größere Sektionen als max_chunks werden nicht expandiert, damit der Kontext klein bleibt.",
    "Der Reranker sortiert die Kandidaten der Vektorsuche nach Relevanz zur Frage.",
] * 3  # 18 pairs ≈ one production rerank (top_k 15)


def _median_ms(fn, reps=15):
    fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def test_gpu_ranks_like_cpu_and_is_faster():
    onnxruntime = pytest.importorskip("onnxruntime")
    if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
        pytest.skip("onnxruntime-directml not installed")
    from sentence_transformers import CrossEncoder

    from src.retrieval.onnx_reranker import DirectMLCrossEncoder

    pairs = [[QUERY, p] for p in PASSAGES]
    cpu = CrossEncoder(MODEL, max_length=512)
    gpu = DirectMLCrossEncoder(MODEL, max_length=512)

    cpu_scores = [float(s) for s in cpu.predict(pairs, show_progress_bar=False)]
    gpu_scores = gpu.predict(pairs)
    assert max(abs(a - b) for a, b in zip(cpu_scores, gpu_scores, strict=True)) < 1e-3
    assert sorted(range(len(pairs)), key=lambda i: -gpu_scores[i]) == \
        sorted(range(len(pairs)), key=lambda i: -cpu_scores[i])

    cpu_ms = _median_ms(lambda: cpu.predict(pairs, show_progress_bar=False))
    gpu_ms = _median_ms(lambda: gpu.predict(pairs))
    print(f"rerank {len(pairs)} pairs: CPU {cpu_ms:.0f} ms, GPU {gpu_ms:.0f} ms")
    assert gpu_ms < cpu_ms / 3, (cpu_ms, gpu_ms)
