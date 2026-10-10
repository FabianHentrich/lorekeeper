"""End-to-end test: ingest a small vault over HTTP, then query it via SSE.

Everything is real except the LLM: ConfigManager + sources.yaml on tmp files,
Markdown parsing + chunking, sentence-transformer embeddings, ChromaDB
(EphemeralClient, in-memory), retriever, prompt templates, conversation manager
and the Generator. The LLM provider is a fake that streams fixed tokens and
records the prompt it received.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.main as main_module
from src.api import routes as routes_module
from src.api.routes import router
from src.config.manager import (
    ConfigManager,
    EmbeddingsConfig,
    RerankingConfig,
    RetrievalConfig,
    SourceConfig,
    VectorStoreConfig,
)
from src.conversation.manager import ConversationManager
from src.generation.generator import Generator
from src.prompts.manager import PromptManager
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import Retriever
from src.retrieval.vectorstore import VectorStoreService

_REPO = Path(__file__).resolve().parents[2]

_ARKENFELD = (
    "# Arkenfeld\n\n"
    "Arkenfeld ist eine befestigte Stadt am Rand der Akademie. Sie wird von einem Rat "
    "aus sieben Magiern regiert, die sich jeden Neumond im alten Turm versammeln, um "
    "über Handel, Verteidigung und die Pflege der großen Bibliothek zu beraten.\n"
)
_ALDRIC = (
    "# Aldric\n\n"
    "Aldric ist ein alter Schmied, der seit dreißig Jahren am Nordtor arbeitet. Er "
    "fertigt Klingen für die Stadtwache und ist bekannt für seine schlechte Laune "
    "am Morgen und seine Vorliebe für dunkles Bier.\n"
)
_ANSWER_TOKENS = ["Ein ", "Rat ", "aus ", "sieben ", "Magiern."]


class FakeLLM:
    """Streams fixed tokens and records the prompts it was given."""
    provider = "fake"
    model = "fake-llm"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_stream(self, prompt, system_prompt=None, stream_result=None, **kw):
        self.prompts.append(prompt)
        if stream_result is not None:
            stream_result.usage.update({"tokens_in": 10, "tokens_out": len(_ANSWER_TOKENS)})
        for tok in _ANSWER_TOKENS:
            yield tok

    async def generate(self, prompt, **kw):
        raise AssertionError("condense/generate not expected in this test")

    async def health_check(self) -> bool:
        return True


@pytest.fixture(scope="module")
def embed_service():
    """Real embedding model, loaded once (same small model as test_rag_pipeline)."""
    return EmbeddingService(EmbeddingsConfig(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu", batch_size=32, normalize=True,
    ))


@pytest.fixture
def app_env(tmp_path, embed_service):
    """Wire real services into src.main around a tmp vault with two notes."""
    vault = tmp_path / "vault"
    (vault / "NPCs").mkdir(parents=True)
    (vault / "Arkenfeld.md").write_text(_ARKENFELD, encoding="utf-8")
    (vault / "NPCs" / "Aldric.md").write_text(_ALDRIC, encoding="utf-8")

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")
    config = ConfigManager(
        settings_path=settings_path,
        prompts_path=_REPO / "config" / "prompts.yaml",
        sources_path=tmp_path / "sources.yaml",
    )
    config.settings.ingestion.sources = [SourceConfig(id="vault", path=str(vault), group="lore")]

    vectorstore = VectorStoreService(
        VectorStoreConfig(mode="embedded", collection_name=f"e2e_{uuid.uuid4().hex}",
                          distance_metric="cosine"),
        embed_service,
    )
    vectorstore._client = chromadb.EphemeralClient()
    bm25 = BM25Index()
    retriever = Retriever(
        # Threshold 0 keeps the test independent of the small model's absolute scores;
        # ranking is still asserted below.
        RetrievalConfig(top_k=5, score_threshold=0.0, reranking=RerankingConfig(enabled=False)),
        embed_service, vectorstore, bm25,
    )
    llm = FakeLLM()

    services = {
        "config": config,
        "conversation_manager": ConversationManager(config.settings.conversation),
        "prompt_manager": PromptManager(prompts_dict=config.prompts),
        "embedding_service": embed_service,
        "vectorstore": vectorstore,
        "bm25_index": bm25,
        "retriever": retriever,
        "provider": llm,
        "condense_provider": None,
        "generator": Generator(provider=llm),
    }
    originals = {n: getattr(main_module, n) for n in services}
    for n, v in services.items():
        setattr(main_module, n, v)
    routes_module._ingest_jobs.clear()

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, vault=vault, llm=llm, vectorstore=vectorstore)

    for n, v in originals.items():
        setattr(main_module, n, v)
    routes_module._ingest_jobs.clear()


def _ingest(client, **params) -> dict:
    """Start /ingest and poll until the job is finished."""
    r = client.post("/ingest", params=params)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        info = client.get(f"/ingest/status/{job_id}").json()
        if info["status"] not in ("queued", "running"):
            assert info["status"] == "done", info
            assert info["errors"] == []
            return info
        time.sleep(0.1)
    pytest.fail("ingest job did not finish")


def _stream(client, question: str, session_id: str | None = None) -> tuple[str, dict]:
    """POST /query/stream and return (joined token text, done event)."""
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id
    r = client.post("/query/stream", json=body)
    assert r.status_code == 200
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert not [e for e in events if e["type"] == "error"], events
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert events[-1]["type"] == "done"
    return text, events[-1]


def test_ingest_then_query_stream(app_env):
    """Ingest the vault, ask a question, and get a streamed answer with sources and session."""
    client = app_env.client

    info = _ingest(client)
    assert info["documents_processed"] == 2
    assert info["documents_skipped"] == 0
    assert info["chunks_created"] > 0
    assert client.get("/stats").json()["chunk_count"] == info["chunks_created"]

    text, done = _stream(client, "Wer regiert die Stadt Arkenfeld?")
    assert text == "".join(_ANSWER_TOKENS)
    assert done["session_id"]
    assert done["model_used"] == "fake-llm"
    assert done["usage"]["tokens_out"] == len(_ANSWER_TOKENS)
    top = done["sources"][0]
    assert top["file"] == "Arkenfeld.md"
    assert top["document_type"] == "markdown"
    # The retrieved chunk text reached the LLM through the real QA template
    assert "sieben Magiern" in app_env.llm.prompts[-1]

    # Follow-up keeps the session
    _, done2 = _stream(client, "Und wer ist Aldric?", session_id=done["session_id"])
    assert done2["session_id"] == done["session_id"]
    assert done2["session_usage"]["tokens_out"] == 2 * len(_ANSWER_TOKENS)


def test_incremental_reingest_skips_updates_and_deletes(app_env):
    """Second run skips unchanged files; edits are re-embedded, deleted files are removed."""
    client = app_env.client
    _ingest(client)
    assert set(app_env.vectorstore.get_content_hashes_for_source("vault")) == {
        "Arkenfeld.md", str(Path("NPCs") / "Aldric.md"),
    }

    again = _ingest(client)
    assert again["documents_skipped"] == 2
    assert (again["chunks_created"], again["chunks_updated"], again["chunks_deleted"]) == (0, 0, 0)

    (app_env.vault / "Arkenfeld.md").write_text(
        _ARKENFELD + "\nSeit dem letzten Winter herrscht dort Ausgangssperre.\n", encoding="utf-8")
    (app_env.vault / "NPCs" / "Aldric.md").unlink()

    changed = _ingest(client)
    assert changed["documents_processed"] == 1
    assert changed["documents_skipped"] == 0
    assert changed["chunks_updated"] > 0
    assert changed["chunks_deleted"] > 0
    assert set(app_env.vectorstore.get_content_hashes_for_source("vault")) == {"Arkenfeld.md"}
    # Chunk total matches what is actually left in the store
    assert client.get("/stats").json()["chunk_count"] == changed["chunks_updated"]
