"""Security tests for the player mode (roadmap 3.11): no game-master content for players.

Real pipeline except the LLM: two sources are ingested over HTTP into an in-memory
ChromaDB — "regeln" (released for players) and "welt" (game master only). Every
game-master text carries a secret sentence and is phrased to match rule questions,
so a leak would show up. The retrieval is configured as permissively as possible:

- a fake cross-encoder that rates game-master chunks highest (worst case for rerank
  and the cap backfill, ``max_per_source=1``),
- section expansion without the top-rank limit,
- vector-only and hybrid (BM25) search.

A player question carries ``metadata_filters={"source_id": {"$in": ["regeln"]}}``
(what ui/roles.py sends). Neither the retrieved chunks, the sources in the done
event nor the prompt handed to the LLM may contain game-master material.
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
from src.api.eval_routes import eval_router
from src.api.routes import router
from src.config.manager import (
    ConfigManager,
    EmbeddingsConfig,
    RerankingConfig,
    RetrievalConfig,
    SectionExpansionConfig,
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

PLAYER_SOURCES = ["regeln"]
PLAYER_FILTER = {"source_id": {"$in": PLAYER_SOURCES}}

# Every game-master text contains one of these — none may reach a player
SECRETS = [
    "Rhaz'Muell ist der wahre Sol Invictus",
    "Malek tötet die Helden absichtlich mit Überschaden",
    "Isolde Vaer ist von Nahlith besessen",
    "Die Sterbeprobe wird für Malek heimlich manipuliert",
]

_RULES = {
    "Regelbuch.md": (
        "# 4 Kampfregeln\n\n"
        "## 4.10 Kampfunfähigkeit\n\n"
        "### Überschaden\n\n"
        "Übersteigt der Schaden die verbleibenden Lebenspunkte um die maximalen LP, "
        "entsteht Überschaden. Folge ist ein zusätzlicher Todeszähler.\n\n"
        "### Sterbeprobe\n\n"
        "Zu Beginn jeder Runde legt der Kampfunfähige eine Probe ab. Ab Wurfzahl 10 "
        "stabilisiert er sich.\n\n"
        "## 4.11 Kampfende\n\n"
        "### Überheilung\n\n"
        "Überheilung verfällt am Ende des Kampfes.\n"
    ),
}
_WORLD = {
    # Same file name as the rulebook in another source: expansion must key by source_id.
    # The secret sits at a chunk index the rulebook doesn't have (3), so a cross-source
    # expansion can't be masked by the chunk de-duplication.
    "Regelbuch.md": (
        "# 4 Kampfregeln\n\n"
        "## 4.10 Kampfunfähigkeit\n\n"
        "### Vorbereitung (SL)\n\nNotizen zur Vorbereitung der Kampfszene am Spieltisch.\n\n"
        "### Ablauf (SL)\n\nDie Szene beginnt im Lagerhaus, die Gegner warten im Dunkeln.\n\n"
        "### Gegner (SL)\n\nZwei Kultisten und ein Schattenwesen stehen bereit.\n\n"
        "### Überschaden (SL-Notiz)\n\n"
        "Überschaden und Todeszähler: Malek tötet die Helden absichtlich mit Überschaden. "
        "Die Sterbeprobe wird für Malek heimlich manipuliert.\n"
    ),
    "Sol Invictus.md": (
        "# Sol Invictus\n\n"
        "## Geheimnis\n\n"
        "Wer ist Sol Invictus wirklich? Rhaz'Muell ist der wahre Sol Invictus. "
        "Kampfregeln, Überschaden und Sterbeprobe gelten für ihn nicht.\n\n"
        "## Erscheinung\n\n"
        "Rhaz'Muell ist der wahre Sol Invictus — eine Gestalt aus Licht und Glut.\n"
    ),
    "NPCs/Isolde Vaer.md": (
        "# Isolde Vaer\n\n"
        "## Wahrheit\n\n"
        "Isolde Vaer ist von Nahlith besessen. Regeln zur Kampfunfähigkeit und "
        "Überheilung nutzt Nahlith gegen die Gruppe.\n"
    ),
}

QUESTIONS = [
    "Was ist Überschaden?",
    "Wie funktioniert die Sterbeprobe?",
    "Wer ist Sol Invictus wirklich?",
    "Was ist mit Isolde Vaer los?",
    "Malek Überschaden Todeszähler Sterbeprobe manipuliert",
]

_ANSWER = ["Laut ", "Regelwerk ", "…"]


class FakeLLM:
    """Records every prompt; condense returns a follow-up full of game-master words."""
    provider = "fake"
    model = "fake-llm"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_stream(self, prompt, system_prompt=None, stream_result=None, **kw):
        self.prompts.append(prompt)
        for tok in _ANSWER:
            yield tok

    async def generate(self, prompt, **kw):
        # A condensed question that drags game-master context into the search query
        from src.generation.providers.base import LLMResponse
        return LLMResponse(content="Rhaz'Muell Sol Invictus Nahlith Isolde Malek Überschaden Geheimnis",
                           model=self.model, provider=self.provider)

    async def health_check(self) -> bool:
        return True


class GameMasterFirstReranker:
    """Cross-encoder stand-in that rates game-master text highest — the worst case."""

    def predict(self, pairs):
        return [10.0 if any(s in text for s in SECRETS) else 1.0 for _, text in pairs]


def _write(folder: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Ingest both sources once; yield client, retriever and the recording LLM."""
    tmp = tmp_path_factory.mktemp("player_security")
    rules_dir, world_dir = tmp / "regeln", tmp / "welt"
    _write(rules_dir, _RULES)
    _write(world_dir, _WORLD)

    settings_path = tmp / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")
    config = ConfigManager(settings_path=settings_path, prompts_path=_REPO / "config" / "prompts.yaml",
                           sources_path=tmp / "sources.yaml")
    config.settings.ingestion.sources = [
        SourceConfig(id="regeln", path=str(rules_dir), group="rules", player_visible=True),
        SourceConfig(id="welt", path=str(world_dir), group="lore"),
    ]

    embed = EmbeddingService(EmbeddingsConfig(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu", batch_size=32, normalize=True,
    ))
    vectorstore = VectorStoreService(
        VectorStoreConfig(mode="embedded", collection_name=f"player_{uuid.uuid4().hex}", distance_metric="cosine"),
        embed,
    )
    vectorstore._client = chromadb.EphemeralClient()
    bm25 = BM25Index()
    retriever = Retriever(
        RetrievalConfig(
            top_k=15, score_threshold=0.0,
            reranking=RerankingConfig(enabled=True, top_k_rerank=8, max_per_source=1),
            section_expansion=SectionExpansionConfig(enabled=True, top_rank=0, max_chunks=50),
        ),
        embed, vectorstore, bm25,
    )
    retriever._reranker = GameMasterFirstReranker()
    llm = FakeLLM()
    config.settings.conversation.condense_question = True

    services = {
        "config": config,
        "conversation_manager": ConversationManager(config.settings.conversation),
        "prompt_manager": PromptManager(prompts_dict=config.prompts),
        "embedding_service": embed,
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
    app.include_router(eval_router)
    with TestClient(app) as client:
        job_id = client.post("/ingest").json()["job_id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            info = client.get(f"/ingest/status/{job_id}").json()
            if info["status"] not in ("queued", "running"):
                break
            time.sleep(0.1)
        assert info["status"] == "done" and not info["errors"], info
        yield SimpleNamespace(client=client, retriever=retriever, llm=llm, rules_dir=rules_dir, config=config)

    for n, v in originals.items():
        setattr(main_module, n, v)
    routes_module._ingest_jobs.clear()


def _no_secret(text: str) -> None:
    leaked = [s for s in SECRETS if s in text]
    assert not leaked, f"game-master content leaked: {leaked}"


def _stream(client, question: str, session_id: str | None = None) -> dict:
    body = {"question": question, "metadata_filters": PLAYER_FILTER}
    if session_id:
        body["session_id"] = session_id
    events = [json.loads(line[6:]) for line in client.post("/query/stream", json=body).text.splitlines()
              if line.startswith("data: ")]
    assert not [e for e in events if e["type"] == "error"], events
    return events[-1]


def test_unfiltered_search_does_find_the_secrets(env):
    """Sanity check: without the player filter the bait works — otherwise the tests below prove nothing."""
    chunks = pytest.importorskip("asyncio").run(env.retriever.retrieve("Wer ist Sol Invictus wirklich?"))
    assert any(any(s in c.content for s in SECRETS) for c in chunks)


@pytest.mark.parametrize("hybrid", [False, True], ids=["vector", "hybrid"])
@pytest.mark.parametrize("question", QUESTIONS)
async def test_player_retrieval_only_returns_released_sources(env, question, hybrid):
    """Vector/BM25 search, rerank, cap backfill and section expansion stay inside the player sources."""
    chunks = await env.retriever.retrieve(question, metadata_filters=PLAYER_FILTER, hybrid=hybrid)

    assert chunks, "player search returned nothing — rule content should match"
    assert {c.metadata.get("source_id") for c in chunks} <= set(PLAYER_SOURCES)
    for c in chunks:
        _no_secret(c.content)


@pytest.mark.parametrize("question", QUESTIONS)
def test_player_answer_path_is_clean(env, question):
    """End to end over /query/stream: sources and the prompt the LLM sees contain no game-master text."""
    done = _stream(env.client, question)

    assert done["sources"], "expected rule sources"
    for src in done["sources"]:
        assert Path(src["source_path"]).is_relative_to(env.rules_dir), src
        _no_secret(src["chunk_preview"])
    _no_secret(env.llm.prompts[-1])


def test_condensed_follow_up_cannot_widen_the_search(env):
    """A follow-up whose condensed query is full of game-master words still only searches player sources."""
    first = _stream(env.client, "Was ist Überschaden?")
    follow_up = _stream(env.client, "Und was steckt dahinter?", session_id=first["session_id"])

    assert follow_up["search_query"].startswith("Rhaz'Muell")   # condense really ran with the bait
    for src in follow_up["sources"]:
        assert Path(src["source_path"]).is_relative_to(env.rules_dir), src
    _no_secret(env.llm.prompts[-1])


# ─── Enforced in the backend (role, roadmap 4.2) — no UI filter needed ─────────

def _stream_as_player(client, question: str, **extra) -> tuple[int, list[dict]]:
    resp = client.post("/query/stream", json={"question": question, "role": "player", **extra})
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    return resp.status_code, events


@pytest.mark.parametrize("question", QUESTIONS)
def test_player_role_alone_restricts_the_search(env, question):
    """role=player without any metadata_filters: the backend builds the restriction itself."""
    status, events = _stream_as_player(env.client, question)

    assert status == 200
    done = events[-1]
    assert done["sources"], "expected rule sources"
    for src in done["sources"]:
        assert Path(src["source_path"]).is_relative_to(env.rules_dir), src
    _no_secret(env.llm.prompts[-1])


@pytest.mark.parametrize("sneaky", [
    {"source_id": {"$in": ["welt"]}},
    {"source_id": "welt"},
    {"group": "lore"},
    {"$and": [{"source_id": {"$in": ["welt", "regeln"]}}, {"group": {"$in": ["lore", "rules"]}}]},
], ids=["source-in", "source-eq", "group", "nested-and"])
def test_player_filter_can_only_narrow(env, sneaky):
    """A filter sent along is AND-ed with the restriction — pointing it at game-master sources finds nothing."""
    status, events = _stream_as_player(env.client, "Wer ist Sol Invictus wirklich?", metadata_filters=sneaky)

    assert status == 200
    for e in events:
        _no_secret(json.dumps(e, ensure_ascii=False))
    for src in events[-1].get("sources", []):
        assert Path(src["source_path"]).is_relative_to(env.rules_dir), src


def test_player_cannot_continue_a_game_master_session(env):
    """With a game master's session id the player gets a fresh session — no condense against GM history."""
    gm = env.client.post("/query/stream", json={"question": "Wer ist Sol Invictus wirklich?"})
    gm_session = [json.loads(line[6:]) for line in gm.text.splitlines() if line.startswith("data: ")][-1]["session_id"]

    _, events = _stream_as_player(env.client, "Und was steckt dahinter?", session_id=gm_session)

    done = events[-1]
    assert done["session_id"] != gm_session
    assert done["search_query"] == "Und was steckt dahinter?"   # no history → no condense with the bait
    _no_secret(env.llm.prompts[-1])


def test_player_preview_is_restricted_too(env):
    data = env.client.post("/eval/preview", json={
        "question": "Wer ist Sol Invictus wirklich?", "role": "player",
        "metadata_filters": {"source_id": {"$in": ["welt"]}},
    })
    assert data.status_code == 200
    assert {c["source_id"] for c in data.json()["chunks"]} <= set(PLAYER_SOURCES)


def test_player_without_released_sources_is_refused(env, monkeypatch):
    for s in env.config.settings.ingestion.sources:
        monkeypatch.setattr(s, "player_visible", False)
    assert env.client.post("/query/stream", json={"question": "Was ist Überschaden?", "role": "player"}).status_code == 403
    assert env.client.post("/eval/preview", json={"question": "x", "role": "player"}).status_code == 403
