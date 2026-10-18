"""Security tests for the player mode (roadmap 3.11): no game-master content for players.

Real pipeline except the LLM: four sources are ingested over HTTP into an in-memory
ChromaDB — "regeln" (rules), "bogen" (lore, a character sheet) and "aushang" (a single file whose
folder holds every other source) released for players,
"welt" (lore) and "sl-regeln" (rules) for the game master only. Every
game-master text carries a secret sentence and is phrased to match rule questions,
so a leak would show up. The retrieval is configured as permissively as possible:

- a fake cross-encoder that rates game-master chunks highest (worst case for rerank
  and the cap backfill, ``max_per_source=1``),
- section expansion without the top-rank limit,
- the rulebook slot with no score minimum — its separate search inside group ``rules``
  would find the game-master rules source if it dropped the request filter,
- vector-only and hybrid (BM25) search.

A player question carries ``metadata_filters={"source_id": {"$in": ["regeln", "bogen", "aushang"]}}``
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
from src.api import sources_routes as sources_routes_module
from src.api.eval_routes import eval_router
from src.api.routes import router
from src.config.manager import (
    ConfigManager,
    EntrySlotConfig,
    RerankingConfig,
    RetrievalConfig,
    RulesSlotConfig,
    SectionExpansionConfig,
    SourceConfig,
    VectorStoreConfig,
)
from src.conversation.manager import ConversationManager
from src.generation.generator import Generator
from src.prompts.manager import PromptManager
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import Retriever
from src.retrieval.vectorstore import VectorStoreService
from tests.integration.conftest import parse_sse

_REPO = Path(__file__).resolve().parents[2]

PLAYER_SOURCES = ["regeln", "bogen", "aushang"]
PLAYER_FILTER = {"source_id": {"$in": PLAYER_SOURCES}}

# Every game-master text contains one of these — none may reach a player
SECRETS = [
    "Rhaz'Muell ist der wahre Sol Invictus",
    "Malek tötet die Helden absichtlich mit Überschaden",
    "Isolde Vaer ist von Nahlith besessen",
    "Die Sterbeprobe wird für Malek heimlich manipuliert",
    "Die Spielleitung würfelt Überschaden verdeckt doppelt",
    "Die Fluchklinge tötet jeden Spieler sofort",
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
_SHEET = {
    "Charakterbogen.md": (
        "# Charakterbogen\n\n"
        "## Kampfwerte\n\n"
        "Hier trägst du Lebenspunkte, Überschaden und Todeszähler ein. Die Sterbeprobe "
        "kreuzt du pro Runde an.\n\n"
        "## Ausrüstung\n\n"
        "Waffen, Rüstung und Tränke mit ihren Werten; Überschaden durch Waffen notierst du hier.\n\n"
        "## Notizen\n\n"
        "Platz für Sterbeproben, Todeszähler und Überschaden aus vergangenen Kämpfen.\n"
        # The file only exists in the game-master source: embeds resolve per source, never across
        "![[Rhaz Muell Portrait.png]]\n"
    ),
}
_GM_RULES = {
    # Rules group, but game master only — the rulebook slot's own search must never reach it
    "Spielleiter-Regeln.md": (
        "# Spielleiter-Regeln\n\n"
        "## Überschaden und Sterbeprobe\n\n"
        "Die Spielleitung würfelt Überschaden verdeckt doppelt. Kampfregeln, Todeszähler "
        "und Sterbeprobe gelten wie im Regelbuch.\n"
    ),
    # A table entry — bait for the entry slot, which finds tables by a row's name instead of by vector
    "Spielleiter-Waffen.md": (
        "# Spielleiter-Waffen\n\n"
        "## Verfluchte Waffen\n\n"
        "|**Waffe**|**Wirkung**|\n|---|---|\n"
        "|**Fluchklinge**|Die Fluchklinge tötet jeden Spieler sofort|\n"
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
        "![[Rhaz Muell Portrait.png]]\n"
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
    "Was macht die Fluchklinge?",
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
def env(tmp_path_factory, embed_service):
    """Ingest both sources once; yield client, retriever and the recording LLM."""
    tmp = tmp_path_factory.mktemp("player_security")
    rules_dir, world_dir = tmp / "regeln", tmp / "welt"
    sheet_dir, gm_rules_dir = tmp / "bogen", tmp / "sl-regeln"
    _write(rules_dir, _RULES)
    _write(world_dir, _WORLD)
    _write(sheet_dir, _SHEET)
    _write(gm_rules_dir, _GM_RULES)
    (world_dir / "Bilder").mkdir()
    (world_dir / "Bilder" / "Rhaz Muell Portrait.png").write_bytes(b"\x89PNG fake")  # embed bait (roadmap 2)
    # A file source's folder is the parent of all others: the image search must not look there (13.1 #6)
    notice = tmp / "Aushang.md"
    notice.write_text("# Aushang\n\nGesucht: ein Händler für die Reise nach Arkenfeld.\n"
                      "![[Rhaz Muell Portrait.png]]\n", encoding="utf-8")

    settings_path = tmp / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")
    config = ConfigManager(settings_path=settings_path, prompts_path=_REPO / "config" / "prompts.yaml",
                           sources_path=tmp / "sources.yaml")
    config.settings.ingestion.sources = [
        SourceConfig(id="regeln", path=str(rules_dir), group="rules", player_visible=True),
        SourceConfig(id="welt", path=str(world_dir), group="lore"),
        SourceConfig(id="bogen", path=str(sheet_dir), group="lore", player_visible=True),
        SourceConfig(id="aushang", path=str(notice), group="lore", player_visible=True),
        SourceConfig(id="sl-regeln", path=str(gm_rules_dir), group="rules"),
    ]

    embed = embed_service
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
            rules_slot=RulesSlotConfig(enabled=True, candidates=5, min_score=-100.0),
            entry_slot=EntrySlotConfig(enabled=True, max_chunks=3),
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
    sources_routes_module._ingest_jobs.clear()

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
        yield SimpleNamespace(client=client, retriever=retriever, llm=llm, config=config,
                              player_dirs=(rules_dir, sheet_dir, notice))

    for n, v in originals.items():
        setattr(main_module, n, v)
    sources_routes_module._ingest_jobs.clear()


def _is_player_file(env, source_path: str) -> bool:
    return any(Path(source_path).is_relative_to(d) for d in env.player_dirs)


def _no_secret(text: str) -> None:
    leaked = [s for s in SECRETS if s in text]
    assert not leaked, f"game-master content leaked: {leaked}"


def _stream(client, question: str, session_id: str | None = None) -> dict:
    body = {"question": question, "metadata_filters": PLAYER_FILTER}
    if session_id:
        body["session_id"] = session_id
    events = parse_sse(client.post("/query/stream", json=body).text)
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
        assert _is_player_file(env, src["source_path"]), src
        _no_secret(src["chunk_preview"])
        assert src["content"], src            # the full chunk text the chat shows (3.16) is really checked
        _no_secret(src["content"])
        assert all(_is_player_file(env, img) for img in src["images"]), src
    _no_secret(env.llm.prompts[-1])


def test_image_embeds_resolve_only_inside_their_own_source(env):
    """Roadmap 2: the game-master note shows its portrait; the character sheet embeds the same name,
    but the file lives only in the game-master source — the player's chunk must stay without it."""
    import asyncio
    gm = asyncio.run(env.retriever.retrieve("Wie sieht Rhaz'Muell aus? Erscheinung"))
    gm_images = {img for c in gm for img in c.metadata.get("images", "").split("|") if img}
    assert any(img.endswith("Rhaz Muell Portrait.png") for img in gm_images)   # bait works unfiltered

    # aushang: a file source — the portrait lies in a subfolder of its folder (ultrareview 13.1 #6)
    for player_source in ("bogen", "aushang"):
        player = env.retriever.vectorstore._get_collection().get(where={"source_id": player_source},
                                                                 include=["metadatas"])
        assert player["metadatas"] and not any(m.get("images") for m in player["metadatas"]), player_source


def test_condensed_follow_up_cannot_widen_the_search(env):
    """A follow-up whose condensed query is full of game-master words still only searches player sources."""
    first = _stream(env.client, "Was ist Überschaden?")
    follow_up = _stream(env.client, "Und was steckt dahinter?", session_id=first["session_id"])

    assert follow_up["search_query"].startswith("Rhaz'Muell")   # condense really ran with the bait
    for src in follow_up["sources"]:
        assert _is_player_file(env, src["source_path"]), src
    _no_secret(env.llm.prompts[-1])


def test_typo_correction_never_shows_a_player_a_game_master_word(env):
    """Review 13.5: truecasing and typo repair use one vocabulary over all sources, so "invictux" is
    corrected to "Invictus", a word only game-master text contains. The corrected query must stay inside
    the retriever — the player's search_query and the prompt carry the question as typed. (Known residual:
    a player who guesses the exact word can compare scores of guess and typo — documented in truecase.py.)"""
    question = "Was ist invictux?"
    env.retriever.truecaser.ensure_built(env.retriever.vectorstore)
    assert "Invictus" in env.retriever.truecaser.apply(question)   # the bait works, else this proves nothing

    status, events = _stream_as_player(env.client, question)

    assert status == 200
    assert events[-1]["search_query"] == question
    assert "Invictus" not in env.llm.prompts[-1]
    _no_secret(env.llm.prompts[-1])


# ─── Enforced in the backend (role, roadmap 4.2) — no UI filter needed ─────────

def _stream_as_player(client, question: str, **extra) -> tuple[int, list[dict]]:
    extra.setdefault("hybrid_search", False)
    resp = client.post("/query/stream", json={"question": question, "role": "player", **extra})
    events = parse_sse(resp.text)
    return resp.status_code, events


@pytest.mark.parametrize("hybrid", [False, True], ids=["vector", "hybrid"])
@pytest.mark.parametrize("question", QUESTIONS)
def test_player_role_alone_restricts_the_search(env, question, hybrid):
    """role=player without any metadata_filters: the backend builds the restriction itself."""
    status, events = _stream_as_player(env.client, question, hybrid_search=hybrid)

    assert status == 200
    done = events[-1]
    assert done["sources"], "expected rule sources"
    for src in done["sources"]:
        assert _is_player_file(env, src["source_path"]), src
    _no_secret(env.llm.prompts[-1])


@pytest.mark.parametrize("sneaky", [
    {"source_id": {"$in": ["welt"]}},
    {"source_id": "welt"},
    {"group": "lore"},
    {"$and": [{"source_id": {"$in": ["welt", "regeln"]}}, {"group": {"$in": ["lore", "rules"]}}]},
    {"source_id": "sl-regeln"},
], ids=["source-in", "source-eq", "group", "nested-and", "gm-rules"])
@pytest.mark.parametrize("hybrid", [False, True], ids=["vector", "hybrid"])
def test_player_filter_can_only_narrow(env, sneaky, hybrid):
    """A filter sent along is AND-ed with the restriction — pointing it at game-master sources finds nothing."""
    status, events = _stream_as_player(env.client, "Wer ist Sol Invictus wirklich?", metadata_filters=sneaky,
                                       hybrid_search=hybrid)

    assert status == 200
    for e in events:
        _no_secret(json.dumps(e, ensure_ascii=False))
    for src in events[-1].get("sources", []):
        assert _is_player_file(env, src["source_path"]), src


def test_player_cannot_continue_a_game_master_session(env):
    """With a game master's session id the player gets a fresh session — no condense against GM history."""
    gm = env.client.post("/query/stream", json={"question": "Wer ist Sol Invictus wirklich?"})
    gm_session = parse_sse(gm.text)[-1]["session_id"]

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


def test_player_without_released_sources_is_refused(env, monkeypatch, caplog):
    for s in env.config.settings.ingestion.sources:
        monkeypatch.setattr(s, "player_visible", False)
    assert env.client.post("/query/stream", json={"question": "Was ist Überschaden?", "role": "player"}).status_code == 403
    assert env.client.post("/eval/preview", json={"question": "x", "role": "player"}).status_code == 403
    refusals = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Refused player")]
    assert len(refusals) == 2, "a refusal must say why in the log, not only as a status code"


@pytest.mark.parametrize("hybrid", [False, True], ids=["vector", "hybrid"])
def test_player_rules_slot_stays_inside_released_sources(env, hybrid):
    """A player narrowed to the character sheet has no rules chunk, so the rulebook slot searches group
    "rules" — the game-master rules source is the best rules hit there and must stay out."""
    status, events = _stream_as_player(env.client, "Wie funktionieren Überschaden und Sterbeprobe?",
                                       metadata_filters={"source_id": "bogen"}, hybrid_search=hybrid)

    assert status == 200
    done = events[-1]
    assert done["sources"], "expected the character sheet"
    assert {Path(s["source_path"]).parent.name for s in done["sources"]} == {"bogen"}
    for e in events:
        _no_secret(json.dumps(e, ensure_ascii=False))
    _no_secret(env.llm.prompts[-1])


class SheetFirstReranker:
    """Game-master text highest (bait), then the character sheet, the released rulebook last —
    so a player's top places are all character-sheet chunks and the rulebook slot has to act."""

    def predict(self, pairs):
        return [10.0 if any(s in text for s in SECRETS) else 5.0 if "Charakterbogen" in text else -3.0
                for _, text in pairs]


async def test_player_rules_slot_brings_the_released_rulebook_not_the_game_master_rules(env, monkeypatch):
    """Positive case: the slot does fire for a player — and picks the released rulebook ("regeln"),
    never the game-master rules source ("sl-regeln") that the bait reranker rates highest."""
    monkeypatch.setattr(env.retriever, "_reranker", SheetFirstReranker())

    chunks = await env.retriever.retrieve("Wie funktionieren Überschaden und Sterbeprobe?",
                                          metadata_filters=PLAYER_FILTER, top_k_rerank=3, max_per_source=0)

    rules = [c for c in chunks if c.metadata.get("group") == "rules"]
    assert len(rules) == 1, [c.metadata.get("source_id") for c in chunks]   # the slot placed exactly one
    assert rules[0].metadata.get("source_id") == "regeln"
    assert {c.metadata.get("source_id") for c in chunks} <= set(PLAYER_SOURCES)
    for c in chunks:
        _no_secret(c.content)



def _restricted(where) -> bool:
    """True if ``where`` contains the player restriction (directly or inside an ``$and``)."""
    if not isinstance(where, dict):
        return False
    if where.get("source_id") == {"$in": PLAYER_SOURCES}:
        return True
    return any(_restricted(cond) for cond in where.get("$and", []))


def test_every_store_access_of_a_player_request_carries_the_restriction(env, monkeypatch):
    """Structural guarantee (roadmap 11.0 P4): every vector search, BM25 search and section load made
    for a player request is filtered — not just the paths that exist today. A new retrieval path that
    forgets the filter fails here even if the test data happens not to leak."""
    vs, bm25 = env.retriever.vectorstore, env.retriever.bm25_index
    calls: list[tuple[str, object]] = []

    def spy(name, fn, where_pos):
        def wrapper(*args, **kwargs):
            calls.append((name, kwargs.get("where", args[where_pos] if len(args) > where_pos else None)))
            return fn(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(vs, "query", spy("vector", vs.query, 2))
    monkeypatch.setattr(vs, "get_chunks_for_file", spy("expansion", vs.get_chunks_for_file, 2))
    monkeypatch.setattr(bm25, "query", spy("bm25", bm25.query, 2))
    monkeypatch.setattr(vs, "get_table_entries", spy("entry", vs.get_table_entries, 2))

    for question in QUESTIONS:
        r = env.client.post("/query/stream", json={"question": question, "role": "player", "hybrid_search": True})
        assert r.status_code == 200

    assert {name for name, _ in calls} == {"vector", "bm25", "expansion", "entry"}, "a path was not exercised"
    unfiltered = [(name, where) for name, where in calls if not _restricted(where)]
    assert not unfiltered, f"store access without the player restriction: {unfiltered}"
