"""Integration tests for the /sources/*, /ingest job lifecycle and /admin routes.

Like test_api_routes.py, these build a bare FastAPI app (no lifespan) and inject
fakes into ``src.main``. Config is a real ConfigManager on tmp files so that
PUT/DELETE /sources exercise the actual sources.yaml persistence.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from urllib.parse import quote

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.main as main_module
from src.api import routes as routes_module
from src.api.routes import router
from src.config.manager import ConfigManager, SourceConfig
from src.ingestion.orchestrator import IngestionOrchestrator, IngestionResult


class FakeVectorStore:
    """Records destructive calls instead of touching ChromaDB."""

    def __init__(self) -> None:
        self.deleted_sources: list[str] = []
        self.wiped = False

    def delete_by_source_id(self, source_id: str) -> int:
        self.deleted_sources.append(source_id)
        return 7

    def wipe_collection(self) -> None:
        self.wiped = True


class FakeBM25:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


@pytest.fixture
def env(tmp_path):
    """Real ConfigManager on tmp files + fake vectorstore/BM25, with a vault folder source."""
    vault = tmp_path / "vault"
    (vault / "NPCs").mkdir(parents=True)
    (vault / ".obsidian").mkdir()
    (vault / "Arkenfeld.md").write_text("# Arkenfeld", encoding="utf-8")
    (vault / "notes.xyz").write_text("unsupported", encoding="utf-8")
    single = tmp_path / "Regelbuch.md"
    single.write_text("# Regeln", encoding="utf-8")

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")
    sources_path = tmp_path / "sources.yaml"
    config = ConfigManager(
        settings_path=settings_path,
        prompts_path=tmp_path / "p.yaml",
        sources_path=sources_path,
    )
    config.settings.ingestion.sources = [
        SourceConfig(id="vault", path=str(vault), group="lore"),
        SourceConfig(id="Pen & Paper", path=str(single), group="rules"),
    ]

    names = ("config", "vectorstore", "bm25_index")
    originals = {n: getattr(main_module, n) for n in names}
    fakes = SimpleNamespace(
        config=config, vectorstore=FakeVectorStore(), bm25_index=FakeBM25(),
        vault=vault, single=single, sources_path=sources_path,
    )
    for n in names:
        setattr(main_module, n, getattr(fakes, n))
    routes_module._ingest_jobs.clear()

    yield fakes

    for n, v in originals.items():
        setattr(main_module, n, v)
    routes_module._ingest_jobs.clear()


@pytest.fixture
def client(env):
    """TestClient used as context manager so background ingest tasks keep their event loop."""
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _wait_for_job(client, job_id: str, timeout: float = 10.0) -> dict:
    """Poll /ingest/status until the job leaves queued/running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = client.get(f"/ingest/status/{job_id}").json()
        if info["status"] not in ("queued", "running"):
            return info
        time.sleep(0.05)
    pytest.fail(f"Job {job_id} did not finish within {timeout}s")


def _fake_result(**fields) -> IngestionResult:
    result = IngestionResult()
    for k, v in fields.items():
        setattr(result, k, v)
    return result


# ─── GET /sources, /sources/{id}/folders ──────────────────────────────────────

def test_list_sources(client, env):
    """Verify all configured sources are returned."""
    ids = [s["id"] for s in client.get("/sources").json()["sources"]]
    assert ids == ["vault", "Pen & Paper"]


def test_folders_lists_dirs_and_supported_files_only(client, env):
    """Verify hidden entries and unsupported formats are skipped."""
    body = client.get("/sources/vault/folders").json()
    assert body["folders"] == [
        {"name": "Arkenfeld.md", "type": "file"},
        {"name": "NPCs", "type": "folder"},
    ]


def test_folders_for_file_source(client, env):
    """Verify a single-file source reports no subfolders; URL-encoded ids are routed."""
    body = client.get(f"/sources/{quote('Pen & Paper', safe='')}/folders").json()
    assert body["folders"] == []
    assert "note" in body


def test_folders_missing_path(client, env):
    """Verify a source whose path vanished returns an error field, not a 500."""
    env.config.settings.ingestion.sources[0].path = str(env.vault / "gone")
    body = client.get("/sources/vault/folders").json()
    assert body["folders"] == [] and "error" in body


def test_folders_unknown_source_404(client, env):
    assert client.get("/sources/nope/folders").status_code == 404


# ─── POST /sources/scan ───────────────────────────────────────────────────────

def test_scan_missing_path_400(client, env):
    assert client.post("/sources/scan", json={}).status_code == 400


def test_scan_folder(client, env):
    body = client.post("/sources/scan", json={"path": str(env.vault)}).json()
    assert body["is_file"] is False
    assert {"name": "NPCs", "type": "folder"} in body["folders"]


def test_scan_file(client, env):
    body = client.post("/sources/scan", json={"path": str(env.single)}).json()
    assert body == {"folders": [], "is_file": True}


def test_scan_nonexistent(client, env):
    body = client.post("/sources/scan", json={"path": str(env.vault / "gone")}).json()
    assert "error" in body


# ─── PUT /sources ─────────────────────────────────────────────────────────────

def _src(id_: str, env) -> dict:
    return {"id": id_, "path": str(env.vault), "group": "lore"}


def test_put_sources_persists_to_sidecar(client, env):
    """Verify the new list replaces the in-memory config and is written to sources.yaml."""
    r = client.put("/sources", json={"sources": [_src("neu", env)]})
    assert r.status_code == 200
    assert [s.id for s in env.config.settings.ingestion.sources] == ["neu"]
    saved = yaml.safe_load(env.sources_path.read_text(encoding="utf-8"))
    assert [s["id"] for s in saved["sources"]] == ["neu"]


def test_player_visible_defaults_off_and_round_trips(client, env, caplog):
    """New sources are game-master-only; the player flag survives PUT, sources.yaml and GET — and the log
    shows which sources players can search from now on."""
    caplog.set_level(logging.INFO, logger="src.api.routes")
    r = client.put("/sources", json={"sources": [
        {**_src("regeln", env), "player_visible": True}, _src("lore", env),
    ]})
    assert r.status_code == 200
    assert "Sources saved: 2 sources, player_visible=['regeln']" in [rec.getMessage() for rec in caplog.records]

    saved = {s["id"]: s for s in yaml.safe_load(env.sources_path.read_text(encoding="utf-8"))["sources"]}
    assert saved["regeln"]["player_visible"] is True
    assert "player_visible" not in saved["lore"]          # default (off) is not written
    listed = {s["id"]: s["player_visible"] for s in client.get("/sources").json()["sources"]}
    assert listed == {"regeln": True, "lore": False}


def test_sidebar_state_lists_player_sources(client, env, monkeypatch):
    """The chat's player mode gets exactly the sources marked player_visible."""
    monkeypatch.setattr(main_module, "provider", SimpleNamespace(provider="ollama", model="qwen3:14b"))
    env.config.settings.ingestion.sources[1].player_visible = True

    body = client.get("/sidebar-state").json()

    assert body["player_source_ids"] == ["Pen & Paper"]


def test_sidebar_state_carries_the_active_providers_timeout(client, env, monkeypatch):
    """The chat takes its read timeout from here instead of fetching /config per message."""
    env.config.settings.llm.ollama.timeout = 123
    env.config.settings.llm.gemini.timeout = 45
    monkeypatch.setattr(main_module, "provider", SimpleNamespace(provider="gemini", model="gemini-2.5-flash"))

    assert client.get("/sidebar-state").json()["llm_timeout"] == 45


def test_put_sources_missing_field_400(client, env):
    assert client.put("/sources", json={}).status_code == 400


def test_put_sources_invalid_group_400(client, env):
    bad = {"id": "x", "path": "p", "group": "nope"}
    assert client.put("/sources", json={"sources": [bad]}).status_code == 400


def test_put_sources_duplicate_ids_400(client, env):
    r = client.put("/sources", json={"sources": [_src("a", env), _src("a", env)]})
    assert r.status_code == 400
    assert "Doppelte" in r.json()["detail"]


@pytest.mark.parametrize("bad_id", ["", "  ", ".", "..", " vault", "vault "])
def test_put_sources_rejects_unroutable_ids(client, env, bad_id):
    """Verify new ids the /sources/{id}/... routes can't serve are rejected, config untouched."""
    r = client.put("/sources", json={"sources": [_src(bad_id, env)]})
    assert r.status_code == 400
    assert "Ungültige Quellen-ID" in r.json()["detail"]
    assert [s.id for s in env.config.settings.ingestion.sources] == ["vault", "Pen & Paper"]


@pytest.mark.parametrize("ok_id", ["Pen & Paper", "foo#1", "a.b", "märchen", "Lore/NPCs"])
def test_put_sources_accepts_free_text_ids(client, env, ok_id):
    assert client.put("/sources", json={"sources": [_src(ok_id, env)]}).status_code == 200


def test_put_sources_legacy_id_does_not_block_saving(client, env):
    """Verify an already configured id that would be rejected as new can still be saved."""
    env.config.settings.ingestion.sources.append(
        SourceConfig(id=" legacy", path=str(env.single), group="rules"))
    payload = [s.model_dump() for s in env.config.settings.ingestion.sources] + [_src("neu", env)]
    assert client.put("/sources", json={"sources": payload}).status_code == 200
    assert [s.id for s in env.config.settings.ingestion.sources][-2:] == [" legacy", "neu"]


# ─── ids containing '/' (routes use the :path converter) ──────────────────────

@pytest.fixture
def slash_source(env):
    env.config.settings.ingestion.sources.append(
        SourceConfig(id="Lore/NPCs", path=str(env.vault), group="lore"))
    return quote("Lore/NPCs", safe="")


def test_folders_with_slash_id(client, env, slash_source):
    assert client.get(f"/sources/{slash_source}/folders").json()["folders"]


def test_reindex_with_slash_id(client, env, slash_source, monkeypatch):
    monkeypatch.setattr(IngestionOrchestrator, "run", lambda self, **kw: _fake_result())
    job_id = client.post(f"/sources/{slash_source}/reindex").json()["job_id"]
    assert _wait_for_job(client, job_id)["status"] == "done"
    assert env.vectorstore.deleted_sources == ["Lore/NPCs"]


def test_delete_with_slash_id(client, env, slash_source):
    assert client.delete(f"/sources/{slash_source}").status_code == 200
    assert "Lore/NPCs" not in [s.id for s in env.config.settings.ingestion.sources]


def test_static_source_routes_not_shadowed(client, env, monkeypatch):
    """Verify /sources/scan and /sources/recategorize still win over the :path routes."""
    import src.ingestion.recategorize as recat_module
    monkeypatch.setattr(recat_module, "recategorize", lambda config, vectorstore: {"ok": True})
    assert client.post("/sources/recategorize").json() == {"ok": True}
    assert client.post("/sources/scan", json={"path": str(env.single)}).json()["is_file"] is True


# ─── DELETE /sources/{id} ─────────────────────────────────────────────────────

def test_delete_source_removes_chunks_and_config(client, env):
    """Verify delete drops chunks, removes the source and persists the sidecar."""
    r = client.delete(f"/sources/{quote('Pen & Paper', safe='')}")
    assert r.status_code == 200
    assert r.json() == {"deleted_chunks": 7}
    assert env.vectorstore.deleted_sources == ["Pen & Paper"]
    assert [s.id for s in env.config.settings.ingestion.sources] == ["vault"]
    saved = yaml.safe_load(env.sources_path.read_text(encoding="utf-8"))
    assert [s["id"] for s in saved["sources"]] == ["vault"]


def test_delete_unknown_source_404(client, env):
    assert client.delete("/sources/nope").status_code == 404
    assert env.vectorstore.deleted_sources == []


# ─── POST /sources/recategorize ───────────────────────────────────────────────

def test_recategorize_returns_summary(client, env, monkeypatch):
    import src.ingestion.recategorize as recat_module
    monkeypatch.setattr(recat_module, "recategorize", lambda config, vectorstore: {"updated": 3})
    assert client.post("/sources/recategorize").json() == {"updated": 3}


def test_recategorize_applies_a_mapping_saved_on_the_sources_page(client, env, monkeypatch):
    """Map a folder on the Sources page, click "Recategorize": the stored chunk gets the new category.

    The route used to fall back to a second, cached ConfigManager that never saw ``PUT /sources`` —
    the button reported 0 updated chunks (roadmap 11.0 #1). The real recategorize runs here."""
    stored = {"ids": ["c1"], "metadatas": [{"source_id": "vault", "source_file": "NPCs/Malek.md",
                                            "content_category": "misc", "group": "lore"}]}
    written = {}
    monkeypatch.setattr(env.vectorstore, "get_all", lambda include: stored, raising=False)
    monkeypatch.setattr(env.vectorstore, "update_metadata_batch",
                        lambda ids, metas: written.update(zip(ids, metas, strict=True)), raising=False)

    sources = client.get("/sources").json()["sources"]
    sources[0]["category_map"] = {"NPCs": "npc"}
    assert client.put("/sources", json={"sources": sources}).status_code == 200

    result = client.post("/sources/recategorize").json()

    assert result["chunks_updated"] == 1
    assert written["c1"]["content_category"] == "npc"


# ─── Ingest job lifecycle ─────────────────────────────────────────────────────

def test_ingest_job_runs_to_done(client, env, monkeypatch):
    """Verify a job goes queued → done, copies result counters, and invalidates BM25."""
    calls = {}

    def fake_run(self, vectorstore=None, only_source_id=None, progress_callback=None, full=False):
        calls.update(full=full, only=only_source_id)
        return _fake_result(documents_processed=5, documents_total=5, documents_skipped=3,
                            chunks_created=4, chunks_deleted=2, errors=["warn"])

    monkeypatch.setattr(IngestionOrchestrator, "run", fake_run)
    job_id = client.post("/ingest").json()["job_id"]
    info = _wait_for_job(client, job_id)

    assert info["status"] == "done"
    assert (info["documents_processed"], info["documents_skipped"]) == (5, 3)
    assert (info["chunks_created"], info["chunks_deleted"]) == (4, 2)
    assert info["errors"] == ["warn"]
    assert calls == {"full": False, "only": None}
    assert env.bm25_index.invalidated == 1


def test_ingest_full_flag_passed_through(client, env, monkeypatch):
    seen = {}

    def fake_run(self, vectorstore=None, only_source_id=None, progress_callback=None, full=False):
        seen["full"] = full
        return _fake_result()

    monkeypatch.setattr(IngestionOrchestrator, "run", fake_run)
    job_id = client.post("/ingest", params={"full": "true"}).json()["job_id"]
    _wait_for_job(client, job_id)
    assert seen["full"] is True


def test_ingest_job_error_status(client, env, monkeypatch):
    """Verify an orchestrator exception marks the job as error and frees the slot."""
    def boom(self, **kwargs):
        raise RuntimeError("chroma locked")

    monkeypatch.setattr(IngestionOrchestrator, "run", boom)
    job_id = client.post("/ingest").json()["job_id"]
    info = _wait_for_job(client, job_id)
    assert info["status"] == "error"
    assert info["errors"] == ["chroma locked"]
    assert env.bm25_index.invalidated == 0
    # A failed job must not block the next one with 409
    monkeypatch.setattr(IngestionOrchestrator, "run", lambda self, **kw: _fake_result())
    assert client.post("/ingest").status_code == 200


def test_ingest_progress_visible_while_running(client, env, monkeypatch):
    """Verify progress callback updates are visible via /ingest/status mid-run."""
    import threading
    release = threading.Event()

    def slow_run(self, vectorstore=None, only_source_id=None, progress_callback=None, full=False):
        r = _fake_result(documents_total=10, documents_processed=4, phase="parsing")
        progress_callback(r)
        release.wait(5)
        return r

    monkeypatch.setattr(IngestionOrchestrator, "run", slow_run)
    job_id = client.post("/ingest").json()["job_id"]
    deadline = time.monotonic() + 5
    info = {}
    while time.monotonic() < deadline:
        info = client.get(f"/ingest/status/{job_id}").json()
        if info["documents_processed"] == 4:
            break
        time.sleep(0.05)
    assert info["status"] == "running"
    assert (info["documents_processed"], info["documents_total"], info["phase"]) == (4, 10, "parsing")
    release.set()
    assert _wait_for_job(client, job_id)["status"] == "done"


def test_ingest_active_reports_the_running_job(client, env, monkeypatch):
    """/ingest/active names the job that blocks a second start (409), and null once it is done."""
    import threading
    release = threading.Event()

    def slow_run(self, vectorstore=None, only_source_id=None, progress_callback=None, full=False):
        release.wait(5)
        return _fake_result()

    assert client.get("/ingest/active").json() is None
    monkeypatch.setattr(IngestionOrchestrator, "run", slow_run)
    job_id = client.post("/ingest").json()["job_id"]

    assert client.post("/ingest").status_code == 409
    active = client.get("/ingest/active").json()
    assert active["job_id"] == job_id
    assert active["status"] in ("queued", "running")

    release.set()
    _wait_for_job(client, job_id)
    assert client.get("/ingest/active").json() is None


def test_reindex_source_deletes_then_ingests_only_that_source(client, env, monkeypatch):
    seen = {}

    def fake_run(self, vectorstore=None, only_source_id=None, progress_callback=None, full=False):
        seen["only"] = only_source_id
        seen["deleted_before"] = list(env.vectorstore.deleted_sources)
        return _fake_result()

    monkeypatch.setattr(IngestionOrchestrator, "run", fake_run)
    job_id = client.post(f"/sources/{quote('Pen & Paper', safe='')}/reindex").json()["job_id"]
    assert _wait_for_job(client, job_id)["status"] == "done"
    assert seen == {"only": "Pen & Paper", "deleted_before": ["Pen & Paper"]}


def test_reindex_unknown_source_404(client, env):
    assert client.post("/sources/nope/reindex").status_code == 404


# ─── /admin/wipe ──────────────────────────────────────────────────────────────

def test_wipe_requires_confirmation(client, env):
    assert client.post("/admin/wipe", json={"confirm": "yes"}).status_code == 400
    assert env.vectorstore.wiped is False


def test_wipe_with_confirmation(client, env):
    assert client.post("/admin/wipe", json={"confirm": "DELETE"}).json() == {"status": "wiped"}
    assert env.vectorstore.wiped is True
