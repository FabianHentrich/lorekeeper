from pathlib import Path

import pytest

from src.config.manager import SourceConfig
from src.ingestion.orchestrator import (
    _resolve_category,
    _compute_content_hash,
    _is_excluded,
    IngestionOrchestrator,
)


def _folder_source(tmp_path: Path, **kwargs) -> SourceConfig:
    """
    Helper function to generate a mocked SourceConfig pointing to a temporary folder structure.

    Args:
        tmp_path (Path): The Pytest fixture temporary directory.
        **kwargs: Optional overrides for SourceConfig initialization.

    Returns:
        SourceConfig: A fully hydrated mock configuration context.
    """
    defaults = {
        "id": "test",
        "path": str(tmp_path),
        "group": "lore",
        "default_category": "misc",
        "category_map": {
            "NPCs": "npc",
            "Orte": "location",
            "Geschichte": "story",
        },
    }
    defaults.update(kwargs)
    return SourceConfig(**defaults)


class TestResolveCategory:
    """Test suite validating dynamic source categorization mapping rules and fallback behaviors."""

    def test_folder_source_top_folder_match(self, tmp_path):
        """
        Verify that files matching a top-level directory in the category map
        successfully resolve to the specified mapped category.
        """
        (tmp_path / "NPCs").mkdir()
        f = tmp_path / "NPCs" / "aldric.md"
        f.write_text("x")
        src = _folder_source(tmp_path)
        cat, grp = _resolve_category(f, src)
        assert cat == "npc"
        assert grp == "lore"

    def test_folder_source_case_insensitive(self, tmp_path):
        """
        Verify that directory name parsing effectively ignores case differences
        when attempting category map lookups.
        """
        (tmp_path / "orte").mkdir()
        f = tmp_path / "orte" / "arkenfeld.md"
        f.write_text("x")
        src = _folder_source(tmp_path)
        cat, grp = _resolve_category(f, src)
        assert cat == "location"
        assert grp == "lore"

    def test_folder_source_unknown_top_folder_falls_back_to_default(self, tmp_path):
        """
        Verify that unknown top-level folders without mappings gracefully fall back
        to the ingestion source's defined default generic category.
        """
        (tmp_path / "Random").mkdir()
        f = tmp_path / "Random" / "x.md"
        f.write_text("x")
        src = _folder_source(tmp_path)
        cat, grp = _resolve_category(f, src)
        assert cat == "misc"
        assert grp == "lore"

    def test_folder_source_root_file_falls_back_to_default(self, tmp_path):
        """
        Verify that files residing strictly at the root folder of the source, having no
        parent directory block, fallback to the default category immediately.
        """
        f = tmp_path / "readme.md"
        f.write_text("x")
        src = _folder_source(tmp_path)
        cat, grp = _resolve_category(f, src)
        assert cat == "misc"
        assert grp == "lore"

    def test_file_source_always_uses_default_category(self, tmp_path):
        """
        Verify that single-file source references always map onto their own
        default definition ignoring generic structural rules.
        """
        f = tmp_path / "rulebook.pdf"
        f.write_text("x")
        src = SourceConfig(id="rb", path=str(f), group="rules", default_category="rules")
        cat, grp = _resolve_category(f, src)
        assert cat == "rules"
        assert grp == "rules"

    def test_dict_entry_overrides_group(self, tmp_path):
        """
        Verify that nested dict-based categorizations in the settings can overwrite
        the parent group context of an item.
        """
        (tmp_path / "Geschichte").mkdir()
        f = tmp_path / "Geschichte" / "akt1.md"
        f.write_text("x")
        src = _folder_source(tmp_path, category_map={
            "Geschichte": {"category": "story", "group": "adventure"},
        })
        cat, grp = _resolve_category(f, src)
        assert cat == "story"
        assert grp == "adventure"

    def test_dict_entry_without_group_inherits_source_group(self, tmp_path):
        """
        Verify that dictionary-based categorize mappings that omit Group definitions
        successfully inherit their group from the fallback source layer.
        """
        (tmp_path / "NPCs").mkdir()
        f = tmp_path / "NPCs" / "aldric.md"
        f.write_text("x")
        src = _folder_source(tmp_path, category_map={
            "NPCs": {"category": "npc"},
        })
        cat, grp = _resolve_category(f, src)
        assert cat == "npc"
        assert grp == "lore"

    def test_mixed_string_and_dict_entries(self, tmp_path):
        """
        Verify that both short string and long object configurations co-exist
        in a single mapping schema without conflicting.
        """
        (tmp_path / "NPCs").mkdir()
        (tmp_path / "Regelwerk").mkdir()
        f_npc = tmp_path / "NPCs" / "a.md"
        f_rules = tmp_path / "Regelwerk" / "b.md"
        f_npc.write_text("x")
        f_rules.write_text("x")
        src = _folder_source(tmp_path, category_map={
            "NPCs": "npc",
            "Regelwerk": {"category": "rules", "group": "rules"},
        })
        assert _resolve_category(f_npc, src) == ("npc", "lore")
        assert _resolve_category(f_rules, src) == ("rules", "rules")


class TestDiscoverFiles:
    """Test suite validating directory walking logic against multiple active Source configs."""

    def _make_orchestrator(self, monkeypatch, sources):
        """
        Helper method injecting synthetic sources into the orchestrator pipeline.
        """
        from src.config.manager import config_manager
        monkeypatch.setattr(config_manager.settings.ingestion, "sources", sources)
        return IngestionOrchestrator()

    def test_file_source_yields_single_file(self, tmp_path, monkeypatch):
        """
        Verify that targeting a solitary file strictly resolves to just that item.
        """
        f = tmp_path / "rulebook.pdf"
        f.write_text("x")
        src = SourceConfig(id="rb", path=str(f), group="rules", default_category="rules")
        orch = self._make_orchestrator(monkeypatch, [src])
        files = orch._discover_files()
        assert len(files) == 1
        assert files[0][0] == f.resolve()
        assert files[0][1].id == "rb"

    def test_folder_source_walks_recursively(self, tmp_path, monkeypatch):
        """
        Verify that declaring a directory initiates recursive walks to ingest structurally nested markdown.
        """
        (tmp_path / "NPCs").mkdir()
        (tmp_path / "NPCs" / "a.md").write_text("x")
        (tmp_path / "NPCs" / "b.md").write_text("x")
        src = _folder_source(tmp_path)
        orch = self._make_orchestrator(monkeypatch, [src])
        files = orch._discover_files()
        assert len(files) == 2

    def test_folder_source_matches_extensions_case_insensitively(self, tmp_path, monkeypatch):
        """
        Verify uppercase extensions (.PNG, .MD) and .jpeg are discovered — rglob is case-sensitive on Linux.
        """
        for name in ("Karte.PNG", "Notiz.MD", "Foto.jpeg", "ignore.txt"):
            (tmp_path / name).write_text("x")
        src = _folder_source(tmp_path)
        orch = self._make_orchestrator(monkeypatch, [src])
        names = sorted(f.name for f, _ in orch._discover_files())
        assert names == ["Foto.jpeg", "Karte.PNG", "Notiz.MD"]

    def test_only_source_id_filter(self, tmp_path, monkeypatch):
        """
        Verify that the Orchestrator can ingest a designated subset of documents filtering
        strictly by the configured Source ID reference.
        """
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_text("x"); f2.write_text("x")
        src1 = SourceConfig(id="s1", path=str(f1), group="rules", default_category="rules")
        src2 = SourceConfig(id="s2", path=str(f2), group="rules", default_category="rules")
        orch = self._make_orchestrator(monkeypatch, [src1, src2])
        files = orch._discover_files(only_source_id="s2")
        assert len(files) == 1
        assert files[0][1].id == "s2"


_NOTE = (
    "# Arkenfeld\n\n"
    "Arkenfeld ist eine Stadt am Rand der Akademie. Sie wird von einem Rat aus "
    "sieben Magiern regiert, die sich jeden Neumond im alten Turm versammeln, um "
    "über Handel, Verteidigung und die Pflege der Bibliothek zu beraten.\n"
)


class FakeVectorStore:
    """Records sync calls; serves pre-seeded {source_id: {source_file: hash}}."""

    def __init__(self, hashes: dict | None = None):
        self.hashes = hashes or {}
        self.calls: list[tuple] = []
        self.upserted: list = []

    def get_content_hashes_for_source(self, source_id):
        return dict(self.hashes.get(source_id, {}))

    def delete_by_source_file(self, source_id, source_file):
        self.calls.append(("delete", source_id, source_file))

    def upsert_chunks(self, chunks):
        self.calls.append(("upsert", len(chunks)))
        self.upserted.extend(chunks)


class TestRunIncremental:
    """Test suite for hash-based incremental sync in IngestionOrchestrator.run()."""

    def _setup(self, tmp_path, monkeypatch):
        from src.config.manager import config_manager
        src = _folder_source(tmp_path, id="vault")
        monkeypatch.setattr(config_manager.settings.ingestion, "sources", [src])
        f = tmp_path / "Arkenfeld.md"
        f.write_text(_NOTE, encoding="utf-8")
        return IngestionOrchestrator(), f

    def test_new_file_is_upserted(self, tmp_path, monkeypatch):
        """Verify a file unknown to the store is parsed, counted as created and upserted."""
        orch, _ = self._setup(tmp_path, monkeypatch)
        vs = FakeVectorStore()
        result = orch.run(vectorstore=vs)
        assert vs.upserted
        assert result.chunks_created == len(vs.upserted)
        assert not [c for c in vs.calls if c[0] == "delete"]

    def test_unchanged_file_is_skipped(self, tmp_path, monkeypatch):
        """Verify a file whose hash matches the store is neither parsed nor re-embedded."""
        orch, f = self._setup(tmp_path, monkeypatch)
        vs = FakeVectorStore({"vault": {"Arkenfeld.md": _compute_content_hash(f)}})
        monkeypatch.setattr(orch.parsers[0], "parse", lambda *a: pytest.fail("parsed unchanged file"))
        result = orch.run(vectorstore=vs)
        assert vs.upserted == []
        assert result.documents_processed == 1
        assert result.chunks_created == result.chunks_updated == result.chunks_deleted == 0

    def test_full_run_reprocesses_unchanged_file(self, tmp_path, monkeypatch):
        """Verify full=True re-chunks an unchanged file (e.g. after chunking settings changed)."""
        orch, f = self._setup(tmp_path, monkeypatch)
        vs = FakeVectorStore({"vault": {"Arkenfeld.md": _compute_content_hash(f)}})
        result = orch.run(vectorstore=vs, full=True)
        assert vs.calls[0] == ("delete", "vault", "Arkenfeld.md")
        assert vs.upserted
        assert result.chunks_updated == len(vs.upserted)

    def test_changed_file_old_chunks_deleted_before_upsert(self, tmp_path, monkeypatch):
        """Verify a changed file drops its old chunks first so a shrunken file leaves no stale tail."""
        orch, _ = self._setup(tmp_path, monkeypatch)
        vs = FakeVectorStore({"vault": {"Arkenfeld.md": "sha256:old"}})
        result = orch.run(vectorstore=vs)
        assert vs.calls[0] == ("delete", "vault", "Arkenfeld.md")
        assert vs.calls[-1][0] == "upsert"
        assert result.chunks_updated == len(vs.upserted) > 0
        assert result.chunks_created == 0

    def test_orphan_file_is_deleted(self, tmp_path, monkeypatch):
        """Verify a file present in the store but gone from disk is removed."""
        orch, f = self._setup(tmp_path, monkeypatch)
        vs = FakeVectorStore({"vault": {
            "Arkenfeld.md": _compute_content_hash(f),
            "Gelöscht.md": "sha256:x",
        }})
        result = orch.run(vectorstore=vs)
        assert ("delete", "vault", "Gelöscht.md") in vs.calls
        assert ("delete", "vault", "Arkenfeld.md") not in vs.calls
        assert result.chunks_deleted == 1

    def test_parse_error_keeps_previous_version(self, tmp_path, monkeypatch):
        """Verify a changed file that fails to parse keeps its old chunks instead of being deleted."""
        orch, _ = self._setup(tmp_path, monkeypatch)
        vs = FakeVectorStore({"vault": {"Arkenfeld.md": "sha256:old"}})

        def _boom(*a):
            raise RuntimeError("broken file")

        monkeypatch.setattr(orch.parsers[0], "parse", _boom)
        result = orch.run(vectorstore=vs)
        assert not [c for c in vs.calls if c[0] == "delete"]
        assert len(result.errors) == 1

    def test_upserts_in_batches_without_splitting_files(self, tmp_path, monkeypatch):
        """Verify chunks are flushed once the batch size is reached, each file's chunks in one batch."""
        import src.ingestion.orchestrator as orch_module
        orch, _ = self._setup(tmp_path, monkeypatch)
        (tmp_path / "Malek.md").write_text(_NOTE.replace("Arkenfeld", "Malek"), encoding="utf-8")
        (tmp_path / "Venthir.md").write_text(_NOTE.replace("Arkenfeld", "Venthir"), encoding="utf-8")
        monkeypatch.setattr(orch_module, "UPSERT_BATCH_CHUNKS", 1)
        vs = FakeVectorStore()
        batches: list[set[str]] = []
        original_upsert = vs.upsert_chunks

        def _record(chunks):
            batches.append({c.source_file for c in chunks})
            original_upsert(chunks)

        vs.upsert_chunks = _record

        result = orch.run(vectorstore=vs)

        assert len(batches) == 3
        assert all(len(files) == 1 for files in batches)
        assert set().union(*batches) == {"Arkenfeld.md", "Malek.md", "Venthir.md"}
        assert result.chunks_created == len(vs.upserted)

    def test_crash_keeps_already_flushed_batches(self, tmp_path, monkeypatch):
        """Verify batches flushed before a failing upsert stay stored instead of being lost with the run."""
        import src.ingestion.orchestrator as orch_module
        orch, _ = self._setup(tmp_path, monkeypatch)
        (tmp_path / "Malek.md").write_text(_NOTE.replace("Arkenfeld", "Malek"), encoding="utf-8")
        monkeypatch.setattr(orch_module, "UPSERT_BATCH_CHUNKS", 1)
        vs = FakeVectorStore()
        original_upsert = vs.upsert_chunks

        def _upsert_then_fail(chunks):
            if vs.upserted:
                raise RuntimeError("ChromaDB gone")
            original_upsert(chunks)

        vs.upsert_chunks = _upsert_then_fail

        with pytest.raises(RuntimeError):
            orch.run(vectorstore=vs)
        assert len({c.source_file for c in vs.upserted}) == 1


class TestSourceBase:
    """Test suite verifying base path resolution for determining file relativity."""

    def test_folder_source_base_is_folder(self, tmp_path):
        """Verify that directory configurations use themselves directly as the root."""
        src = SourceConfig(id="x", path=str(tmp_path), group="lore", default_category="misc")
        assert IngestionOrchestrator._source_base(src) == tmp_path.resolve()

    def test_file_source_base_is_parent(self, tmp_path):
        """Verify that file configurations anchor their relativity to their host folder."""
        f = tmp_path / "x.pdf"
        f.write_text("x")
        src = SourceConfig(id="x", path=str(f), group="rules", default_category="rules")
        assert IngestionOrchestrator._source_base(src) == tmp_path.resolve()


class TestContentHash:
    """Test suite validating document stability checks through SHA256 hashes."""

    def test_hash_format(self, tmp_path):
        """Verify hashed content begins with `sha256:` prefix framing."""
        f = tmp_path / "test.md"
        f.write_text("Content", encoding="utf-8")
        h = _compute_content_hash(f)
        assert h.startswith("sha256:")
        assert len(h) > 10

    def test_different_content_different_hash(self, tmp_path):
        """Verify distinct texts effectively issue unique signatures avoiding collisions."""
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("Content A", encoding="utf-8")
        f2.write_text("Content B", encoding="utf-8")
        assert _compute_content_hash(f1) != _compute_content_hash(f2)


class TestExcludePatterns:
    """Test suite validating standard exclusions (e.g., Obsidian hidden folders)."""

    def test_obsidian_excluded(self):
        """Verify `.obsidian/` metadata content defaults to exclusion."""
        base = Path("/vault")
        f = Path("/vault/.obsidian/config.json")
        assert _is_excluded(f, base, [".obsidian/*"])

    def test_trash_excluded(self):
        """Verify `.trash/` soft-deleted items default to exclusion."""
        base = Path("/vault")
        f = Path("/vault/.trash/deleted.md")
        assert _is_excluded(f, base, [".trash/*"])

    def test_alt_files_excluded(self):
        """Verify suffixed alternating document revisions default to exclusion."""
        base = Path("/vault")
        f = Path("/vault/NPCs/old alt.md")
        assert _is_excluded(f, base, ["*alt.md"])

    def test_normal_file_not_excluded(self):
        """Verify conventional files fallthrough exclusions cleanly."""
        base = Path("/vault")
        f = Path("/vault/NPCs/aldric.md")
        assert not _is_excluded(f, base, [".obsidian/*", ".trash/*"])
