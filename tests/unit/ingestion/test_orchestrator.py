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
