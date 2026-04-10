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
    def test_folder_source_top_folder_match(self, tmp_path):
        (tmp_path / "NPCs").mkdir()
        f = tmp_path / "NPCs" / "aldric.md"
        f.write_text("x")
        src = _folder_source(tmp_path)
        cat, grp = _resolve_category(f, src)
        assert cat == "npc"
        assert grp == "lore"

    def test_folder_source_case_insensitive(self, tmp_path):
        (tmp_path / "orte").mkdir()
        f = tmp_path / "orte" / "arkenfeld.md"
        f.write_text("x")
        src = _folder_source(tmp_path)
        cat, grp = _resolve_category(f, src)
        assert cat == "location"
        assert grp == "lore"

    def test_folder_source_unknown_top_folder_falls_back_to_default(self, tmp_path):
        (tmp_path / "Random").mkdir()
        f = tmp_path / "Random" / "x.md"
        f.write_text("x")
        src = _folder_source(tmp_path)
        cat, grp = _resolve_category(f, src)
        assert cat == "misc"
        assert grp == "lore"

    def test_folder_source_root_file_falls_back_to_default(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("x")
        src = _folder_source(tmp_path)
        cat, grp = _resolve_category(f, src)
        assert cat == "misc"
        assert grp == "lore"

    def test_file_source_always_uses_default_category(self, tmp_path):
        f = tmp_path / "rulebook.pdf"
        f.write_text("x")
        src = SourceConfig(id="rb", path=str(f), group="rules", default_category="rules")
        cat, grp = _resolve_category(f, src)
        assert cat == "rules"
        assert grp == "rules"

    def test_dict_entry_overrides_group(self, tmp_path):
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
    def _make_orchestrator(self, monkeypatch, sources):
        from src.config.manager import config_manager
        monkeypatch.setattr(config_manager.settings.ingestion, "sources", sources)
        return IngestionOrchestrator()

    def test_file_source_yields_single_file(self, tmp_path, monkeypatch):
        f = tmp_path / "rulebook.pdf"
        f.write_text("x")
        src = SourceConfig(id="rb", path=str(f), group="rules", default_category="rules")
        orch = self._make_orchestrator(monkeypatch, [src])
        files = orch._discover_files()
        assert len(files) == 1
        assert files[0][0] == f.resolve()
        assert files[0][1].id == "rb"

    def test_folder_source_walks_recursively(self, tmp_path, monkeypatch):
        (tmp_path / "NPCs").mkdir()
        (tmp_path / "NPCs" / "a.md").write_text("x")
        (tmp_path / "NPCs" / "b.md").write_text("x")
        src = _folder_source(tmp_path)
        orch = self._make_orchestrator(monkeypatch, [src])
        files = orch._discover_files()
        assert len(files) == 2

    def test_only_source_id_filter(self, tmp_path, monkeypatch):
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
    def test_folder_source_base_is_folder(self, tmp_path):
        src = SourceConfig(id="x", path=str(tmp_path), group="lore", default_category="misc")
        assert IngestionOrchestrator._source_base(src) == tmp_path.resolve()

    def test_file_source_base_is_parent(self, tmp_path):
        f = tmp_path / "x.pdf"
        f.write_text("x")
        src = SourceConfig(id="x", path=str(f), group="rules", default_category="rules")
        assert IngestionOrchestrator._source_base(src) == tmp_path.resolve()


class TestContentHash:
    def test_hash_format(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("Content", encoding="utf-8")
        h = _compute_content_hash(f)
        assert h.startswith("sha256:")
        assert len(h) > 10

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("Content A", encoding="utf-8")
        f2.write_text("Content B", encoding="utf-8")
        assert _compute_content_hash(f1) != _compute_content_hash(f2)


class TestExcludePatterns:
    def test_obsidian_excluded(self):
        base = Path("/vault")
        f = Path("/vault/.obsidian/config.json")
        assert _is_excluded(f, base, [".obsidian/*"])

    def test_trash_excluded(self):
        base = Path("/vault")
        f = Path("/vault/.trash/deleted.md")
        assert _is_excluded(f, base, [".trash/*"])

    def test_alt_files_excluded(self):
        base = Path("/vault")
        f = Path("/vault/NPCs/old alt.md")
        assert _is_excluded(f, base, ["*alt.md"])

    def test_normal_file_not_excluded(self):
        base = Path("/vault")
        f = Path("/vault/NPCs/aldric.md")
        assert not _is_excluded(f, base, [".obsidian/*", ".trash/*"])
