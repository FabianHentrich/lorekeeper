from pathlib import Path

import pytest

from src.ingestion.orchestrator import (
    _get_content_category,
    _compute_content_hash,
    _is_excluded,
)


class TestContentCategory:
    def test_npc_folder(self):
        base = Path("/vault")
        assert _get_content_category(Path("/vault/NPCs/aldric.md"), base) == "npc"

    def test_location_folder(self):
        base = Path("/vault")
        assert _get_content_category(Path("/vault/Orte/arkenfeld.md"), base) == "location"

    def test_story_folder(self):
        base = Path("/vault")
        assert _get_content_category(Path("/vault/Geschichte - Söldner/akt1.md"), base) == "story"

    def test_unknown_folder(self):
        base = Path("/vault")
        assert _get_content_category(Path("/vault/Random/file.md"), base) == "misc"

    def test_root_file(self):
        base = Path("/vault")
        assert _get_content_category(Path("/vault/readme.md"), base) == "misc"


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
