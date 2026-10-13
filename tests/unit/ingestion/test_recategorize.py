from unittest.mock import MagicMock

from src.config.manager import SourceConfig
from src.ingestion.recategorize import _match_source, recategorize


def _make_source(**kwargs):
    """
    Helper function to generate a mocked SourceConfig template.
    """
    defaults = {"id": "src1", "path": "/vault", "group": "lore", "default_category": "misc"}
    defaults.update(kwargs)
    return SourceConfig(**defaults)


class TestMatchSource:
    """Test suite validating heuristics for re-identifying detached metadata chunks against active sources."""

    def test_match_by_source_id(self):
        """Verify chunks carrying a valid `source_id` directly pin to the matching SourceConfig."""
        s = _make_source(id="my-vault")
        result = _match_source({"source_id": "my-vault"}, [s])
        assert result is s

    def test_match_by_source_id_unknown(self):
        """Verify mismatched explicit `source_id` gracefully returns None rather than exploding."""
        s = _make_source(id="my-vault")
        result = _match_source({"source_id": "other"}, [s])
        assert result is None

    def test_no_match_returns_none(self):
        """Chunks without (or with an unknown) source_id stay unmatched — no guessing by path."""
        s = _make_source(id="v")
        assert _match_source({}, [s]) is None
        assert _match_source({"source_file": "NPCs/aldric.md"}, [s]) is None


class TestRecategorize:
    """Test suite validating batch ChromaDB structural updates based on source configuration mutations."""

    def _make_vs(self, ids, metadatas):
        """
        Helper to cleanly mock a vectorstore collection and its document retrieval interfaces.
        """
        vs = MagicMock()
        collection = MagicMock()
        vs._get_collection.return_value = collection
        collection.get.return_value = {"ids": ids, "metadatas": metadatas}
        return vs

    def test_updates_changed_metadata(self, tmp_path):
        """
        Verify that if current chunk metadata diverges from the active yaml configuration,
        an update batch is securely triggered altering the vectorstore contents.
        """
        folder = tmp_path / "vault"
        folder.mkdir()
        (folder / "NPCs").mkdir()
        (folder / "NPCs" / "aldric.md").write_text("x")

        source = SourceConfig(
            id="v", path=str(folder), group="lore",
            default_category="misc", category_map={"NPCs": "npc"},
        )

        config = MagicMock()
        config.settings.ingestion.sources = [source]

        vs = self._make_vs(
            ids=["chunk1"],
            metadatas=[{
                "source_id": "v",
                "source_file": "NPCs/aldric.md",
                "content_category": "old_cat",
                "group": "old_group",
            }],
        )

        result = recategorize(config=config, vectorstore=vs)
        assert result["chunks_updated"] == 1
        assert result["chunks_skipped"] == 0
        vs.update_metadata_batch.assert_called_once()
        updated_meta = vs.update_metadata_batch.call_args[0][1][0]
        assert updated_meta["content_category"] == "npc"
        assert updated_meta["group"] == "lore"
        assert updated_meta["source_id"] == "v"

    def test_skips_already_correct(self, tmp_path):
        """
        Verify that chunks already aligned perfectly with current configuration maps
        do not silently accrue meaningless update cycles against the vector database.
        """
        folder = tmp_path / "vault"
        folder.mkdir()
        (folder / "NPCs").mkdir()
        (folder / "NPCs" / "aldric.md").write_text("x")

        source = SourceConfig(
            id="v", path=str(folder), group="lore",
            default_category="misc", category_map={"NPCs": "npc"},
        )
        config = MagicMock()
        config.settings.ingestion.sources = [source]

        vs = self._make_vs(
            ids=["chunk1"],
            metadatas=[{
                "source_id": "v",
                "source_file": "NPCs/aldric.md",
                "content_category": "npc",
                "group": "lore",
            }],
        )

        result = recategorize(config=config, vectorstore=vs)
        assert result["chunks_skipped"] == 1
        assert result["chunks_updated"] == 0
        vs.update_metadata_batch.assert_not_called()

    def test_unmatched_chunks_counted(self):
        """
        Verify that chunks permanently lost from the active workspace mapping
        increment purely as unmatched statistics.
        """
        config = MagicMock()
        config.settings.ingestion.sources = []

        vs = self._make_vs(
            ids=["orphan1"],
            metadatas=[{"source_id": "gone", "source_file": "x.md"}],
        )

        result = recategorize(config=config, vectorstore=vs)
        assert result["chunks_unmatched"] == 1
        assert result["chunks_updated"] == 0

    def test_dict_category_map_overrides_group(self, tmp_path):
        """
        Verify complex mapping structures assigning nested `group` keys accurately
        translate these overwrites into the final vectorstore metadata.
        """
        folder = tmp_path / "vault"
        folder.mkdir()
        (folder / "Geschichte").mkdir()
        (folder / "Geschichte" / "akt1.md").write_text("x")

        source = SourceConfig(
            id="v", path=str(folder), group="lore",
            default_category="misc",
            category_map={"Geschichte": {"category": "story", "group": "adventure"}},
        )
        config = MagicMock()
        config.settings.ingestion.sources = [source]

        vs = self._make_vs(
            ids=["chunk1"],
            metadatas=[{
                "source_id": "v",
                "source_file": "Geschichte/akt1.md",
                "content_category": "story",
                "group": "lore",  # old: was lore, should become adventure
            }],
        )

        result = recategorize(config=config, vectorstore=vs)
        assert result["chunks_updated"] == 1
        updated_meta = vs.update_metadata_batch.call_args[0][1][0]
        assert updated_meta["group"] == "adventure"
        assert updated_meta["content_category"] == "story"
