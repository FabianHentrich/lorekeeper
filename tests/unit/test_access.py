import pytest

from src.api.access import (
    GAME_MASTER,
    PLAYER,
    InvalidFilter,
    NoPlayerSources,
    effective_filters,
    player_source_ids,
    validate_filters,
)
from src.config.manager import SourceConfig

SOURCES = [
    SourceConfig(id="regeln", path="C:/r.pdf", group="rules", player_visible=True),
    SourceConfig(id="welt", path="C:/vault", group="lore"),
    SourceConfig(id="sim", path="C:/s.md", group="simulation", player_visible=True),
]
RELEASED = {"source_id": {"$in": ["regeln", "sim"]}}


class TestPlayerSourceIds:
    def test_only_released_sources(self):
        assert player_source_ids(SOURCES) == ["regeln", "sim"]


class TestEffectiveFilters:
    def test_game_master_filter_is_passed_through(self):
        assert effective_filters(GAME_MASTER, SOURCES, None) is None
        assert effective_filters(GAME_MASTER, SOURCES, {"group": {"$in": ["lore"]}}) == {"group": {"$in": ["lore"]}}

    def test_player_without_own_filter_gets_the_released_sources(self):
        assert effective_filters(PLAYER, SOURCES, None) == RELEASED

    def test_player_filter_is_anded_so_it_can_only_narrow(self):
        """A source_id in the request becomes its own condition next to the restriction — never a replacement."""
        sneaky = {"source_id": {"$in": ["welt"]}, "group": "lore"}
        assert effective_filters(PLAYER, SOURCES, sneaky) == {"$and": [
            RELEASED, {"source_id": {"$in": ["welt"]}}, {"group": "lore"},
        ]}

    def test_player_without_released_sources_is_refused(self):
        with pytest.raises(NoPlayerSources):
            effective_filters(PLAYER, [SOURCES[1]], None)


class TestValidateFilters:
    """Only filters ChromaDB and BM25 both evaluate are let through (422 instead of a 500 or a
    silently ignored condition)."""

    @pytest.mark.parametrize("ok", [
        None,
        {"group": "rules"},
        {"source_id": {"$in": ["a", "b"]}},
        {"chunk_index": {"$eq": 2}},
        {"group": {"$in": ["lore", "rules"]}, "content_category": {"$in": ["npc"]}},   # the UI's game-master filter
        {"$and": [{"group": "lore"}, {"source_id": {"$ne": "x"}}]},
        {"$and": [{"group": "lore"}, {"$and": [{"source_id": "a"}, {"chunk_index": 1}]}]},
    ])
    def test_valid(self, ok):
        validate_filters(ok)

    @pytest.mark.parametrize("bad", [
        {},
        {"$and": []},
        {"$and": [{"group": "lore"}]},
        {"source_id": None},
        {"source_id": ["a"]},
        {"source_id": {"$foo": 1}},
        {"source_id": {"$in": []}},
        {"source_id": {"$in": "a"}},
        {"source_id": {"$eq": 1, "$ne": 2}},
        {"$not": {"group": "lore"}},
        {"$or": [{"group": "lore"}, {"group": "rules"}]},              # BM25 would drop every hit
        {"group": {"$nin": ["simulation"]}},                            # BM25 would ignore it
        {"chunk_index": {"$gte": 2}},                                   # BM25 would ignore it
        {"chunk_index": {"$gt": "5"}},
        {"source_id": {"$in": ["a", 1]}},                               # ChromaDB: one type per list
        {"source_id": {"$contains": "a"}},
        {"$and": [{"group": "lore", "source_id": "a"}, {"group": "x"}]},  # ChromaDB: one key per condition
        {"$and": [None, {"group": "x"}]},
    ])
    def test_invalid(self, bad):
        with pytest.raises(InvalidFilter):
            validate_filters(bad)

    def test_checked_for_both_roles(self):
        sources = [SourceConfig(id="regeln", path="x", group="rules", player_visible=True)]
        for role in (PLAYER, GAME_MASTER):
            with pytest.raises(InvalidFilter):
                effective_filters(role, sources, {"$and": []})
