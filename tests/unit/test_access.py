import pytest

from src.api.access import GAME_MASTER, PLAYER, NoPlayerSources, effective_filters, player_source_ids
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
