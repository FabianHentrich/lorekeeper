"""Player / game-master mode in the chat (ui/roles.py, ui/LoreKeeper.py)."""
import roles

from tests.ui.conftest import open_page

PLAYER_FILTER = {"source_id": {"$in": ["regelbuch"]}}


def _ask(at, question="Was ist eine Marke?"):
    at.chat_input[0].set_value(question).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_role_picker_comes_first(fake_backend):
    """Without a role there is no chat — just the two buttons, and nothing is loaded yet."""
    at = open_page("chat", role=None).run()

    labels = [b.label for b in at.button]
    assert roles.LABELS[roles.PLAYER] in labels and roles.LABELS[roles.GAME_MASTER] in labels
    assert len(at.chat_input) == 0
    assert fake_backend.calls == []


def test_picking_player_opens_restricted_chat(fake_backend):
    at = open_page("chat", role=None).run()
    at.button(key="_role_player").click().run()

    assert at.session_state["_role"] == roles.PLAYER
    assert len(at.chat_input) == 1
    assert not [c for c in at.sidebar.checkbox if c.label == "🗺️ Lore"]   # no filter widgets for players


def test_player_question_is_limited_to_released_sources(fake_backend):
    _ask(open_page("chat", role=roles.PLAYER).run())

    assert fake_backend.bodies["POST /query/stream"][-1]["metadata_filters"] == PLAYER_FILTER


def test_game_master_question_is_unrestricted(fake_backend):
    _ask(open_page("chat", role=roles.GAME_MASTER).run())

    assert "metadata_filters" not in fake_backend.bodies["POST /query/stream"][-1]


def test_switching_role_starts_a_new_session(fake_backend):
    at = _ask(open_page("chat", role=roles.GAME_MASTER).run())
    assert at.session_state["session_id"] == "s1"

    switch = next(b for b in at.sidebar.button if b.label == "🔄 Rolle wechseln")
    switch.click().run()

    assert "_role" not in at.session_state
    assert at.session_state["messages"] == []
    assert at.session_state["session_id"] is None
    assert len(at.chat_input) == 0   # back at the picker


def test_player_without_released_sources_cannot_ask(fake_backend, monkeypatch):
    from tests.ui import conftest
    monkeypatch.setitem(conftest.PAYLOADS["/sidebar-state"], "player_source_ids", [])

    at = open_page("chat", role=roles.PLAYER).run()

    assert at.chat_input[0].disabled
    assert any("keine Quelle freigegeben" in w.value for w in at.warning)


class TestQueryFilters:
    """Pure filter logic in ui/roles.py."""

    def test_player_gets_exactly_the_released_sources(self):
        assert roles.query_filters(roles.PLAYER, ["a", "b"], None, None) == {"source_id": {"$in": ["a", "b"]}}

    def test_player_ignores_sidebar_filters(self):
        """Leftover game-master filters (session state) must not widen or replace the player filter."""
        assert roles.query_filters(roles.PLAYER, ["a"], {"$in": ["lore"]}, {"$in": ["npc"]}) == \
            {"source_id": {"$in": ["a"]}}

    def test_game_master_uses_sidebar_filters(self):
        assert roles.query_filters(roles.GAME_MASTER, ["a"], {"$in": ["lore"]}, None) == {"group": {"$in": ["lore"]}}
        assert roles.query_filters(roles.GAME_MASTER, ["a"], None, None) is None


def test_game_master_can_exclude_simulation(fake_backend):
    """Unticking 📊 Simulation limits the search to the other three groups."""
    at = open_page("chat", role=roles.GAME_MASTER).run()
    next(c for c in at.sidebar.checkbox if c.label == "📊 Simulation").uncheck().run()
    _ask(at)

    assert fake_backend.bodies["POST /query/stream"][-1]["metadata_filters"] == \
        {"group": {"$in": ["lore", "adventure", "rules"]}}
