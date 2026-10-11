"""Player / game-master mode for the chat.

Before the chat opens, the user picks a role:

- **Spieler** (player): every question is restricted to the sources marked
  "🎲 Spieler" on the Sources page (``player_visible`` in ``sources.yaml``;
  meant for the rulebook PDFs). Lore, adventures and game-master material are
  never searched, so they can't end up in an answer.
- **Spielleiter** (game master): the whole world, with the usual source filters.

This is a convenience mode, not access control — anyone can pick "Spielleiter"
(admin pages and provider controls stay behind ``LOREKEEPER_ADMIN_PASSWORD``).
The restriction itself is enforced server-side in the Streamlit process: the
browser never talks to the backend, so it cannot drop the filter.
"""
PLAYER = "player"
GAME_MASTER = "game_master"

LABELS = {PLAYER: "🎲 Spieler", GAME_MASTER: "📜 Spielleiter"}


def query_filters(role: str, player_source_ids: list[str],
                  group_filter: dict | None, category_filter: dict | None) -> dict | None:
    """Metadata filters for a chat question in the given role.

    Players get exactly their released sources — the sidebar's group/category
    filters are ignored, so nothing can widen the search. Game masters get the
    sidebar filters (or none). Call only for players with at least one source.
    """
    if role == PLAYER:
        return {"source_id": {"$in": list(player_source_ids)}}
    filters = {}
    if group_filter is not None:
        filters["group"] = group_filter
    if category_filter is not None:
        filters["content_category"] = category_filter
    return filters or None
