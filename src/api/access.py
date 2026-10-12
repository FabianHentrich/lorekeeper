"""Role-based source access, enforced in the backend.

Which sources players may search is configured centrally (``player_visible`` in
sources.yaml). The UI picks the role; the backend builds the player restriction
itself, so a request can narrow it with its own ``metadata_filters`` but never widen it.
"""
from typing import Literal

from src.config.manager import SourceConfig

PLAYER = "player"
GAME_MASTER = "game_master"
Role = Literal["player", "game_master"]


class NoPlayerSources(Exception):
    """Player request while no source is released for players — refuse instead of searching unfiltered."""


def player_source_ids(sources: list[SourceConfig]) -> list[str]:
    """Ids of the sources released for players."""
    return [s.id for s in sources if s.player_visible]


def effective_filters(role: Role, sources: list[SourceConfig], requested: dict | None) -> dict | None:
    """The metadata filter a request may use.

    Game masters get their own filter unchanged. Players get the restriction to the released
    sources, AND-ed with whatever they sent — each requested key becomes its own condition,
    so a ``source_id`` in the request can only narrow the released set.
    """
    if role != PLAYER:
        return requested
    ids = player_source_ids(sources)
    if not ids:
        raise NoPlayerSources()
    restriction = {"source_id": {"$in": ids}}
    if not requested:
        return restriction
    return {"$and": [restriction] + [{k: v} for k, v in requested.items()]}
