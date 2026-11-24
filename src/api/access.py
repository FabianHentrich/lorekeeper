"""Role-based source access, enforced in the backend.

Which sources players may search is configured centrally (``player_visible`` in
sources.yaml). The UI picks the role; the backend builds the player restriction
itself, so a request can narrow it with its own ``metadata_filters`` but never widen it.
"""
import logging
from typing import Literal

from fastapi import HTTPException

from src.config.manager import SourceConfig

logger = logging.getLogger(__name__)

PLAYER = "player"
GAME_MASTER = "game_master"
Role = Literal["player", "game_master"]


_SCALARS = (str, int, float, bool)
# Only what both ChromaDB *and* the BM25 filter (bm25_index._matches_filter) evaluate the same way:
# BM25 ignores $nin/$gt/… (the condition would silently do nothing) and drops every hit on $or.
_SUPPORTED = "$eq, $ne, $in, $and"


class InvalidFilter(ValueError):
    """metadata_filters outside the supported subset — answered with 422 instead of a 500 (ChromaDB)
    or a silently ignored condition (BM25)."""


def validate_filters(filters: dict | None, _nested: bool = False) -> None:
    """Check a request's ``metadata_filters``: field equality, ``$eq``/``$ne`` with a scalar, ``$in``
    with a non-empty list of one type, ``$and`` with ≥ 2 conditions of exactly one key each.

    Top-level filters may have several keys (the retriever splits them into single conditions);
    conditions inside ``$and`` must have exactly one, as ChromaDB requires.
    """
    if filters is None and not _nested:
        return
    if not isinstance(filters, dict) or not filters:
        raise InvalidFilter("Filter muss ein nicht-leeres Objekt sein")
    if _nested and len(filters) != 1:
        raise InvalidFilter(f"Jede Bedingung in $and braucht genau einen Schlüssel: {filters!r}")
    for key, value in filters.items():
        if key == "$and":
            if not isinstance(value, list) or len(value) < 2:
                raise InvalidFilter("$and braucht eine Liste mit mindestens zwei Bedingungen")
            for cond in value:
                validate_filters(cond, _nested=True)
        elif key.startswith("$"):
            raise InvalidFilter(f"Operator {key} wird nicht unterstützt (erlaubt: {_SUPPORTED})")
        elif isinstance(value, _SCALARS):
            continue
        elif isinstance(value, dict) and len(value) == 1:
            op, operand = next(iter(value.items()))
            if op in ("$eq", "$ne") and isinstance(operand, _SCALARS):
                continue
            if (op == "$in" and isinstance(operand, list) and operand
                    and all(isinstance(x, _SCALARS) for x in operand)
                    and len({type(x) for x in operand}) == 1):
                continue
            raise InvalidFilter(f"Ungültige Bedingung für {key}: {value!r} (erlaubt: {_SUPPORTED})")
        else:
            raise InvalidFilter(f"Ungültiger Wert für {key}: {value!r}")


class NoPlayerSources(Exception):
    """Player request while no source is released for players — refuse instead of searching unfiltered."""


def player_source_ids(sources: list[SourceConfig]) -> list[str]:
    """Ids of the sources released for players."""
    return [s.id for s in sources if s.player_visible]


def effective_filters(role: Role, sources: list[SourceConfig], requested: dict | None) -> dict | None:
    """The metadata filter a request may use.

    Raises ``InvalidFilter`` for filters outside the supported subset. Game masters get their own filter
    unchanged. Players get the restriction to the released
    sources, AND-ed with whatever they sent — each requested key becomes its own condition,
    so a ``source_id`` in the request can only narrow the released set.
    """
    validate_filters(requested)
    if role != PLAYER:
        return requested
    ids = player_source_ids(sources)
    if not ids:
        raise NoPlayerSources()
    restriction = {"source_id": {"$in": ids}}
    if not requested:
        return restriction
    return {"$and": [restriction] + [{k: v} for k, v in requested.items()]}


def request_filters(role: Role, sources: list[SourceConfig], requested: dict | None, what: str) -> dict | None:
    """``effective_filters`` for a route: 403 when no source is released for players, 422 for an invalid
    filter, each refusal logged with the role and ``what`` ("query", "preview"). The one place every
    retrieval route maps these errors — a copy per route could drift on the player path."""
    try:
        return effective_filters(role, sources, requested)
    except NoPlayerSources:
        logger.warning("Refused %s %s: no source is released for players", role, what)
        raise HTTPException(status_code=403, detail="Keine Quelle für Spieler freigegeben") from None
    except InvalidFilter as e:
        logger.warning("Refused %s %s: invalid metadata_filters (%s)", role, what, e)
        raise HTTPException(status_code=422, detail=f"Ungültiger metadata_filters: {e}") from None
