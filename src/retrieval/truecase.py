"""Restore the source spelling of lower-cased question words before the search ("überschaden" → "Überschaden").

The embedding model and the reranker are case-sensitive, and German nouns are capitalised in the sources:
the golden set fully lower-cased lost 8 of 149 sections (143 → 135, 14.10.), and every 👎 in the chat was
typed in lower case ("was ist überschaden?" found "Überheilung"). The vocabulary comes from the indexed
chunks: a word the user typed in lower case is replaced by its capitalised form only if the sources write
it that way almost always. Words the user capitalised or wrote in mixed case stay as typed.

Built lazily on the first query and dropped after every ingest (like the BM25 index). Only used internally
for the search — the corrected text is never sent to the client, because the vocabulary also covers
game-master-only chunks and a changed spelling would tell a player that a word exists there.
"""
import logging
import re
import threading
from collections import Counter

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_MIN_LEN = 3            # short words ("an", "im") are function words — never touched
_MIN_COUNT = 2          # a word seen once is no evidence of its spelling
_MIN_CAPITAL_SHARE = 0.8  # "Was" at a sentence start must not turn every "was" into "Was"


def build_vocabulary(texts) -> dict[str, str]:
    """Lower-case word → its capitalised spelling, for words the texts write capitalised ≥ 80 % of the time."""
    forms: dict[str, Counter] = {}
    for text in texts:
        for word in _WORD.findall(text or ""):
            if len(word) >= _MIN_LEN:
                forms.setdefault(word.lower(), Counter())[word] += 1
    vocab: dict[str, str] = {}
    for lower, counts in forms.items():
        total = sum(counts.values())
        # every capitalised form counts as evidence ("Kampf" 450× and "KAMPF" 235× in talent names against
        # "kampf" 42×), the normal spelling wins over all-caps
        capitalised = {f: c for f, c in counts.items() if f[0].isupper()}
        if total < _MIN_COUNT or sum(capitalised.values()) / total < _MIN_CAPITAL_SHARE:
            continue
        normal = {f: c for f, c in capitalised.items() if not f.isupper()} or capitalised
        vocab[lower] = max(normal, key=lambda f: normal[f])
    return vocab


def apply(query: str, vocab: dict[str, str]) -> str:
    """Replace fully lower-case words of ``query`` by their source spelling and start it with a capital, as a
    written question does; words the user capitalised or wrote in mixed case stay as typed."""
    restored = _WORD.sub(lambda m: vocab.get(m.group(0), m.group(0)) if m.group(0).islower() else m.group(0), query)
    stripped = restored.lstrip()
    if stripped[:1].islower():
        lead = len(restored) - len(stripped)
        restored = restored[:lead] + stripped[0].upper() + stripped[1:]
    return restored


class TrueCaser:
    """The vocabulary over the indexed chunks, built once and replaced as a whole.

    Same lifecycle as ``BM25Index``: ``ensure_built`` under a lock (two first queries build once),
    ``invalidate`` after an ingest bumps a generation so a build started before it is not published.
    """

    def __init__(self):
        """Empty until the first ``ensure_built``."""
        self._vocab: dict[str, str] | None = None
        self._generation = 0
        self._lock = threading.Lock()

    def ensure_built(self, vectorstore) -> None:
        """Build the vocabulary from all stored chunks unless a current one exists."""
        if self._vocab is not None:
            return
        with self._lock:
            if self._vocab is not None:
                return
            generation = self._generation
            data = vectorstore.get_all(include=["documents"])
            vocab = build_vocabulary((data or {}).get("documents") or [])
            if generation == self._generation:
                self._vocab = vocab
                logger.info("Truecasing vocabulary built: %d words", len(vocab))

    def invalidate(self) -> None:
        """Drop the vocabulary (the collection changed); the next query rebuilds it."""
        self._generation += 1
        self._vocab = None

    def apply(self, query: str) -> str:
        """``query`` with lower-case words in their source spelling (unchanged until built)."""
        vocab = self._vocab
        return apply(query, vocab) if vocab else query
