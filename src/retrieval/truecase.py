"""Restore the source spelling of lower-cased question words before the search ("überschaden" → "Überschaden").

The embedding model and the reranker are case-sensitive, and German nouns are capitalised in the sources:
the golden set fully lower-cased lost 8 of 149 sections (143 → 135, 14.10.), and every 👎 in the chat was
typed in lower case ("was ist überschaden?" found "Überheilung"). The vocabulary comes from the indexed
chunks: a word the user typed in lower case is replaced by its capitalised form only if the sources write
it that way almost always. Words the user capitalised or wrote in mixed case stay as typed.

Before that, typos are repaired (roadmap 3.15, 17.10.): a question word that occurs nowhere in the sources
is replaced by the one capitalised source word (a noun from the vocabulary above) within edit distance 1, or
2 from 10 letters on ("üebrschaden" → "Überschaden"; adjacent swaps count as one edit). Guard rails, each
from a false correction in the golden-set measurement: same first letter ("bewirken" → "Bezirke"), no
correction to a word that starts the other (inflections: "Decknamen"/"Deckname"), no correction when two
candidates are about equally frequent. Measured: clean golden set unchanged, one-typo golden set loses
nothing anymore (7 → 0 changed hits); a typo in the first letter stays as typed.

Built lazily on the first query and dropped after every ingest (like the BM25 index). Only used internally
for the search — the corrected text is never sent to the client, because the vocabulary also covers
game-master-only chunks and a changed spelling would tell a player that a word exists there
(tests/integration/test_player_mode_security.py, mutation-checked). Accepted residual (review 13.5): a
player who already guesses an exact game-master word can confirm it — a typo of it, or its lower-case form,
gets the same player-visible sources and scores as the word itself. Nothing is shown that wasn't guessed;
a second, player-only vocabulary would close it (~0.5 s more build time).
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
_TYPO_MIN_LEN = 5       # shorter words have too many neighbours at distance 1
_TYPO_TARGET_MIN_COUNT = 3  # a correction target must be an established source word
_TYPO_LONG = 10         # from this length two edits are allowed (from 8 letters: "zeichnet" → "Zeichen")
_TYPO_CLEAR_LEAD = 3    # among equally close candidates the most frequent wins only with 3× the count


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


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Optimal string alignment distance (insert, delete, replace, adjacent swap = 1); ``limit + 1`` as soon
    as it is certain to exceed ``limit``."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    before: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], before[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        before, prev = prev, cur
    return prev[-1]


def _deletes(word: str, depth: int) -> set[str]:
    """The word and every form with up to ``depth`` letters removed — two words within distance ``depth``
    share at least one of them (symmetric-delete lookup, no scan over the whole vocabulary)."""
    out = frontier = {word}
    for _ in range(depth):
        frontier = {w[:i] + w[i + 1:] for w in frontier for i in range(len(w))}
        out = out | frontier
    return out


class TypoIndex:
    """Every source word (to know what is no typo) and a delete index over the correction targets."""

    def __init__(self, texts, vocab: dict[str, str]):
        """``vocab`` is ``build_vocabulary``'s result: only its capitalised source words are targets."""
        self.counts = Counter(w.lower() for text in texts for w in _WORD.findall(text or ""))
        self.vocab = vocab
        self.index: dict[str, set[str]] = {}
        for word, count in self.counts.items():
            if count >= _TYPO_TARGET_MIN_COUNT and word in vocab:
                for key in _deletes(word, 2 if len(word) >= _TYPO_LONG else 1):
                    self.index.setdefault(key, set()).add(word)

    def fix(self, word: str) -> str | None:
        """The source spelling ``word`` was probably meant as, or None (known word, no clear candidate)."""
        low = word.lower()
        if len(low) < _TYPO_MIN_LEN or low in self.counts:
            return None
        limit = 2 if len(low) >= _TYPO_LONG else 1
        candidates = {c for key in _deletes(low, limit) for c in self.index.get(key, ())
                      if c[0] == low[0] and not c.startswith(low) and not low.startswith(c)}
        ranked = sorted((d, -self.counts[c], c) for c in candidates if (d := _edit_distance(low, c, limit)) <= limit)
        if not ranked:
            return None
        best = ranked[0]
        tied = [r for r in ranked[1:] if r[0] == best[0]]
        if tied and -best[1] < _TYPO_CLEAR_LEAD * -tied[0][1]:
            return None
        return self.vocab[best[2]]

    def apply(self, query: str) -> str:
        """``query`` with every typo replaced by the source spelling of the word it was meant as."""
        return _WORD.sub(lambda m: self.fix(m.group(0)) or m.group(0), query)


class TrueCaser:
    """The vocabulary over the indexed chunks, built once and replaced as a whole.

    Same lifecycle as ``BM25Index``: ``ensure_built`` under a lock (two first queries build once),
    ``invalidate`` after an ingest bumps a generation so a build started before it is not published.
    """

    def __init__(self):
        """Empty until the first ``ensure_built``."""
        # Vocabulary and typo index are published together — a query never sees one without the other
        self._built: tuple[dict[str, str], TypoIndex] | None = None
        self._generation = 0
        self._lock = threading.Lock()

    def ensure_built(self, vectorstore) -> None:
        """Build the vocabulary and the typo index from all stored chunks unless current ones exist."""
        if self._built is not None:
            return
        with self._lock:
            if self._built is not None:
                return
            generation = self._generation
            data = vectorstore.get_all(include=["documents"])
            texts = (data or {}).get("documents") or []
            vocab = build_vocabulary(texts)
            typos = TypoIndex(texts, vocab)
            if generation == self._generation:
                self._built = (vocab, typos)
                logger.info("Truecasing vocabulary built: %d words, %d typo keys", len(vocab), len(typos.index))

    def invalidate(self) -> None:
        """Drop the vocabulary (the collection changed); the next query rebuilds it."""
        self._generation += 1
        self._built = None

    def apply(self, query: str) -> str:
        """``query`` with typos repaired and lower-case words in their source spelling (unchanged until built)."""
        built = self._built
        if not built:
            return query
        vocab, typos = built
        return apply(typos.apply(query), vocab)
