"""Truecasing of lower-cased questions before the search (roadmap 3.15/👎 14.10.)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.manager import RerankingConfig, RetrievalConfig
from src.retrieval.retriever import Retriever
from src.retrieval.truecase import TrueCaser, TypoIndex, apply, build_vocabulary

TEXTS = [
    "Überschaden führt zu einem Todeszähler. Überschaden entsteht, wenn …",
    "Was ist Kampf? Der Wert KAMPF steigt. Im Kampf zählt die Wurfzahl. KAMPF Rang 3.",
    "was du willst, Was immer",          # 'was' as often lower as capitalised → never changed
    "Einmal genannt: Drachenkönig",      # seen once → no evidence
    "Im Gasthaus, im Wald",              # short words are left alone
    "nur kampf",
]


class TestVocabulary:
    def test_capitalised_words_map_to_their_source_spelling(self):
        vocab = build_vocabulary(TEXTS)
        assert vocab["überschaden"] == "Überschaden"

    def test_all_caps_counts_as_capital_evidence_but_the_normal_spelling_wins(self):
        """'Kampf' 2× + 'KAMPF' 2× against 'kampf' 1× — the all-caps talent name must not be the result."""
        assert build_vocabulary(TEXTS)["kampf"] == "Kampf"

    @pytest.mark.parametrize("word", ["was", "drachenkönig", "im"])
    def test_no_evidence_no_change(self, word):
        assert word not in build_vocabulary(TEXTS)


class TestApply:
    VOCAB = {"überschaden": "Überschaden", "kampf": "Kampf"}

    def test_lower_case_words_get_the_source_spelling_and_the_question_a_capital(self):
        assert apply("was ist überschaden?", self.VOCAB) == "Was ist Überschaden?"

    def test_words_the_user_capitalised_or_shouted_stay(self):
        assert apply("Wie läuft der KAMPF mit Überschaden?", self.VOCAB) == "Wie läuft der KAMPF mit Überschaden?"

    def test_unknown_words_and_leading_space_are_kept(self):
        assert apply("  regeln im kampf", self.VOCAB) == "  Regeln im Kampf"


class TestTypos:
    """Typo repair toward capitalised source words (roadmap 3.15, 17.10.) — one test per guard rail."""

    SOURCE = ["Überschaden Überschaden Überschaden Überheilung",
              "Deckname Deckname Deckname Bezirke Bezirke Bezirke",
              "Lebenspunkte Lebenspunkte Lebenspunkte Lebenspunkte",
              "Wunde Wunde Wunde Runde Runde Runde",
              "bewirken zeigt"]

    def _index(self):
        return TypoIndex(self.SOURCE, build_vocabulary(self.SOURCE))

    def test_the_chat_typo_is_repaired(self):
        """Adjacent swap = one edit: the 👎 case from 14.10."""
        assert self._index().apply("was ist üebrschaden?") == "was ist Überschaden?"

    def test_two_edits_only_for_long_words(self):
        idx = self._index()
        assert idx.apply("Lbeenspunktte") == "Lebenspunkte"     # 13 letters, two edits
        assert idx.apply("Dkcename") == "Dkcename"             # 8 letters, two edits — too far

    def test_a_word_from_the_sources_is_no_typo(self):
        """'Rinde' occurs once — rare, but real; one edit from the frequent 'Runde' it must stay."""
        source = self.SOURCE + ["Rinde"]
        assert TypoIndex(source, build_vocabulary(source)).fix("Rinde") is None

    def test_the_first_letter_must_match(self):
        assert self._index().fix("Gezirke") is None

    def test_inflections_are_not_corrected(self):
        """'Decknamen' is a form of 'Deckname', not a typo of it."""
        assert self._index().fix("Decknamen") is None

    def test_equally_likely_candidates_leave_the_word_alone(self):
        """'Wunde' and 'Wurde' are both one edit from 'Wuxde' and equally frequent — no guess."""
        source = self.SOURCE + ["Wurde Wurde Wurde"]
        idx = TypoIndex(source, build_vocabulary(source))
        assert idx.fix("Wuxde") is None
        clear = self.SOURCE + ["Wurde Wurde Wurde"] + ["Wunde"] * 9      # 12× against 3× — clear lead
        assert TypoIndex(clear, build_vocabulary(clear)).fix("Wuxde") == "Wunde"

    def test_short_words_are_never_touched(self):
        assert self._index().fix("Rnde") is None               # one edit from "Runde", but only 4 letters


class TestTrueCaser:
    def _store(self, *docs):
        vs = MagicMock()
        vs.get_all.return_value = {"documents": list(docs)}
        return vs

    def test_builds_once_and_again_after_invalidate(self):
        vs = self._store("Überschaden, Überschaden")
        tc = TrueCaser()
        assert tc.apply("überschaden") == "überschaden"   # not built yet → unchanged
        tc.ensure_built(vs)
        tc.ensure_built(vs)
        assert vs.get_all.call_count == 1
        assert tc.apply("überschaden") == "Überschaden"
        tc.invalidate()                                     # after an ingest
        tc.ensure_built(vs)
        assert vs.get_all.call_count == 2

    def test_a_build_invalidated_while_running_is_not_published(self):
        """An ingest that finishes during the build would otherwise leave the old vocabulary in place."""
        tc = TrueCaser()
        vs = MagicMock()

        def _get_all(include):
            tc.invalidate()
            return {"documents": ["Überschaden, Überschaden"]}

        vs.get_all.side_effect = _get_all
        tc.ensure_built(vs)
        assert tc.apply("überschaden") == "überschaden"


@pytest.mark.asyncio
async def test_retriever_searches_with_the_source_spelling():
    """'was ist überschaden?' found 'Überheilung' — the embedding must get 'Was ist Überschaden?'."""
    vs = MagicMock()
    vs.get_all.return_value = {"documents": ["Überschaden ist …", "Mehr zu Überschaden"]}
    vs.query.return_value = []
    emb = AsyncMock()
    emb.embed_text.return_value = [0.1, 0.2, 0.3]
    ret = Retriever(config=RetrievalConfig(reranking=RerankingConfig(enabled=False)), embedding_service=emb,
                    vectorstore=vs)
    await ret.retrieve("was ist überschaden?")
    emb.embed_text.assert_awaited_once_with("Was ist Überschaden?")
