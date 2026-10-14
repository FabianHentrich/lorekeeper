"""Truecasing of lower-cased questions before the search (roadmap 3.15/👎 14.10.)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.manager import RerankingConfig, RetrievalConfig
from src.retrieval.retriever import Retriever
from src.retrieval.truecase import TrueCaser, apply, build_vocabulary

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
