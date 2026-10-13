import asyncio
import json
from types import SimpleNamespace

import pytest

from evaluation.evaluate_retrieval import (
    load_qa_pairs,
    run_evaluation_with_retriever,
    save_report,
    section_hit,
    source_hit,
)
from src.config.manager import RetrievalConfig


class TestSectionHit:
    """Matching retrieved chunk headings against a numbered rulebook section."""

    def test_exact_section_matches(self):
        """The section's own heading counts as a hit."""
        assert section_hit("4.10", ["**4.10 Kampfunfähigkeit und Tod** > **Kampfunfähigkeit**"])

    def test_subsection_matches_parent(self):
        """A sub-section lies inside its parent section."""
        assert section_hit("5.4", ["**5.4.1 Krieger** > **Wuchtschlag**"])

    def test_prefix_of_other_number_does_not_match(self):
        """'4.1' must not match '4.10' and vice versa."""
        assert not section_hit("4.1", ["**4.10 Kampfunfähigkeit und Tod**"])
        assert not section_hit("4.10", ["**4.1 Der Ablauf eines Kampfes**"])

    def test_neighbouring_section_misses(self):
        """The Überschaden/Überheilung confusion is a section miss."""
        assert not section_hit("4.10", ["**4.11 Kampfende** > **Überheilung**", "Orte > Arkenfeld"])

    def test_empty_headings(self):
        """No chunks → no hit."""
        assert not section_hit("2.5", [])

    def test_unnumbered_heading_matches_by_name(self):
        """Simulation reports have named headings: exact name or name plus suffix."""
        headings = ["PnP Begegnungsbericht (für LLM) > Begegnungsbaukasten"]
        assert section_hit("Begegnungsbaukasten", headings)
        assert section_hit("Gesamtwertung", ["Bericht > Gesamtwertung (R3 vs Median aus 6 Profilen)"])
        assert not section_hit("Begegnungsbau", headings)


class _FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.config = RetrievalConfig()
        self.calls: list[dict] = []

    async def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self.chunks


def _chunk(source_file, source_id, heading=""):
    return SimpleNamespace(source_file=source_file, heading=heading, score=0.9,
                           metadata={"source_id": source_id})


def _hit(pair, chunks) -> bool:
    report = asyncio.run(run_evaluation_with_retriever([pair], _FakeRetriever(chunks), 15, 8))
    return report["details"][0]["hit"]


class TestExpectedSourceIds:
    """Equally named files (two simulation llm_report.md) are told apart by source id."""

    PAIR = {"id": "s1", "question": "?", "expected_sources": ["llm_report.md"]}

    def test_file_name_alone_matches_any_source(self):
        assert _hit(self.PAIR, [_chunk("llm_report.md", "sim-faehigkeiten-bericht")])

    def test_source_id_pins_the_file(self):
        pair = {**self.PAIR, "expected_source_ids": ["sim-begegnungen-bericht"]}
        assert not _hit(pair, [_chunk("llm_report.md", "sim-faehigkeiten-bericht")])
        assert _hit(pair, [_chunk("llm_report.md", "sim-faehigkeiten-bericht"),
                           _chunk("llm_report.md", "sim-begegnungen-bericht")])


def test_cli_summary_survives_a_cp1252_console(monkeypatch, tmp_path):
    """Emoji in retrieved headings crashed the miss list on the Windows console (cp1252)."""
    import io
    import sys

    import evaluation.evaluate_retrieval as ev

    miss = {"id": "m1", "question": "Wer ist Kalen?", "expected_sources": ["a.md"],
            "retrieved_files": ["b.md"], "expected_section": "1.1", "retrieved_headings": ["🎭 Maskenball"]}
    report = {"hit_rate": 0.0, "total_questions": 1, "details": [{"hit": False}],
              "breakdown_by_source_type": {}, "section_total": 1, "section_hit_rate": 0.0,
              "misses": [miss], "section_misses": [miss]}
    monkeypatch.setattr(ev, "load_qa_pairs", lambda path: [{}])
    monkeypatch.setattr(ev, "run_evaluation", lambda *a, **kw: report)
    monkeypatch.setattr(sys, "argv", ["evaluate_retrieval", "--output", str(tmp_path), "--top-k", "5",
                                      "--top-k-rerank", "3"])
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)

    ev.main()

    console.flush()
    assert "🎭 Maskenball" in console.buffer.getvalue().decode("utf-8")


class TestSharedHelpers:
    """One hit rule and one report naming for CLI, end-to-end script and /eval API (roadmap 7.4 A6)."""

    @pytest.mark.parametrize("files, expected", [
        ([r"Orte\Arkenfeld.md"], True),        # Windows separator
        (["Arkenfeld.md"], True),               # no folder
        (["Orte/Arkenfeldt.md"], False),
        ([], False),
    ])
    def test_source_hit_by_file_name(self, files, expected):
        assert source_hit(["Orte/Arkenfeld.md"], files) is expected

    def test_source_hit_honours_expected_source_ids(self):
        files, ids = ["llm_report.md", "llm_report.md"], ["sim-faehigkeiten", "sim-begegnungen"]
        assert source_hit(["llm_report.md"], files[1:], ids[1:], ["sim-faehigkeiten"]) is False
        assert source_hit(["llm_report.md"], files, ids, ["sim-faehigkeiten"]) is True

    def test_load_qa_pairs_missing_or_empty_file(self, tmp_path):
        assert load_qa_pairs(tmp_path / "missing.yaml") == []
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        assert load_qa_pairs(empty) == []

    @pytest.mark.parametrize("eval_type", ["retrieval", "e2e"])
    def test_save_report_uses_the_prefix_the_api_cleans_up(self, tmp_path, eval_type):
        path = save_report({"hit_rate": 1.0, "frage": "Überschaden"}, tmp_path / "results", eval_type)
        assert path.name.startswith(f"{eval_type}_") and path.suffix == ".json"
        assert json.loads(path.read_text(encoding="utf-8"))["frage"] == "Überschaden"



class TestParameterDefaults:
    """Parameters left at None take the retriever's config — the report shows what was really used."""

    PAIR = {"id": "d1", "question": "?", "expected_sources": ["a.md"]}

    def test_none_takes_config_values(self):
        retriever = _FakeRetriever([_chunk("a.md", "vault")])
        retriever.config.top_k = 21
        retriever.config.reranking.top_k_rerank = 6
        retriever.config.reranking.max_per_source = 0  # 0 = no cap, must not fall back to a default

        report = asyncio.run(run_evaluation_with_retriever([self.PAIR], retriever))

        assert report["config"] == {"top_k": 21, "top_k_rerank": 6, "max_per_source": 0, "hybrid": None}
        assert retriever.calls[0]["top_k"] == 21
        assert retriever.calls[0]["max_per_source"] == 0

    def test_explicit_values_win(self):
        retriever = _FakeRetriever([_chunk("a.md", "vault")])
        report = asyncio.run(run_evaluation_with_retriever([self.PAIR], retriever, 10, 4, 2))
        assert report["config"]["top_k"] == 10
        assert report["config"]["top_k_rerank"] == 4
        assert report["config"]["max_per_source"] == 2
