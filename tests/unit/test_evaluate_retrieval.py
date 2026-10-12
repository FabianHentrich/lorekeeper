import asyncio
from types import SimpleNamespace

from evaluation.evaluate_retrieval import run_evaluation_with_retriever, section_hit


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

    async def retrieve(self, **kwargs):
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
