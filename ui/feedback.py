"""Turning chat ratings (👍/👎 from logs/feedback.jsonl) into golden-set questions."""


def pairs_from_feedback(entries: list[dict], existing: list[dict]) -> list[dict]:
    """Golden-set pairs for rated answers. 👍 keeps the answer's sources as expected sources;
    👎 leaves them empty — those sources were likely wrong, the right one is entered by hand.
    A condensed follow-up is taken in its standalone form (``search_query``) — "Und wer ist sein
    Vater?" means nothing without the chat. The rater's comment goes into the notes. Questions
    already in the golden set are skipped; ids continue as f001, f002, …"""
    known = {p.get("question", "").strip() for p in existing}
    used = {p.get("id", "") for p in existing}
    n = 0
    pairs = []
    for e in entries:
        asked = e.get("question", "").strip()
        question = (e.get("search_query") or asked).strip()
        if not question or question in known:
            continue
        while f"f{n:03d}" in used or n == 0:
            n += 1
        used.add(f"f{n:03d}")
        known.add(question)
        sources = list(dict.fromkeys(e.get("sources", [])))
        up = e.get("rating") == "up"
        pairs.append({
            "id": f"f{n:03d}",
            "question": question,
            "source_type": "pdf" if sources and all(s.lower().endswith(".pdf") for s in sources) else "markdown",
            "category": "feedback",
            "expected_sources": sources if up else [],
            "expected_answer_contains": [],
            "notes": (f"Aus Feedback 👍 ({e.get('ts', '')}, {e.get('role', '')})" if up else
                      f"Aus Feedback 👎 ({e.get('ts', '')}, {e.get('role', '')}) — erwartete Quelle eintragen; "
                      f"damals gefunden: {', '.join(sources) or '—'}")
                     + (f" — im Chat gefragt: „{asked}“" if asked and asked != question else "")
                     + (f" — Kommentar: „{e['comment']}“" if e.get("comment") else ""),
        })
    return pairs
