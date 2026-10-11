"""Live security check of the player mode against the running backend and the real index.

Run with ``pytest -m live`` while LoreKeeper runs (``start.ps1`` or ``start.ps1 -Server``).

1. Every golden-set question about vault knowledge (NPCs, places, adventures, secrets —
   everything not answered by the rulebook) is retrieved with the player filter: no
   chunk may come from a source that isn't released for players.
2. A few game-master secrets are asked through the full answer path (LLM included):
   neither the answer nor its sources may reveal them.

The released sources come from the backend (/sidebar-state → player_source_ids), the
same list the chat uses.
"""
from pathlib import Path

import pytest
import requests
import yaml

pytestmark = pytest.mark.live

API = "http://127.0.0.1:8000"
GOLDEN = Path(__file__).resolve().parents[2] / "evaluation" / "qa_pairs.yaml"
RULES_FILES = {"pnp_regelbuch.pdf"}

# Questions whose answer is game-master knowledge, with words only that knowledge contains
SECRET_PROBES = [
    ("Wer ist Sol Invictus wirklich?", ["rhaz"]),
    ("Was ist mit Isolde Vaer los?", ["nahlith", "besessen"]),
    ("Welche Untoten hat Malek Nocthar gespeichert?", ["ardomis", "rhadomir"]),
    ("Wie wurde Thal'Emyra zerstört und was bedeutet K-37?", ["alaric", "eona"]),
    ("Wer ist Alaric von Lysander wirklich?", ["archon", "thal'emyra", "eona"]),
]


def _vault_questions() -> list[tuple[str, str]]:
    pairs = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["pairs"]
    return [(p["id"], p["question"]) for p in pairs
            if not RULES_FILES & {s.split("/")[-1] for s in p.get("expected_sources", [])}]


@pytest.fixture(scope="module")
def player_ids() -> list[str]:
    try:
        state = requests.get(f"{API}/sidebar-state", timeout=5).json()
    except requests.RequestException:
        pytest.skip(f"no backend on {API}")
    ids = state.get("player_source_ids")
    assert ids, "no source is released for players — the player mode would be locked (Sources page → 🎲 Spieler)"
    return ids


def _player_filter(ids: list[str]) -> dict:
    return {"source_id": {"$in": ids}}


@pytest.mark.parametrize("hybrid", [False, True], ids=["vector", "hybrid"])
@pytest.mark.parametrize("qid, question", _vault_questions(), ids=[q[0] for q in _vault_questions()])
def test_vault_question_retrieves_only_player_sources(player_ids, qid, question, hybrid):
    data = requests.post(f"{API}/eval/preview", timeout=120, json={
        "question": question, "metadata_filters": _player_filter(player_ids), "hybrid": hybrid,
    }).json()

    leaked = [(c["source_id"], c["source_file"]) for c in data["chunks"] if c["source_id"] not in player_ids]
    assert not leaked, f"[{qid}] player retrieval reached unreleased sources: {leaked}"


@pytest.mark.parametrize("question, secret_words", SECRET_PROBES, ids=[p[0][:40] for p in SECRET_PROBES])
def test_secret_stays_out_of_player_answers(player_ids, question, secret_words):
    """Full answer path including the LLM, the way the chat asks in player mode."""
    resp = requests.post(f"{API}/query", timeout=300, json={
        "question": question, "metadata_filters": _player_filter(player_ids),
    }).json()

    answer = resp["answer"].lower()
    assert not [w for w in secret_words if w in answer], f"answer reveals a secret: {resp['answer'][:300]}"
    files = {s["file"] for s in resp["sources"]}
    released = {Path(s["path"]).name for s in requests.get(f"{API}/sources", timeout=10).json()["sources"]
                if s["id"] in player_ids}
    assert files <= released, f"answer used unreleased sources: {files - released}"
