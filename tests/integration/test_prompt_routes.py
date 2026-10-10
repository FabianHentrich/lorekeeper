"""Integration tests for the prompt variant endpoints — path handling only."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.prompt_routes as mod

_PROMPTS = {"system": "S", "qa": "Q {{ question }}", "condense": "C", "no_context": "N"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient whose variants dir points at tmp_path/variants."""
    variants = tmp_path / "variants"
    variants.mkdir()
    monkeypatch.setattr(mod, "_variants_dir", lambda: variants)
    app = FastAPI()
    app.include_router(mod.prompt_router)
    return TestClient(app)


def test_save_and_get_variant(client, tmp_path):
    """Verify a plain name (spaces, umlauts) round-trips inside the variants dir."""
    body = {"name": "Kurz & knapp", "description": "", "prompts": _PROMPTS}
    assert client.put("/prompts/variants/Kürzer Stil", json=body).status_code == 200
    assert (tmp_path / "variants" / "Kürzer Stil.yaml").exists()
    assert client.get("/prompts/variants/Kürzer Stil").json()["prompts"]["system"] == "S"


@pytest.mark.parametrize("name", ["..%5Cevil", "sub%5Cevil"])
def test_variant_names_cannot_escape_variants_dir(client, tmp_path, name):
    """Verify encoded backslashes are rejected for write, read and delete."""
    body = {"name": "x", "description": "", "prompts": _PROMPTS}
    assert client.put(f"/prompts/variants/{name}", json=body).status_code == 400
    assert client.get(f"/prompts/variants/{name}").status_code == 400
    assert client.delete(f"/prompts/variants/{name}").status_code == 400
    assert not (tmp_path / "evil.yaml").exists()
