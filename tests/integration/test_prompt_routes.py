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


# ─── Active prompts, variants and preview against a real ConfigManager ──────────

import src.main as main_module  # noqa: E402
from src.config.manager import ConfigManager  # noqa: E402
from src.prompts.manager import PromptManager  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    """config/prompts.yaml on tmp files; variants live next to it in config/prompts/ (the real layout)."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    active = cfg_dir / "prompts.yaml"
    active.write_text("system: Alt\nqa: 'Alt {{ question }}'\ncondense: C\nno_context: N\n", encoding="utf-8")
    config = ConfigManager(settings_path=tmp_path / "settings.yaml", prompts_path=active,
                           sources_path=tmp_path / "sources.yaml")
    monkeypatch.setattr(main_module, "config", config, raising=False)
    monkeypatch.setattr(main_module, "prompt_manager", None, raising=False)
    app = FastAPI()
    app.include_router(mod.prompt_router)
    return TestClient(app), active, cfg_dir / "prompts"


def _variant(name="Neu", **prompts):
    return {"name": name, "description": "Beschreibung", "prompts": {**_PROMPTS, **prompts}}


def test_get_active_reads_the_file(env):
    client, _, _ = env
    body = client.get("/prompts/active").json()
    assert body["name"] == "active"
    assert body["prompts"]["qa"] == "Alt {{ question }}"


def test_get_active_404_without_file(env):
    client, active, _ = env
    active.unlink()
    assert client.get("/prompts/active").status_code == 404


def test_put_active_saves_and_hot_reloads(env):
    client, active, _ = env
    assert client.put("/prompts/active", json=_variant(system="Neu")).status_code == 200
    assert "Neu" in active.read_text(encoding="utf-8")
    assert isinstance(main_module.prompt_manager, PromptManager)


def test_put_active_rejects_broken_jinja_and_keeps_the_file(env):
    client, active, _ = env
    before = active.read_text(encoding="utf-8")
    r = client.put("/prompts/active", json=_variant(qa="{% if %}"))
    assert r.status_code == 422
    assert "'qa'" in r.json()["detail"]
    assert active.read_text(encoding="utf-8") == before
    assert main_module.prompt_manager is None


def test_list_variants_without_directory_is_empty(env):
    client, _, _ = env
    assert client.get("/prompts/variants").json() == []


def test_activated_variant_is_marked_active(env):
    """After activating, the list must show that variant as the active one."""
    client, _, _ = env
    client.put("/prompts/variants/Kurz", json=_variant("Kurz", system="Kurz"))
    client.put("/prompts/variants/Lang", json=_variant("Lang", system="Lang"))

    r = client.post("/prompts/activate/Kurz")

    assert r.status_code == 200 and r.json()["name"] == "Kurz"
    assert {v["name"]: v["is_active"] for v in client.get("/prompts/variants").json()} == \
        {"Kurz": True, "Lang": False}
    assert client.get("/prompts/active").json()["prompts"]["system"] == "Kurz"
    assert isinstance(main_module.prompt_manager, PromptManager)


def test_unknown_variant_is_404_for_get_delete_and_activate(env):
    client, _, _ = env
    assert client.get("/prompts/variants/Fehlt").status_code == 404
    assert client.delete("/prompts/variants/Fehlt").status_code == 404
    assert client.post("/prompts/activate/Fehlt").status_code == 404


def test_delete_variant(env):
    client, _, variants = env
    client.put("/prompts/variants/Weg", json=_variant("Weg"))
    assert client.delete("/prompts/variants/Weg").json() == {"deleted": "Weg"}
    assert not (variants / "Weg.yaml").exists()


def test_save_variant_rejects_broken_jinja(env):
    client, _, variants = env
    assert client.put("/prompts/variants/Kaputt", json=_variant(system="{{ x ")).status_code == 422
    assert not (variants / "Kaputt.yaml").exists()


def test_activating_a_hand_edited_broken_variant_is_rejected(env):
    """A variant file edited outside the UI is validated before it replaces the active prompts."""
    client, active, variants = env
    variants.mkdir()
    (variants / "Hand.yaml").write_text("system: '{% for %}'\nqa: Q\ncondense: C\nno_context: N\n",
                                        encoding="utf-8")
    before = active.read_text(encoding="utf-8")
    assert client.post("/prompts/activate/Hand").status_code == 422
    assert active.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("template, data, status, expected", [
    ("Frage: {{ question }}", {"question": "Was ist Arkenfeld?"}, 200, "Frage: Was ist Arkenfeld?"),
    ("{% if %}", {}, 422, "Jinja2-Syntaxfehler"),
    ("{{ missing.attr() }}", {}, 422, "Render-Fehler"),
    # Sandbox: the classic route from a string to Python internals is refused (roadmap 11.0 #11)
    ("{{ ''.__class__.__mro__[1].__subclasses__() }}", {}, 422, "Render-Fehler"),
])
def test_preview(env, template, data, status, expected):
    client, _, _ = env
    r = client.post("/prompts/preview", json={"template_text": template, "sample_data": data})
    assert r.status_code == status
    assert expected in (r.json().get("rendered") or r.json().get("detail"))
