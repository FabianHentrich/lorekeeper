import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from jinja2 import Environment, TemplateSyntaxError

from src.api.prompt_schemas import (
    PromptSet,
    PromptVariant,
    PromptVariantSummary,
    RenderPreviewRequest,
    RenderPreviewResponse,
)
from src.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

prompt_router = APIRouter(prefix="/prompts", tags=["prompts"])

_TEMPLATE_KEYS = ("system", "qa", "condense", "no_context")


def _active_path() -> Path:
    """Resolve the active prompts path from the running ConfigManager so both
    subsystems agree on the same file regardless of CWD."""
    import src.main as main_module
    return main_module.config._prompts_path


def _variants_dir() -> Path:
    """Resolve the directory where prompt variants are stored."""
    return _active_path().parent / "prompts"

_jinja_env = Environment()


def _variant_path(name: str) -> Path:
    """Resolve a variant file, rejecting names that would escape the variants dir (e.g. '..' + backslash on Windows)."""
    if not name or name in (".", "..") or Path(name).name != name:
        raise HTTPException(status_code=400, detail=f"Ungültiger Variantenname: {name!r}")
    return _variants_dir() / f"{name}.yaml"


def _validate_templates(prompts: PromptSet):
    """Validate Jinja2 syntax for all templates. Raises HTTPException on error."""
    for key in _TEMPLATE_KEYS:
        raw = getattr(prompts, key)
        try:
            _jinja_env.from_string(raw)
        except TemplateSyntaxError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Jinja2-Syntaxfehler in '{key}': {e}",
            ) from e


def _reload_prompt_manager(prompts_dict: dict):
    """Hot-reload the application's PromptManager instance with new dictionaries."""
    import src.main as main_module
    main_module.prompt_manager = PromptManager(prompts_dict=prompts_dict)


def _read_variant_file(path: Path) -> dict:
    """Load a prompts YAML file (the active prompts.yaml or a variant) as a dict."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _templates(raw: dict) -> dict:
    """The four templates of a loaded prompts file; missing ones are empty."""
    return {k: raw.get(k, "") for k in _TEMPLATE_KEYS}


def _variant_from_raw(raw: dict, default_name: str) -> PromptVariant:
    """A loaded prompts file as a PromptVariant (name/description from ``_meta`` if present)."""
    meta = raw.get("_meta", {})
    return PromptVariant(
        name=meta.get("name", default_name),
        description=meta.get("description", ""),
        prompts=PromptSet(**_templates(raw)),
    )


# ─── Active prompts ─────────────────────────────────────────────────────────────

@prompt_router.get("/active", response_model=PromptVariant)
async def get_active():
    """Retrieve the currently active prompt configuration from disk."""
    if not _active_path().exists():
        raise HTTPException(status_code=404, detail="config/prompts.yaml not found")
    return _variant_from_raw(_read_variant_file(_active_path()), "active")


@prompt_router.put("/active", response_model=PromptVariant)
async def update_active(variant: PromptVariant):
    """Update and hot-reload the active prompt configuration immediately."""
    _validate_templates(variant.prompts)
    import src.main as main_module
    prompts_dict = variant.prompts.model_dump()
    main_module.config.save_prompts(prompts_dict)
    _reload_prompt_manager(prompts_dict)
    logger.info("Active prompts updated and reloaded")
    return variant


# ─── Variants ───────────────────────────────────────────────────────────────────

@prompt_router.get("/variants", response_model=list[PromptVariantSummary])
async def list_variants():
    """List all available prompt variants discovered in the variants directory.

    A variant is active when its four templates equal the active ones. File bytes can't
    be compared: variants carry `_meta`, the active prompts.yaml never does.
    """
    if not _variants_dir().exists():
        return []

    active = _templates(_read_variant_file(_active_path())) if _active_path().exists() else None

    summaries = []
    for f in sorted(_variants_dir().glob("*.yaml")):
        raw = _read_variant_file(f)
        meta = raw.get("_meta", {})
        summaries.append(PromptVariantSummary(
            name=f.stem,
            description=meta.get("description", ""),
            is_active=_templates(raw) == active,
        ))
    return summaries


@prompt_router.get("/variants/{name}", response_model=PromptVariant)
async def get_variant(name: str):
    """Load a specific prompt variant by its filename/name."""
    path = _variant_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Variante '{name}' nicht gefunden")
    return _variant_from_raw(_read_variant_file(path), name)


@prompt_router.put("/variants/{name}", response_model=PromptVariant)
async def save_variant(name: str, variant: PromptVariant):
    """Save or overwrite a specific prompt variant to the disk."""
    _validate_templates(variant.prompts)
    _variants_dir().mkdir(parents=True, exist_ok=True)

    data = {"_meta": {"name": variant.name, "description": variant.description}, **variant.prompts.model_dump()}

    path = _variant_path(name)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    logger.info(f"Variant saved: {name}")
    return variant


@prompt_router.delete("/variants/{name}")
async def delete_variant(name: str):
    """Delete a specific prompt variant file from the disk."""
    path = _variant_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Variante '{name}' nicht gefunden")
    path.unlink()
    logger.info(f"Variant deleted: {name}")
    return {"deleted": name}


@prompt_router.post("/activate/{name}", response_model=PromptVariant)
async def activate_variant(name: str):
    """Promote a specific variant to be the currently active prompt configuration.

    This copies the variant's content to the active prompts.yaml file and triggers
    a hot-reload of the PromptManager within the FastAPI app.
    """
    path = _variant_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Variante '{name}' nicht gefunden")

    variant = _variant_from_raw(_read_variant_file(path), name)
    _validate_templates(variant.prompts)

    import src.main as main_module
    prompts_dict = variant.prompts.model_dump()
    main_module.config.save_prompts(prompts_dict)
    _reload_prompt_manager(prompts_dict)

    logger.info(f"Variant activated: {name}")
    return variant


# ─── Preview ────────────────────────────────────────────────────────────────────

@prompt_router.post("/preview", response_model=RenderPreviewResponse)
async def preview_template(req: RenderPreviewRequest):
    """Render a raw Jinja2 template string against provided dummy payload data.

    Used by the UI to show live previews without needing to overwrite active systems.
    """
    try:
        template = _jinja_env.from_string(req.template_text)
        rendered = template.render(**req.sample_data)
    except TemplateSyntaxError as e:
        raise HTTPException(status_code=422, detail=f"Jinja2-Syntaxfehler: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Render-Fehler: {e}") from e
    return RenderPreviewResponse(rendered=rendered)
