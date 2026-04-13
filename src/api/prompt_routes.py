import hashlib
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

_ACTIVE_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts.yaml"
_VARIANTS_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"
_TEMPLATE_KEYS = ("system", "qa", "condense", "no_context")

_jinja_env = Environment()


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


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
            )


def _reload_prompt_manager(prompts_dict: dict):
    import src.main as main_module
    main_module.prompt_manager = PromptManager(prompts_dict=prompts_dict)
    main_module.config._prompts_raw = prompts_dict


def _read_variant_file(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw


def _prompts_dict_from_set(ps: PromptSet) -> dict:
    return {k: getattr(ps, k) for k in _TEMPLATE_KEYS}


# ─── Active prompts ─────────────────────────────────────────────────

@prompt_router.get("/active", response_model=PromptVariant)
async def get_active():
    if not _ACTIVE_PATH.exists():
        raise HTTPException(status_code=404, detail="config/prompts.yaml not found")
    raw = _read_variant_file(_ACTIVE_PATH)
    prompts = {k: raw.get(k, "") for k in _TEMPLATE_KEYS}
    meta = raw.get("_meta", {})
    return PromptVariant(
        name=meta.get("name", "active"),
        description=meta.get("description", ""),
        prompts=PromptSet(**prompts),
    )


@prompt_router.put("/active", response_model=PromptVariant)
async def update_active(variant: PromptVariant):
    _validate_templates(variant.prompts)
    import src.main as main_module
    prompts_dict = _prompts_dict_from_set(variant.prompts)
    main_module.config.save_prompts(prompts_dict)
    _reload_prompt_manager(prompts_dict)
    logger.info("Active prompts updated and reloaded")
    return variant


# ─── Variants ────────────────────────────────────────────────────────

@prompt_router.get("/variants", response_model=list[PromptVariantSummary])
async def list_variants():
    if not _VARIANTS_DIR.exists():
        return []

    active_hash = _file_hash(_ACTIVE_PATH) if _ACTIVE_PATH.exists() else ""

    summaries = []
    for f in sorted(_VARIANTS_DIR.glob("*.yaml")):
        raw = _read_variant_file(f)
        meta = raw.get("_meta", {})
        summaries.append(PromptVariantSummary(
            name=f.stem,
            description=meta.get("description", ""),
            is_active=_file_hash(f) == active_hash if active_hash else False,
        ))
    return summaries


@prompt_router.get("/variants/{name}", response_model=PromptVariant)
async def get_variant(name: str):
    path = _VARIANTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Variante '{name}' nicht gefunden")
    raw = _read_variant_file(path)
    meta = raw.get("_meta", {})
    prompts = {k: raw.get(k, "") for k in _TEMPLATE_KEYS}
    return PromptVariant(
        name=meta.get("name", name),
        description=meta.get("description", ""),
        prompts=PromptSet(**prompts),
    )


@prompt_router.put("/variants/{name}", response_model=PromptVariant)
async def save_variant(name: str, variant: PromptVariant):
    _validate_templates(variant.prompts)
    _VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "_meta": {"name": variant.name, "description": variant.description},
    }
    for k in _TEMPLATE_KEYS:
        data[k] = getattr(variant.prompts, k)

    path = _VARIANTS_DIR / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    logger.info(f"Variant saved: {name}")
    return variant


@prompt_router.delete("/variants/{name}")
async def delete_variant(name: str):
    path = _VARIANTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Variante '{name}' nicht gefunden")
    path.unlink()
    logger.info(f"Variant deleted: {name}")
    return {"deleted": name}


@prompt_router.post("/activate/{name}", response_model=PromptVariant)
async def activate_variant(name: str):
    path = _VARIANTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Variante '{name}' nicht gefunden")

    raw = _read_variant_file(path)
    prompts_dict = {k: raw.get(k, "") for k in _TEMPLATE_KEYS}

    ps = PromptSet(**prompts_dict)
    _validate_templates(ps)

    import src.main as main_module
    main_module.config.save_prompts(prompts_dict)
    _reload_prompt_manager(prompts_dict)

    meta = raw.get("_meta", {})
    logger.info(f"Variant activated: {name}")
    return PromptVariant(
        name=meta.get("name", name),
        description=meta.get("description", ""),
        prompts=ps,
    )


# ─── Preview ────────────────────────────────────────────────────────

@prompt_router.post("/preview", response_model=RenderPreviewResponse)
async def preview_template(req: RenderPreviewRequest):
    try:
        template = _jinja_env.from_string(req.template_text)
        rendered = template.render(**req.sample_data)
    except TemplateSyntaxError as e:
        raise HTTPException(status_code=422, detail=f"Jinja2-Syntaxfehler: {e}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Render-Fehler: {e}")
    return RenderPreviewResponse(rendered=rendered)
