"""The backend's HTTP API as one router: chat queries, sources/index, system.

Split into three modules by area (roadmap 11.0 P7) — order matters only within a module. ``/eval/*`` and
``/prompts/*`` are separate routers, included in ``src.main``.
"""
from fastapi import APIRouter

from src.api.query_routes import router as query_router
from src.api.sources_routes import router as sources_router
from src.api.system_routes import router as system_router

router = APIRouter()
router.include_router(query_router)
router.include_router(sources_router)
router.include_router(system_router)
