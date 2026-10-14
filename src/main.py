"""
Main module for the LoreKeeper application.

This module configures and starts the FastAPI application, manages the lifecycle
of shared resources, and sets up background tasks for health checking and session
garbage collection. It also handles the auto-start of the Ollama service if
configured as the LLM provider.
"""

import asyncio
import logging
import shutil
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI

from src.api.eval_routes import eval_router
from src.api.prompt_routes import prompt_router
from src.api.routes import router
from src.config.manager import ConfigManager
from src.conversation.manager import ConversationManager
from src.generation.generator import Generator
from src.generation.provider_factory import ProviderFactory
from src.generation.providers.base import BaseLLMProvider
from src.prompts.manager import PromptManager
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import Retriever
from src.retrieval.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)

# Global service instances (initialized in lifespan)
config: ConfigManager = None  # type: ignore
conversation_manager: ConversationManager = None  # type: ignore
prompt_manager: PromptManager = None  # type: ignore
embedding_service: EmbeddingService = None  # type: ignore
vectorstore: VectorStoreService = None  # type: ignore
bm25_index: BM25Index = None  # type: ignore
retriever: Retriever = None  # type: ignore
provider: BaseLLMProvider = None  # type: ignore
condense_provider: BaseLLMProvider | None = None
generator: Generator = None  # type: ignore

_ollama_process: subprocess.Popen | None = None

# Output of an Ollama started by LoreKeeper — the only trace when it crashes (VRAM full, driver reset).
# Appended across starts; moved to .1 once it passes _OLLAMA_LOG_MAX_BYTES (Ollama logs every request).
_OLLAMA_LOG = Path("logs/ollama.log")
_OLLAMA_LOG_MAX_BYTES = 10_485_760


async def _ensure_ollama(base_url: str, timeout: float = 30.0) -> subprocess.Popen | None:
    """Start Ollama if it's not already running. Returns the process if we started it."""
    # Check if already running
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                logger.info("Ollama already running")
                return None
    except Exception:
        pass

    # Find ollama binary
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        logger.warning("Ollama binary not found in PATH — skipping auto-start")
        return None

    logger.info("Starting Ollama (output: %s)...", _OLLAMA_LOG)
    _OLLAMA_LOG.parent.mkdir(parents=True, exist_ok=True)
    if _OLLAMA_LOG.exists() and _OLLAMA_LOG.stat().st_size > _OLLAMA_LOG_MAX_BYTES:
        _OLLAMA_LOG.replace(_OLLAMA_LOG.with_name(_OLLAMA_LOG.name + ".1"))
    # The child keeps its own handle — ours can be closed right after the start. Own process group: a
    # console Ctrl+C reaches every process of the console, and uvicorn's dev reloader restarts the worker
    # with exactly that — Ollama died mid-answer (503) with every code change (roadmap 12.3). It is
    # stopped on purpose in _shutdown instead.
    with _OLLAMA_LOG.open("ab") as out:
        proc = subprocess.Popen([ollama_bin, "serve"], stdout=out, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    # Wait until it's ready
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    logger.info(f"Ollama started (PID {proc.pid})")
                    return proc
        except Exception:
            pass
        await asyncio.sleep(1)

    logger.error("Ollama failed to start within timeout")
    proc.kill()
    return None


async def _shutdown(tasks: list[asyncio.Task]) -> None:
    """Stop the background tasks and an Ollama that LoreKeeper started — also after a failed startup
    (an Ollama started before the crash used to keep running). Cancelled tasks are awaited, so they
    end before the process does."""
    global _ollama_process
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if _ollama_process:
        logger.info(f"Stopping Ollama (PID {_ollama_process.pid})")
        # Ctrl+Break to Ollama's own process group: it shuts down with its llama-server runner, which a
        # TerminateProcess of ollama.exe alone could leave running with the model in VRAM
        _ollama_process.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            _ollama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Ollama did not stop within 10 s — killing it")
            _ollama_process.kill()
        _ollama_process = None
    logger.info("LoreKeeper shutting down")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the FastAPI application, initializing shared resources and background tasks."""
    global config, conversation_manager, prompt_manager
    global embedding_service, vectorstore, bm25_index, retriever
    global provider, condense_provider, generator
    global _ollama_process

    config = ConfigManager(
        settings_path=Path("config/settings.yaml"),
        prompts_path=Path("config/prompts.yaml"),
    )
    settings = config.settings

    # Initialize logging first
    from src.logging_setup import setup_logging
    setup_logging(settings.logging)

    logger.info("Starting LoreKeeper...")

    # try/finally: a startup that fails halfway (model load, ChromaDB) must still stop what it started
    tasks: list[asyncio.Task] = []
    try:
        # Auto-start Ollama if provider is ollama
        if settings.llm.provider == "ollama":
            _ollama_process = await _ensure_ollama(settings.llm.ollama.base_url)

        # Conversation
        conversation_manager = ConversationManager(settings.conversation)

        # Prompts
        prompt_manager = PromptManager(prompts_dict=config.prompts)

        # Embeddings + Vector Store
        embedding_service = EmbeddingService(settings.embeddings)
        await embedding_service.embed_text("warmup")  # pre-load model before first query
        vectorstore = VectorStoreService(settings.vectorstore, embedding_service)
        vectorstore.health_check()  # pre-connect ChromaDB
        # Open the collection now, not in the first request: applies hnsw ef_search at startup (as documented)
        vectorstore.count()

        # Retrieval
        bm25_index = BM25Index()
        retriever = Retriever(settings.retrieval, embedding_service, vectorstore, bm25_index)
        if settings.retrieval.reranking.enabled:
            retriever.get_reranker()  # pre-load reranker model at startup

        # LLM
        provider, condense_provider, generator = ProviderFactory.build_all(settings)

        # Start session GC
        tasks.append(asyncio.create_task(conversation_manager.start_gc()))

        # Start background health checker
        from src.api.system_routes import _health_loop, update_health
        # Seed cache with initial state (vectorstore + provider already initialized); logs the first state
        try:
            update_health(vectorstore.health_check(), await provider.health_check())
        except Exception as e:
            logger.warning("Initial health check failed (%s: %s); the health loop retries", type(e).__name__, e)
        tasks.append(asyncio.create_task(_health_loop()))

        logger.info(f"LoreKeeper ready (provider={settings.llm.provider}, model={getattr(provider, 'model', '?')})")

        yield
    finally:
        await _shutdown(tasks)


app = FastAPI(
    title="LoreKeeper",
    description="RAG-based Q&A System for Worldbuilding Documents",
    version="0.2.0",
    lifespan=lifespan,
)


_QUIET_PATHS = {"/health", "/sidebar-state"}
# The response starts before the answer is generated — its duration is only the time to the headers.
# The real total is in the "Query timings" line.
_STREAMING_PATHS = {"/query/stream"}


@app.middleware("http")
async def request_logging(request, call_next):
    """Log incoming HTTP requests and their processing times, ignoring specific paths.

    Successful GETs (page loads, job polling every 1–1.5 s) go to DEBUG; changes and errors stay visible.

    An unhandled exception gets its own line here; the traceback follows from uvicorn on ``uvicorn.error``,
    which ``setup_logging`` routes into the log file.
    """
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.error("%s %s → unhandled exception (%.0fms)", request.method, request.url.path,
                     (time.time() - start) * 1000)
        raise
    if request.url.path in _QUIET_PATHS:
        return response
    duration_ms = (time.time() - start) * 1000
    if response.status_code >= 500:
        level = logging.WARNING
    elif request.method == "GET" and response.status_code < 400:
        level = logging.DEBUG
    else:
        level = logging.INFO
    timing = "stream started after" if request.url.path in _STREAMING_PATHS else "in"
    logger.log(level, "%s %s → %d (%s %.0fms)", request.method, request.url.path, response.status_code,
               timing, duration_ms)
    return response


app.include_router(router)
app.include_router(eval_router)
app.include_router(prompt_router)
