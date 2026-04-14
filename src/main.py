import asyncio
import logging
import shutil
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

    logger.info("Starting Ollama...")
    proc = subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait until it's ready
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
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

    # Retrieval
    bm25_index = BM25Index()
    retriever = Retriever(settings.retrieval, embedding_service, vectorstore, bm25_index)
    if settings.retrieval.reranking.enabled:
        retriever.get_reranker()  # pre-load reranker model at startup

    # LLM
    provider = ProviderFactory.create(settings.llm)
    condense_provider = ProviderFactory.create_condense_provider(settings.llm, settings.conversation)
    fallback_provider = ProviderFactory.create_fallback(settings.llm)
    generator = Generator(provider=provider, fallback_provider=fallback_provider)

    # Start session GC
    gc_task = asyncio.create_task(conversation_manager.start_gc())

    # Start background health checker
    from src.api.routes import _health_loop, _health_cache
    # Seed cache with initial state (vectorstore + provider already initialized)
    try:
        chroma_ok = vectorstore.health_check()
        llm_ok = await provider.health_check()
        _health_cache.update({"ts": time.monotonic(), "chroma": chroma_ok, "llm": llm_ok})
    except Exception:
        pass
    health_task = asyncio.create_task(_health_loop())

    logger.info(f"LoreKeeper ready (provider={settings.llm.provider}, model={getattr(provider, 'model', '?')})")

    yield

    health_task.cancel()
    gc_task.cancel()

    if _ollama_process:
        logger.info(f"Stopping Ollama (PID {_ollama_process.pid})")
        _ollama_process.terminate()
        _ollama_process.wait(timeout=10)

    logger.info("LoreKeeper shutting down")


app = FastAPI(
    title="LoreKeeper",
    description="RAG-based Q&A System for Worldbuilding Documents",
    version="0.2.0",
    lifespan=lifespan,
)


_QUIET_PATHS = {"/health", "/sidebar-state"}


@app.middleware("http")
async def request_logging(request, call_next):
    """Log incoming HTTP requests and their processing times, ignoring specific paths."""
    start = time.time()
    response = await call_next(request)
    if request.url.path in _QUIET_PATHS:
        return response
    duration_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.0f}ms)"
    )
    return response


app.include_router(router)
app.include_router(eval_router)
app.include_router(prompt_router)
