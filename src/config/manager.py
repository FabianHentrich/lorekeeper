"""Configuration: a Pydantic model per settings section, loading, and the Settings-page allow-list.

Override order (highest wins): environment variables (``LLM__OLLAMA__MODEL`` style, ``__`` = nesting) >
``.env`` > ``config/settings.local.yaml`` (written by the Settings page) > ``config/settings.yaml`` > the code
defaults below. The code defaults are only fallbacks — several differ from the shipped settings.yaml (e.g.
``reranking.backend``, ``rules_slot``, LLM ``max_tokens``/``timeout``). Sources come from the gitignored
sidecar ``config/sources.yaml``. Only keys in ``_EDITABLE_KEYS`` can be changed via ``PUT /config``;
``save_settings`` mutates the live objects in place, so services holding references see new values on
their next call. Full reference: docs/configuration.md.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Allowed source groups — fixed on purpose, the UI filters are built on exactly these four.
SourceGroup = Literal["lore", "adventure", "rules", "simulation"]


# Allow-list of keys that the Settings UI is permitted to edit via
# ConfigManager.save_settings. Unknown keys are silently dropped — this keeps
# secrets, infra-level knobs, and ingest-time schema off the UI surface.
_EDITABLE_KEYS: dict = {
    "retrieval": {
        "top_k": None,
        "score_threshold": None,
        "reranking": {
            "enabled": None,
            "top_k_rerank": None,
            "max_per_source": None,
        },
        "hybrid": {
            "enabled": None,
            "bm25_weight": None,
            "bm25_top_k": None,
        },
    },
    "llm": {
        "fallback_enabled": None,
        "ollama": {
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
            "timeout": None,
        },
        "gemini": {
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
            "timeout": None,
        },
    },
    "conversation": {
        "window_size": None,
        "condense_question": None,
        "session_timeout_minutes": None,
    },
    "chunking": {
        "strategy": None,
        "max_chunk_size": None,
        "chunk_overlap": None,
        "min_chunk_size": None,
    },
}

# Shown by GET /config for context but not editable there (models are picked in settings.yaml).
# No file paths and no secrets in the API response.
_READ_ONLY_KEYS: dict = {
    "llm": {"provider": None, "ollama": {"model": None}, "gemini": {"model": None}},
    "embeddings": {"model": None, "device": None},
    "vectorstore": {"collection_name": None},
}


class SourceConfig(BaseModel):
    """A single ingestion source. Either a folder or a single file."""
    id: str                                     # Stable user-supplied identifier
    path: str                                   # File OR folder
    group: SourceGroup                          # UI filter group
    default_category: str | None = None         # Used for files & flat sources
    category_map: dict[str, str | dict[str, str]] = {}  # Top-folder -> category or {category, group}
    exclude_patterns: list[str] = []            # Additive to global excludes
    player_visible: bool = False                # Searchable in the chat's player mode (default: game master only)

    @field_validator("category_map")
    @classmethod
    def _validate_category_map(cls, v: dict) -> dict:
        """Reject malformed category_map entries — each value must be a string or a dict with 'category'."""
        valid_groups = {"lore", "adventure", "rules", "simulation"}
        for folder, entry in v.items():
            if isinstance(entry, str):
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"category_map[{folder!r}]: expected str or dict, got {type(entry).__name__}")
            if "category" not in entry:
                raise ValueError(f"category_map[{folder!r}]: dict entry must have 'category' key")
            if "group" in entry and entry["group"] not in valid_groups:
                raise ValueError(f"category_map[{folder!r}]: group must be one of {valid_groups}")
        return v


class PdfConfig(BaseModel):
    """PDF parsing (``ingestion.pdf``). Changes need a full reindex — they leave the file hash alone, so an
    incremental run skips the PDFs."""
    ocr_enabled: bool = True         # OCR (rapidocr) for areas without extractable text; on failure: text without OCR
    ocr_language: str = "deu"        # ISO 639-3 language code
    ocr_dpi: int = 300               # render resolution for OCR
    extract_images: bool = True      # write embedded images out as image chunks (not retrieved today, roadmap 2)
    image_format: str = "png"


class IngestionConfig(BaseModel):
    """What gets indexed. ``sources`` is filled from config/sources.yaml, not from settings.yaml."""
    sources: list[SourceConfig] = []
    supported_formats: list[str] = [".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp"]  # file suffixes, case-insensitive
    # fnmatch patterns relative to the source folder, for every source (sources can add their own)
    exclude_patterns: list[str] = [".obsidian/*", ".trash/*", "*alt.md", "*(1).md", "*.draft.*"]
    pdf: PdfConfig = PdfConfig()


class ChunkingConfig(BaseModel):
    """How parsed sections are split. Takes effect only after a full reindex (``POST /ingest?full=true``)."""
    strategy: str = "heading_aware"  # heading_aware | recursive | fixed_size
    max_chunk_size: int = 256        # estimated tokens (len/3.5) per prose chunk; tables stay whole
    chunk_overlap: int = 30          # must stay below max_chunk_size — the split loops advance by the difference
    min_chunk_size: int = 20         # smaller chunks are merged into the previous one (same heading only)

    @model_validator(mode="after")
    def _overlap_below_size(self):
        """An overlap ≥ the chunk size would make the word-split loops stand still (ingest hangs)."""
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) muss kleiner als max_chunk_size ({self.max_chunk_size}) sein"
            )
        return self


class EmbeddingsConfig(BaseModel):
    """Embedding model. Changing ``model`` (or where it runs) needs a full re-index."""
    model: str = "intfloat/multilingual-e5-base"  # e5 models get query:/passage: prefixes automatically
    device: str = "auto"             # auto | cpu | cuda | mps — on AMD under Windows it runs on the CPU
    batch_size: int = 64             # texts per encode call while indexing
    normalize: bool = True           # L2-normalise vectors — required for cosine scores


class VectorStoreConfig(BaseModel):
    """ChromaDB location and collection."""
    persist_directory: str = "./chroma_data"   # embedded ChromaDB inside the backend process
    collection_name: str = "lorekeeper"
    # Only "cosine" gives correct scores (score = 1 - distance); fixed when the collection is created
    distance_metric: str = "cosine"
    # HNSW search breadth. 1000 makes the search exact and identical across processes at our size
    # (4.7 k chunks; 100 missed exact top-15 members for 38 of 278 golden questions and varied per
    # process) for ~2 ms per query — roadmap 5.5. Applied to an existing collection at startup.
    hnsw_ef_search: int = 1000


class RerankingConfig(BaseModel):
    """Cross-encoder rerank of the vector/BM25 candidates; editable values apply on the next query."""
    enabled: bool = True
    model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingual; loaded at startup
    # "directml": GPU via onnxruntime-directml (falls back to torch on the CPU if unavailable)
    backend: Literal["torch", "directml"] = "torch"
    top_k_rerank: int = 8            # chunks after the rerank = LLM context (before section expansion)
    max_per_source: int = 3          # soft cap per file (0 = unlimited); backfilled from the overflow if short


class HybridSearchConfig(BaseModel):
    """BM25 keyword search fused with the vector hits (RRF). Per request: ``hybrid_search``."""
    enabled: bool = False
    bm25_weight: float = 0.3         # BM25 share in RRF (0.0 = pure vector, 1.0 = pure BM25)
    bm25_top_k: int = 15             # how many BM25 candidates to fetch


class SectionExpansionConfig(BaseModel):
    """≥ 2 hits under one parent heading → the LLM gets the whole section in document order."""
    enabled: bool = True
    max_chunks: int = 25             # sections larger than this are not expanded (protects the LLM context)
    top_rank: int = 3                # only expand if a hit is among the first N ranked chunks (0 = no limit)


class RulesSlotConfig(BaseModel):
    """Reserve one final place for the rulebook (roadmap 3.10).

    Questions naming a vault entity *and* a rule term ("Was bewirkt Ritterlichkeit bei Escanor?"): the
    entity's file fills every vector candidate, the rulebook never reaches the reranker. Values in use
    (settings.yaml: 5 candidates, min_score -4) come from a golden-set sweep, see docs/configuration.md.
    """
    enabled: bool = False
    candidates: int = 8              # extra vector candidates from the "rules" group, reranked with the rest
    min_score: float = -5.0          # reserve one top_k_rerank slot for the best rules chunk if it scores ≥ this


class RetrievalConfig(BaseModel):
    """Search parameters; ``top_k``/``top_k_rerank``/``max_per_source``/hybrid can be overridden per request."""
    top_k: int = 15                  # vector candidates before the rerank (keep ≥ 2 × top_k_rerank)
    score_threshold: float = 0.5     # minimum cosine score — 0.3 lets near-orthogonal chunks in, don't lower
    reranking: RerankingConfig = RerankingConfig()
    hybrid: HybridSearchConfig = HybridSearchConfig()
    section_expansion: SectionExpansionConfig = SectionExpansionConfig()
    rules_slot: RulesSlotConfig = RulesSlotConfig()


class OllamaConfig(BaseModel):
    """Local Ollama server. Sampling values and timeout are read per call (live edits apply); url and model
    need a provider rebuild."""
    base_url: str = "http://127.0.0.1:11434"  # 127.0.0.1, not localhost (~2 s per new connection on Windows)
    model: str = "qwen3:14b"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 1024           # output tokens per answer
    timeout: int = 120               # seconds; the chat derives its read timeout from it


class GeminiConfig(BaseModel):
    """Google Gemini (cloud, paid). The key itself never lives in a settings file."""
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"  # name of the environment variable holding the key
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 1024           # output tokens per answer
    timeout: int = 30                # seconds


class LLMConfig(BaseModel):
    """Active provider at startup (switchable at runtime via ``POST /provider``) and the fallback."""
    provider: str = "ollama"         # ollama | gemini
    fallback_provider: str | None = "gemini"
    fallback_enabled: bool = False   # retry with the fallback if the main provider fails before the first token
    ollama: OllamaConfig = OllamaConfig()
    gemini: GeminiConfig = GeminiConfig()


class ConversationConfig(BaseModel):
    """Chat sessions and their history."""
    window_size: int = 8             # QA pairs of history for rewriting follow-up questions (condense)
    max_context_tokens: int = 4096   # hard cap on that history in estimated tokens; oldest pairs drop first
    # Answer prompt: last N QA pairs (0 = off); older answers cut at max_chars — more dilutes the sources
    answer_history_pairs: int = 1
    answer_history_max_chars: int = 1500
    condense_question: bool = True   # rewrite follow-ups with a reference ("sein Vater") before retrieval
    condense_model: str | None = None  # separate (faster) Ollama model for condense; None = main model
    session_timeout_minutes: int = 60  # idle sessions are dropped after this
    session_gc_interval_seconds: int = 300  # how often the cleanup runs


class LoggingConfig(BaseModel):
    """Log level, daily log file and third-party loggers to quieten."""
    level: str = "INFO"              # DEBUG | INFO | WARNING | ERROR
    file: str | None = "logs/lorekeeper.log"  # None = console only
    # One file per day; the day changes at this hour (after midnight, so a game night stays in one file)
    rotate_hour: int = 6
    backup_count: int = 30           # past days kept (lorekeeper.<YYYY-MM-DD>.log), older ones are deleted
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    # Set to ERROR. Keep in sync with config/settings.yaml — the YAML list replaces this one, it doesn't add to it.
    # httpx2: the HTTP client of the openai SDK (one line per Ollama call); google_genai: "AFC is enabled" per call
    suppress: list[str] = [
        "httpx", "httpx2", "httpcore", "openai", "google_genai", "chromadb", "huggingface_hub",
        "sentence_transformers", "safetensors", "model_utils",
    ]


class Settings(BaseSettings):
    """Central configuration schema powered by Pydantic settings.

    Reads and merges configuration from multiple layers:
    1. Environment variables (highest priority, strictly mapped via double-underscore e.g. LLM__PROVIDER)
    2. .env file
    3. YAML: settings.yaml with settings.local.yaml merged over it (passed in by ConfigManager)
    4. Code defaults (lowest priority)

    This acts as the unified, strictly-typed configuration object for the entire application.
    """
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ingestion: IngestionConfig = IngestionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    conversation: ConversationConfig = ConversationConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        """Priority: env vars > .env file > YAML > defaults."""
        return (
            kwargs["env_settings"],
            kwargs["dotenv_settings"],
            kwargs["init_settings"],  # YAML data passed via __init__
            kwargs["file_secret_settings"],
        )


class ConfigManager:
    """Manages the loading, merging, and saving of application configurations.

    Builds the Pydantic settings from ``settings.yaml`` plus the ``settings.local.yaml`` sidecar, loads the
    sources from the ``sources.yaml`` sidecar, and writes changes back: Settings page → ``settings.local.yaml``
    (``settings.yaml`` is never rewritten), sources → ``sources.yaml``, prompts → ``prompts.yaml``.
    """
    def __init__(
        self,
        settings_path: Path = Path("config/settings.yaml"),
        prompts_path: Path = Path("config/prompts.yaml"),
        sources_path: Path = Path("config/sources.yaml"),
    ):
        """Load settings (+ UI overrides sidecar), sources sidecar, and prompts from the given paths."""
        self._settings_path = settings_path
        # UI edits land here so the hand-written, commented settings.yaml is never rewritten
        self._local_settings_path = settings_path.with_name("settings.local.yaml")
        self._sources_path = sources_path
        self._prompts_path = prompts_path

        # YAML data is passed as init_settings (lowest priority after defaults)
        self.settings = Settings(**_deep_merge(_read_yaml(settings_path), _read_yaml(self._local_settings_path)))

        self._load_sources_sidecar()

        self._prompts_raw: dict = {}
        if prompts_path.exists():
            self._prompts_raw = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or {}

    def _load_sources_sidecar(self) -> None:
        """Load config/sources.yaml into settings.ingestion.sources (warns if there is none)."""
        if self._sources_path.exists():
            raw = yaml.safe_load(self._sources_path.read_text(encoding="utf-8")) or {}
            sources_raw = raw.get("sources", []) or []
            self.settings.ingestion.sources = [SourceConfig(**s) for s in sources_raw]
            return
        logger.warning(
            "No sources configured. Create config/sources.yaml or add sources "
            "via the UI (Sources page) to start indexing documents."
        )

    def save_sources(self) -> None:
        """Persist current ingestion.sources to config/sources.yaml."""
        data = {
            "sources": [s.model_dump(exclude_defaults=True) for s in self.settings.ingestion.sources]
        }
        atomic_write_text(self._sources_path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    def editable_snapshot(self) -> dict:
        """Current values of every UI-editable setting (``_EDITABLE_KEYS``) plus the read-only fields
        the Settings page displays (``_READ_ONLY_KEYS``) — derived from the live settings, so the
        snapshot can't drift from the allow-list. Secrets and paths are never included."""
        return _filter_allowed(self.settings.model_dump(), _deep_merge(_EDITABLE_KEYS, _READ_ONLY_KEYS))

    def save_settings(self, updates: dict) -> dict:
        """Merge ``updates`` into the live Settings and persist to the settings.local.yaml sidecar.

        Only keys listed in ``_EDITABLE_KEYS`` are accepted; unknown keys are
        dropped silently. The merged result is re-validated through Pydantic —
        type errors raise ``pydantic.ValidationError``. On success the running
        Settings instance is mutated in place so services holding references to
        the nested config objects (Retriever, ConversationManager, Providers)
        immediately see the new values.

        Ingest-time settings (``chunking.*``) are persisted but have no live
        effect until the next reindex.

        Returns the post-save editable snapshot.
        """
        filtered = _filter_allowed(updates, _EDITABLE_KEYS)
        if not filtered:
            return self.editable_snapshot()

        local = _deep_merge(_read_yaml(self._local_settings_path), filtered)

        # Validation: construct a fresh Settings from base + overrides so
        # Pydantic rejects bad values before we touch the live instance or
        # write to disk.
        Settings(**_deep_merge(_read_yaml(self._settings_path), local))

        # Apply in place so existing service references stay valid.
        _apply_updates(self.settings, filtered)

        atomic_write_text(
            self._local_settings_path,
            "# Written by the Settings page — overrides config/settings.yaml. Delete to reset.\n"
            + yaml.safe_dump(local, sort_keys=False, allow_unicode=True, default_flow_style=False),
        )

        return self.editable_snapshot()

    def save_prompts(self, prompts_dict: dict) -> None:
        """Persist prompts to config/prompts.yaml."""
        clean = {k: v for k, v in prompts_dict.items() if not k.startswith("_")}
        atomic_write_text(
            self._prompts_path,
            yaml.safe_dump(clean, sort_keys=False, allow_unicode=True, default_flow_style=False),
        )
        self._prompts_raw = clean

    @property
    def prompts(self):
        """Raw prompts dict as loaded from config/prompts.yaml."""
        return self._prompts_raw

    @property
    def prompts_path(self) -> Path:
        """The active prompts file — the prompt routes keep their variants next to it."""
        return self._prompts_path


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to a temp file next to ``path``, then ``os.replace`` it over the target.

    A crash or power loss mid-write leaves the old file intact instead of a truncated one —
    ``sources.yaml`` is gitignored and has no backup (roadmap 11.0 #9). Line endings as with
    ``Path.write_text`` (platform default).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_yaml(path: Path) -> dict:
    """Load a YAML mapping, or {} if the file is missing or empty."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _filter_allowed(updates: dict, allowed: dict) -> dict:
    """Return a copy of ``updates`` containing only keys that appear in the
    ``allowed`` tree. Nested dicts are recursed into; scalar leaves pass
    through when the corresponding allow-list entry is present (value None)."""
    if not isinstance(updates, dict):
        return {}
    result: dict = {}
    for key, value in updates.items():
        if key not in allowed:
            continue
        sub = allowed[key]
        if sub is None:
            result[key] = value
        elif isinstance(sub, dict) and isinstance(value, dict):
            nested = _filter_allowed(value, sub)
            if nested:
                result[key] = nested
    return result


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into a copy of ``base``. Dict values are
    merged key-wise; all other values are replaced wholesale."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_updates(target, updates: dict) -> None:
    """Mutate ``target`` (a Pydantic model) in place with values from ``updates``.
    Nested dicts recurse into nested model instances so object identity of
    sub-models is preserved — services holding references keep working."""
    for key, value in updates.items():
        current = getattr(target, key, None)
        if isinstance(value, dict) and isinstance(current, BaseModel):
            _apply_updates(current, value)
        else:
            setattr(target, key, value)


_default: ConfigManager | None = None


def get_default_config() -> ConfigManager:
    """Lazily construct the default ConfigManager.

    Unlike a module-level instance, this avoids reading YAML from disk at import time,
    preventing circular import explosions and unwanted IO during test initializations.
    """
    global _default
    if _default is None:
        _default = ConfigManager()
    return _default


def __getattr__(name: str):
    """Lazy attribute hook so `from src.config.manager import config_manager` still works without import-time IO."""
    if name == "config_manager":
        return get_default_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
