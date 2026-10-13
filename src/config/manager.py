import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = logging.getLogger(__name__)

# Allowed source groups. Starr gehalten — Erweiterung erfordert Code-Änderung.
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
    ocr_enabled: bool = True
    ocr_language: str = "deu"
    ocr_dpi: int = 300
    extract_images: bool = True
    image_format: str = "png"


class IngestionConfig(BaseModel):
    sources: list[SourceConfig] = []
    supported_formats: list[str] = [".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp"]
    exclude_patterns: list[str] = [".obsidian/*", ".trash/*", "*alt.md", "*(1).md", "*.draft.*"]
    pdf: PdfConfig = PdfConfig()


class ChunkingConfig(BaseModel):
    strategy: str = "heading_aware"
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
    model: str = "intfloat/multilingual-e5-base"
    device: str = "auto"
    batch_size: int = 64
    normalize: bool = True


class VectorStoreConfig(BaseModel):
    persist_directory: str = "./chroma_data"   # embedded ChromaDB inside the backend process
    collection_name: str = "lorekeeper"
    distance_metric: str = "cosine"


class RerankingConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # "directml": GPU via onnxruntime-directml (falls back to torch on the CPU if unavailable)
    backend: Literal["torch", "directml"] = "torch"
    top_k_rerank: int = 8
    max_per_source: int = 3          # cap number of chunks from any single file (0 = unlimited)


class HybridSearchConfig(BaseModel):
    enabled: bool = False
    bm25_weight: float = 0.3         # BM25 share in RRF (0.0 = pure vector, 1.0 = pure BM25)
    bm25_top_k: int = 15             # how many BM25 candidates to fetch


class SectionExpansionConfig(BaseModel):
    enabled: bool = True
    max_chunks: int = 25             # sections larger than this are not expanded (protects the LLM context)
    top_rank: int = 3                # only expand if a hit is among the first N ranked chunks (0 = no limit)


class RulesSlotConfig(BaseModel):
    # Questions naming a vault entity *and* a rule term ("Was bewirkt Ritterlichkeit bei Escanor?"):
    # the entity's file fills every vector candidate, the rulebook never reaches the reranker.
    enabled: bool = False
    candidates: int = 8              # extra vector candidates from the "rules" group, reranked with the rest
    min_score: float = -5.0          # reserve one top_k_rerank slot for the best rules chunk if it scores ≥ this


class RetrievalConfig(BaseModel):
    top_k: int = 15
    score_threshold: float = 0.5
    reranking: RerankingConfig = RerankingConfig()
    hybrid: HybridSearchConfig = HybridSearchConfig()
    section_expansion: SectionExpansionConfig = SectionExpansionConfig()
    rules_slot: RulesSlotConfig = RulesSlotConfig()


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:14b"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 1024
    timeout: int = 120


class GeminiConfig(BaseModel):
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 1024
    timeout: int = 30


class LLMConfig(BaseModel):
    provider: str = "ollama"
    fallback_provider: str | None = "gemini"
    fallback_enabled: bool = False
    ollama: OllamaConfig = OllamaConfig()
    gemini: GeminiConfig = GeminiConfig()


class ConversationConfig(BaseModel):
    window_size: int = 8
    max_context_tokens: int = 4096
    # Answer prompt: last N QA pairs (0 = off); older answers cut at max_chars — more dilutes the sources
    answer_history_pairs: int = 1
    answer_history_max_chars: int = 1500
    condense_question: bool = True
    condense_model: str | None = None
    session_timeout_minutes: int = 60
    session_gc_interval_seconds: int = 300


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str | None = "logs/lorekeeper.log"
    max_bytes: int = 10_485_760  # 10 MB
    backup_count: int = 5
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    suppress: list[str] = [
        "httpx", "httpcore", "chromadb", "huggingface_hub", "sentence_transformers", "safetensors", "model_utils",
    ]


class Settings(BaseSettings):
    """Central configuration schema powered by Pydantic settings.

    Reads and merges configuration from multiple layers:
    1. Environment variables (highest priority, strictly mapped via double-underscore e.g. LLM__PROVIDER)
    2. .env file
    3. Base YAML file
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

    It orchestrates the initialization of the Pydantic Settings layer with the
    base `settings.yaml`, handles legacy migrations for the ingestion sources via a
    sidecar `sources.yaml` file, and provides IO methods for updating prompts and sources dynamically.
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
        _log.warning(
            "No sources configured. Create config/sources.yaml or add sources "
            "via the UI (Sources page) to start indexing documents."
        )

    def save_sources(self) -> None:
        """Persist current ingestion.sources to config/sources.yaml."""
        self._sources_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sources": [s.model_dump(exclude_defaults=True) for s in self.settings.ingestion.sources]
        }
        self._sources_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

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

        self._local_settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._local_settings_path.write_text(
            "# Written by the Settings page — overrides config/settings.yaml. Delete to reset.\n"
            + yaml.safe_dump(local, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

        return self.editable_snapshot()

    def save_prompts(self, prompts_dict: dict) -> None:
        """Persist prompts to config/prompts.yaml."""
        self._prompts_path.parent.mkdir(parents=True, exist_ok=True)
        clean = {k: v for k, v in prompts_dict.items() if not k.startswith("_")}
        self._prompts_path.write_text(
            yaml.safe_dump(clean, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
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
