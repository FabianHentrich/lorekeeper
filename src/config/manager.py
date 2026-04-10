import logging
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = logging.getLogger(__name__)

# Allowed source groups. Starr gehalten — Erweiterung erfordert Code-Änderung.
SourceGroup = Literal["lore", "adventure", "rules"]


class SourceConfig(BaseModel):
    """A single ingestion source. Either a folder or a single file."""
    id: str                                     # Stable user-supplied identifier
    path: str                                   # File OR folder
    group: SourceGroup                          # UI filter group
    default_category: str | None = None         # Used for files & flat sources
    category_map: dict[str, str | dict[str, str]] = {}  # Top-folder -> category or {category, group}
    exclude_patterns: list[str] = []            # Additive to global excludes

    @field_validator("category_map")
    @classmethod
    def _validate_category_map(cls, v: dict) -> dict:
        valid_groups = {"lore", "adventure", "rules"}
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


class IngestionConfig(BaseModel):
    sources: list[SourceConfig] = []
    # DEPRECATED — kept only for backwards compatibility. If sources is empty
    # but document_paths is set, the loader migrates them on the fly.
    document_paths: list[str] = []
    supported_formats: list[str] = [".md", ".pdf", ".png", ".jpg", ".webp"]
    exclude_patterns: list[str] = [".obsidian/*", ".trash/*", "*alt.md", "*(1).md", "*.draft.*"]
    watch_for_changes: bool = False


class ChunkingConfig(BaseModel):
    strategy: str = "heading_aware"
    max_chunk_size: int = 256
    chunk_overlap: int = 30
    min_chunk_size: int = 20


class EmbeddingsConfig(BaseModel):
    model: str = "intfloat/multilingual-e5-base"
    device: str = "auto"
    batch_size: int = 64
    normalize: bool = True


class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    mode: str = "embedded"
    persist_directory: str = "./chroma_data"
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    collection_name: str = "lorekeeper"
    distance_metric: str = "cosine"


class RerankingConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    top_k_rerank: int = 8
    max_per_source: int = 3          # cap number of chunks from any single file (0 = unlimited)


class RetrievalConfig(BaseModel):
    top_k: int = 15
    score_threshold: float = 0.5
    reranking: RerankingConfig = RerankingConfig()


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
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
    fallback_provider: Optional[str] = "gemini"
    fallback_enabled: bool = False
    ollama: OllamaConfig = OllamaConfig()
    gemini: GeminiConfig = GeminiConfig()


class ConversationConfig(BaseModel):
    window_size: int = 8
    max_context_tokens: int = 4096
    condense_question: bool = True
    condense_model: Optional[str] = None
    session_timeout_minutes: int = 60
    session_gc_interval_seconds: int = 300


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[str] = "logs/lorekeeper.log"
    max_bytes: int = 10_485_760  # 10 MB
    backup_count: int = 5
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    suppress: list[str] = ["httpx", "httpcore", "chromadb", "huggingface_hub", "sentence_transformers", "safetensors", "model_utils"]


class UIConfig(BaseModel):
    title: str = "LoreKeeper"
    subtitle: str = "Frag deine Welt."
    api_url: str = "http://localhost:8000"


class Settings(BaseSettings):
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
    api: APIConfig = APIConfig()
    ui: UIConfig = UIConfig()

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
    def __init__(
        self,
        settings_path: Path = Path("config/settings.yaml"),
        prompts_path: Path = Path("config/prompts.yaml"),
        sources_path: Path = Path("config/sources.yaml"),
    ):
        self._settings_path = settings_path
        self._sources_path = sources_path

        yaml_data = {}
        if settings_path.exists():
            yaml_data = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}

        # YAML data is passed as init_settings (lowest priority after defaults)
        self.settings = Settings(**yaml_data)

        # Load sources sidecar (if present), then apply migration shim if needed
        self._load_sources_sidecar()

        self._prompts_raw = {}
        if prompts_path.exists():
            self._prompts_raw = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or {}

    def _load_sources_sidecar(self) -> None:
        """Load config/sources.yaml into settings.ingestion.sources.

        Falls back to migrating settings.ingestion.document_paths into
        Source entries (group=lore) when no sidecar exists. This shim emits a
        warning so users notice they should migrate.
        """
        if self._sources_path.exists():
            raw = yaml.safe_load(self._sources_path.read_text(encoding="utf-8")) or {}
            sources_raw = raw.get("sources", []) or []
            self.settings.ingestion.sources = [SourceConfig(**s) for s in sources_raw]
            return

        # No sidecar — try migration shim from deprecated document_paths.
        # If neither exists, sources stays empty and we warn.
        legacy = self.settings.ingestion.document_paths
        if legacy:
            _log.warning(
                "config/sources.yaml not found; migrating ingestion.document_paths "
                "to sources (group=lore). Create config/sources.yaml to silence this "
                "warning and customize per-source group/category mapping."
            )
            migrated = []
            for path in legacy:
                src_id = Path(path).name.lower().replace(" ", "_") or "source"
                migrated.append(SourceConfig(
                    id=src_id,
                    path=path,
                    group="lore",
                    default_category="misc",
                ))
            self.settings.ingestion.sources = migrated
        else:
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

    @property
    def prompts(self):
        return self._prompts_raw


config_manager = ConfigManager()
