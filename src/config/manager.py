from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseModel):
    document_paths: list[str] = ["./data/PnP-Welt"]
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
    suppress: list[str] = ["httpx", "httpcore", "chromadb", "huggingface_hub", "sentence_transformers"]


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
    ):
        yaml_data = {}
        if settings_path.exists():
            yaml_data = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}

        # YAML data is passed as init_settings (lowest priority after defaults)
        self.settings = Settings(**yaml_data)

        self._prompts_raw = {}
        if prompts_path.exists():
            self._prompts_raw = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or {}

    @property
    def prompts(self):
        return self._prompts_raw


config_manager = ConfigManager()
