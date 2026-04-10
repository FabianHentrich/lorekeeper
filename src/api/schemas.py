from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    metadata_filters: dict | None = None
    top_k: int | None = None
    top_k_rerank: int | None = None
    max_per_source: int | None = None


class SourceReference(BaseModel):
    file: str
    source_path: str = ""
    document_type: str
    heading: str | None = None
    chunk_preview: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    session_id: str
    retrieval_scores: list[float]
    model_used: str
    latency_ms: float


class IngestJobResponse(BaseModel):
    job_id: str
    status: str


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str = "starting"
    documents_processed: int = 0
    documents_total: int = 0
    chunks_created: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    errors: list[str] = []
    duration_seconds: float = 0.0


class HealthResponse(BaseModel):
    status: str
    chromadb: bool
    llm: bool
    sources_configured: bool = True


class StatsResponse(BaseModel):
    chunk_count: int


class ProviderInfo(BaseModel):
    provider: str
    model: str


class GeminiKeyStatus(BaseModel):
    has_key: bool
    source: str = "none"


class SidebarState(BaseModel):
    health: HealthResponse
    provider: ProviderInfo
    gemini_status: GeminiKeyStatus
    available_categories: list[str]


class SwitchProviderRequest(BaseModel):
    provider: str  # "ollama" | "gemini"
