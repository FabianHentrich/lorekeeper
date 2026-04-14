from pydantic import BaseModel


class QAPair(BaseModel):
    id: str
    question: str
    source_type: str = "markdown"
    category: str = ""
    expected_sources: list[str] = []
    expected_answer_contains: list[str] = []
    notes: str = ""


class QAPairList(BaseModel):
    pairs: list[QAPair]


class RetrievalPreviewRequest(BaseModel):
    question: str
    top_k: int = 15
    top_k_rerank: int = 8
    max_per_source: int = 3
    hybrid: bool | None = None


class RetrievedChunkResponse(BaseModel):
    source_file: str
    document_type: str
    heading: str | None
    score: float
    content_preview: str


class RetrievalPreviewResponse(BaseModel):
    chunks: list[RetrievedChunkResponse]
    latency_ms: float


class EvalJobRequest(BaseModel):
    top_k: int = 15
    top_k_rerank: int = 8
    max_per_source: int = 3
    eval_type: str = "retrieval"
    hybrid: bool | None = None


class EvalJobResponse(BaseModel):
    job_id: str
    status: str


class EvalJobStatus(BaseModel):
    job_id: str
    status: str = "queued"
    progress: int = 0
    total: int = 0
    error: str | None = None
    result_file: str | None = None


class EvalResultSummary(BaseModel):
    filename: str
    timestamp: str
    eval_type: str
    hit_rate: float
    total_questions: int
    config: dict
