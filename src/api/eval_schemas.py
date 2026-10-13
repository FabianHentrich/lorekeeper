"""Request and response models of the /eval API.

Retrieval parameters left at None take the value from settings.yaml (resolved by the retriever).
"""
from typing import Literal

from pydantic import BaseModel

from src.api.access import Role


class QAPair(BaseModel):
    """One golden-set question with what a hit must contain (evaluation/qa_pairs.yaml)."""
    id: str
    question: str
    source_type: str = "markdown"
    category: str = ""
    expected_sources: list[str] = []
    expected_answer_contains: list[str] = []
    expected_section: str = ""  # numbered heading prefix (e.g. "4.10") or heading name a retrieved chunk must lie in
    expected_source_ids: list[str] = []  # optional: the expected file must come from one of these sources
    notes: str = ""


class QAPairList(BaseModel):
    """The whole golden set — PUT replaces the file."""
    pairs: list[QAPair]


class RetrievalPreviewRequest(BaseModel):
    """One question against the retriever without the LLM."""
    question: str
    # None = the value from settings.yaml (retrieval.top_k / reranking.top_k_rerank / reranking.max_per_source)
    top_k: int | None = None
    top_k_rerank: int | None = None
    max_per_source: int | None = None
    hybrid: bool | None = None
    metadata_filters: dict | None = None  # same as /query — e.g. the player-mode source filter
    role: Role = "game_master"  # same as /query


class RetrievedChunkResponse(BaseModel):
    """A preview hit; ``content_preview`` is the start of the chunk text."""
    source_id: str = ""
    source_file: str
    document_type: str
    heading: str | None
    score: float
    content_preview: str


class RetrievalPreviewResponse(BaseModel):
    """Preview hits in final order plus the retrieval time."""
    chunks: list[RetrievedChunkResponse]
    latency_ms: float


class InspectedChunk(BaseModel):
    """A stored chunk with full text and metadata (chunk inspector)."""
    id: str
    chunk_index: int
    heading: str
    chars: int
    content: str
    metadata: dict


class ChunkListResponse(BaseModel):
    """All chunks of one file in document order."""
    source_id: str
    source_file: str
    chunks: list[InspectedChunk]


class EvalJobRequest(BaseModel):
    """Start a golden-set run: ``retrieval`` (no LLM) or ``e2e`` (through /query)."""
    # None = the value from settings.yaml, as in RetrievalPreviewRequest
    top_k: int | None = None
    top_k_rerank: int | None = None
    max_per_source: int | None = None
    eval_type: Literal["retrieval", "e2e"] = "retrieval"  # anything else → 422
    hybrid: bool | None = None


class EvalJobResponse(BaseModel):
    """Id of the started eval job."""
    job_id: str
    status: str


class EvalJobStatus(BaseModel):
    """Progress of an eval job; ``result_file`` is set once the report is saved."""
    job_id: str
    status: str = "queued"
    progress: int = 0
    total: int = 0
    error: str | None = None
    result_file: str | None = None


class EvalResultSummary(BaseModel):
    """Key figures of a stored report for the results list."""
    filename: str
    timestamp: str
    eval_type: str
    hit_rate: float
    total_questions: int
    config: dict
