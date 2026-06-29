"""
Pydantic Schemas for Request/Response validation
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Document Schemas ────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""
    document_id: int
    filename: str
    original_filename: str
    file_size: int
    status: str
    message: str


class DocumentStatusResponse(BaseModel):
    """Document indexing status."""
    document_id: int
    filename: str
    original_filename: str
    file_size: int
    page_count: Optional[int] = None
    doc_type: Optional[str] = None
    language: str = "fr"
    status: str
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    """List of indexed documents."""
    total: int
    documents: list[DocumentStatusResponse]


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""
    document_id: int
    message: str


# ─── Query Schemas ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request to ask a question."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Question in natural language about your financial documents",
        example="Quel est le ratio de solvabilité mentionné dans le rapport annuel ?",
    )
    doc_ids: Optional[list[int]] = Field(
        default=None,
        description="Optional list of document IDs to restrict the search scope",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve (default: from settings)",
    )
    language: Optional[str] = Field(
        default="fr",
        description="Response language (fr | en)",
    )


class SourceChunk(BaseModel):
    """A retrieved source chunk used to build the answer."""
    document_id: int
    document_name: str
    page: Optional[int] = None
    chunk_index: int
    content: str
    similarity_score: float


class QueryResponse(BaseModel):
    """Response to a natural language query."""
    question: str
    answer: str
    sources: list[SourceChunk]
    model_used: str
    chunks_retrieved: int
    processing_time_ms: float
    has_answer: bool = True
    warning: Optional[str] = None


# ─── Health Schemas ───────────────────────────────────────────────────────────

class OllamaStatus(BaseModel):
    available: bool
    model: str
    embed_model: str
    error: Optional[str] = None


class VectorStoreStatus(BaseModel):
    available: bool
    collection: str
    document_count: int
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall system health."""
    status: str  # healthy | degraded | unhealthy
    version: str
    environment: str
    ollama: OllamaStatus
    vector_store: VectorStoreStatus
    database: dict[str, Any]
