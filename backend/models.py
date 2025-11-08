from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# Chat Models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Citation(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    chunk_text: str
    relevance_score: float
    chunk_id: str


class QueryRequest(BaseModel):
    question: str
    collection_name: str = "hr_policies"
    conversation_history: List[Message] = []
    include_citations: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    guardrail_triggered: bool = False
    rejection_reason: Optional[str] = None
    processing_time_ms: float


# Ingestion Models
class IngestRequest(BaseModel):
    collection_name: str
    guardrail_name: str
    force_reindex: bool = False


class DocumentStatus(BaseModel):
    filename: str
    status: str  # "success", "error", "skipped"
    chunks_created: int
    error_message: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    collection_name: str
    documents_processed: int
    total_chunks: int
    document_details: List[DocumentStatus]
    processing_time_seconds: float
    errors: List[str] = []


# Health Check
class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    cuda_version: Optional[str] = None
    embedding_model_loaded: bool
    active_collection: Optional[str] = None
    duckdb_status: str


# Embedding Models
class EmbeddingRequest(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    dimension: int
