"""
Pydantic models and schemas for requests, responses, and internal data structures.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AnalysisMode(str, Enum):
    """Supported analysis modes matching frontend contract."""
    SUMMARIZE = "summarize"
    KEY_POINTS = "key_points"
    SIMPLIFY = "simplify"


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = "ok"
    model: str
    api_key_configured: bool
    provider: str = "anthropic"
    version: str = "2.0.0"
    rag_enabled: bool = True


class DocumentChunk(BaseModel):
    """Represents a chunk of a document stored in the vector index."""
    chunk_id: int
    text: str
    page_number: Optional[int] = None
    char_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Vector search retrieval result."""
    chunk: DocumentChunk
    similarity_score: float
