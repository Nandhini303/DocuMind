"""
FastAPI dependency injection providers.
"""

from app.core.config import settings
from app.services.llm_service import LLMService
from app.services.text_chunker import TextChunker
from app.services.embeddings import SemanticEmbeddingService
from app.services.vector_store import VectorStore
from app.services.rag_engine import RAGEngine

# Singleton service instances for re-use across requests
_llm_service = LLMService()
_chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
_embedding_service = SemanticEmbeddingService()


def get_llm_service() -> LLMService:
    return _llm_service


def get_chunker() -> TextChunker:
    return _chunker


def get_vector_store() -> VectorStore:
    return VectorStore(embedding_service=_embedding_service)


def get_rag_engine() -> RAGEngine:
    return RAGEngine(
        llm_service=get_llm_service(),
        chunker=get_chunker(),
        vector_store=get_vector_store(),
    )
