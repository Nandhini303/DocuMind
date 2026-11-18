"""
In-memory vector store for chunk indexing and similarity search.
"""

from typing import List, Optional, Tuple
from app.models.schemas import DocumentChunk, SearchResult
from app.services.embeddings import BaseEmbeddingService, SemanticEmbeddingService


class VectorStore:
    """In-memory vector index supporting cosine similarity retrieval."""

    def __init__(self, embedding_service: Optional[BaseEmbeddingService] = None):
        self.embedding_service = embedding_service or SemanticEmbeddingService()
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []

    def clear(self) -> None:
        """Empties the index."""
        self.chunks = []
        self.embeddings = []

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Embeds and indexes a single chunk."""
        emb = self.embedding_service.embed_text(chunk.text)
        self.chunks.append(chunk)
        self.embeddings.append(emb)

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Batch embeds and indexes a list of chunks."""
        if not chunks:
            return
        embeddings = self.embedding_service.embed_batch([c.text for c in chunks])
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Computes dot product between two unit-normalized vectors."""
        if len(vec_a) != len(vec_b):
            return 0.0
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Retrieves top_k chunks most semantically relevant to the query string.
        """
        if not self.chunks or not query:
            return []

        query_emb = self.embedding_service.embed_text(query)
        scored: List[Tuple[float, DocumentChunk]] = []

        for emb, chunk in zip(self.embeddings, self.chunks):
            score = self._cosine_similarity(query_emb, emb)
            if score >= threshold:
                scored.append((score, chunk))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(chunk=chunk, similarity_score=round(score, 4))
            for score, chunk in scored[:top_k]
        ]

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Returns all indexed chunks."""
        return list(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)
