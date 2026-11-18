"""
RAG (Retrieval-Augmented Generation) engine for DocuMind.
Handles intelligent chunking, vector indexing, semantic search retrieval,
and context synthesis for documents of any size.
"""

import logging
from typing import AsyncGenerator, List, Dict
from app.core.config import settings
from app.models.schemas import AnalysisMode, DocumentChunk, SearchResult
from app.services.document_parser import ExtractedDocument
from app.services.text_chunker import TextChunker
from app.services.embeddings import SemanticEmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService

logger = logging.getLogger("documind.rag")


class RAGEngine:
    """Orchestrates document vectorization, semantic retrieval, and augmented generation."""

    def __init__(
        self,
        llm_service: LLMService,
        chunker: TextChunker,
        vector_store: VectorStore,
    ):
        self.llm_service = llm_service
        self.chunker = chunker
        self.vector_store = vector_store

    def is_large_document(self, text: str) -> bool:
        """Determines if the document should be routed through RAG pipeline."""
        return len(text) > settings.DIRECT_PROCESSING_CHAR_LIMIT

    def _get_mode_queries(self, mode: AnalysisMode) -> List[str]:
        """Generates semantic retrieval queries tailored to the analysis mode."""
        if mode == AnalysisMode.SUMMARIZE:
            return [
                "executive summary main thesis core argument introduction",
                "key findings results analysis main discussion",
                "conclusion implications final outcome recommendations",
            ]
        elif mode == AnalysisMode.KEY_POINTS:
            return [
                "critical takeaways key points major results",
                "decisions milestones metrics actionable items",
                "bottom line essential conclusion",
            ]
        elif mode == AnalysisMode.SIMPLIFY:
            return [
                "core concept fundamental explanation purpose",
                "practical examples real-world meaning overview",
            ]
        return ["main ideas summary key points"]

    def retrieve_context(self, mode: AnalysisMode, top_k_per_query: int = 3) -> List[DocumentChunk]:
        """
        Executes multi-query vector search and returns deduplicated,
        chronologically ordered chunks to preserve narrative flow.
        """
        queries = self._get_mode_queries(mode)
        seen_ids = set()
        retrieved_chunks: List[DocumentChunk] = []

        for q in queries:
            results = self.vector_store.similarity_search(query=q, top_k=top_k_per_query)
            for res in results:
                cid = res.chunk.chunk_id
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    retrieved_chunks.append(res.chunk)

        # Sort chronologically by chunk_id to maintain natural reading order
        retrieved_chunks.sort(key=lambda c: c.chunk_id)
        return retrieved_chunks

    def build_rag_prompt(
        self,
        mode: AnalysisMode,
        doc: ExtractedDocument,
        chunks: List[DocumentChunk],
    ) -> str:
        """
        Assembles retrieved chunks and metadata into an augmented prompt.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f" (Page {chunk.page_number})" if chunk.page_number else ""
            context_parts.append(f"--- Section {i}{page_info} ---\n{chunk.text}")

        joined_context = "\n\n".join(context_parts)
        meta_info = f"Document length: {doc.page_count} pages, ~{doc.word_count} words."

        prompt = (
            f"Note: This is a large document ({meta_info}). The most relevant sections have been "
            f"extracted and ordered below using semantic vector search:\n\n"
            f"{joined_context}\n\n"
            f"Based on the extracted sections above, perform the requested analysis comprehensively."
        )
        return prompt

    async def process_and_stream(
        self,
        doc: ExtractedDocument,
        mode: AnalysisMode,
    ) -> AsyncGenerator[str, None]:
        """
        Main pipeline: decides between direct streaming (for compact documents)
        and semantic vector RAG (for large documents).
        """
        text = doc.full_text

        # 1. Compact / standard document -> direct streaming
        if not self.is_large_document(text):
            logger.info("Processing document via direct LLM stream (< %d chars)", settings.DIRECT_PROCESSING_CHAR_LIMIT)
            async for delta in self.llm_service.stream_analysis(mode=mode, content=text):
                yield delta
            return

        # 2. Large document -> RAG pipeline
        logger.info(
            "Processing large document (%d chars) via Vector RAG pipeline",
            len(text),
        )

        # Chunk document
        chunks = self.chunker.chunk_document(doc)
        logger.info("Segmented document into %d semantic chunks", len(chunks))

        # Index chunks in vector store
        self.vector_store.clear()
        self.vector_store.add_chunks(chunks)

        # Retrieve relevant sections
        relevant_chunks = self.retrieve_context(mode=mode, top_k_per_query=settings.TOP_K_CHUNKS)
        logger.info("Retrieved %d prioritized chunks for RAG synthesis", len(relevant_chunks))

        # Build augmented prompt
        augmented_content = self.build_rag_prompt(mode=mode, doc=doc, chunks=relevant_chunks)

        # Stream through LLM with mode system instructions
        async for delta in self.llm_service.stream_analysis(mode=mode, content=augmented_content):
            yield delta
