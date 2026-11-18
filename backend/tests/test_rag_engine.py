"""
Unit tests for RAGEngine.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import AnalysisMode, DocumentChunk
from app.services.document_parser import ExtractedDocument
from app.services.text_chunker import TextChunker
from app.services.vector_store import VectorStore
from app.services.embeddings import SemanticEmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_engine import RAGEngine
from app.core.config import settings


class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        self.llm_service = LLMService()
        self.chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        self.vector_store = VectorStore(embedding_service=SemanticEmbeddingService(dimensions=64))
        self.rag = RAGEngine(
            llm_service=self.llm_service,
            chunker=self.chunker,
            vector_store=self.vector_store,
        )

    def test_document_size_routing(self):
        short_text = "Short text under the threshold limit."
        long_text = "A" * (settings.DIRECT_PROCESSING_CHAR_LIMIT + 100)

        self.assertFalse(self.rag.is_large_document(short_text))
        self.assertTrue(self.rag.is_large_document(long_text))

    def test_rag_prompt_construction(self):
        doc = ExtractedDocument(
            full_text="Complete document text.",
            pages=[(1, "Page 1"), (2, "Page 2")],
        )
        chunks = [
            DocumentChunk(chunk_id=0, text="Section 1 details.", page_number=1, char_count=18),
            DocumentChunk(chunk_id=1, text="Section 2 details.", page_number=2, char_count=18),
        ]
        prompt = self.rag.build_rag_prompt(mode=AnalysisMode.SUMMARIZE, doc=doc, chunks=chunks)
        self.assertIn("Section 1 (Page 1)", prompt)
        self.assertIn("Section 2 (Page 2)", prompt)
        self.assertIn("Document length: 2 pages", prompt)

    def test_retrieve_context_deduplication_and_order(self):
        chunks = [
            DocumentChunk(chunk_id=0, text="Introduction to the research study and objectives.", page_number=1, char_count=50),
            DocumentChunk(chunk_id=1, text="Methodology of data collection and statistical tools.", page_number=2, char_count=52),
            DocumentChunk(chunk_id=2, text="Results and analytical findings showing significant gains.", page_number=3, char_count=58),
            DocumentChunk(chunk_id=3, text="Conclusion, future outlook, and final policy recommendations.", page_number=4, char_count=61),
        ]
        self.vector_store.add_chunks(chunks)

        retrieved = self.rag.retrieve_context(mode=AnalysisMode.SUMMARIZE, top_k_per_query=2)
        # Check deduplication
        retrieved_ids = [c.chunk_id for c in retrieved]
        self.assertEqual(len(retrieved_ids), len(set(retrieved_ids)))
        # Check chronological sort
        self.assertEqual(retrieved_ids, sorted(retrieved_ids))


if __name__ == "__main__":
    unittest.main()
