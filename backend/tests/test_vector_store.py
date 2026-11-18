"""
Unit tests for SemanticEmbeddingService and VectorStore.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import DocumentChunk
from app.services.embeddings import SemanticEmbeddingService
from app.services.vector_store import VectorStore


class TestEmbeddingsAndVectorStore(unittest.TestCase):

    def setUp(self):
        self.embedding_service = SemanticEmbeddingService(dimensions=128)
        self.store = VectorStore(embedding_service=self.embedding_service)

    def test_embedding_vector_properties(self):
        text = "DocuMind uses semantic search and vector retrieval."
        vec = self.embedding_service.embed_text(text)
        self.assertEqual(len(vec), 128)
        # Verify L2 normalization: norm should equal 1.0 (within float tolerance)
        norm_squared = sum(x * x for x in vec)
        self.assertAlmostEqual(norm_squared, 1.0, places=4)

    def test_empty_text_embedding(self):
        vec = self.embedding_service.embed_text("")
        self.assertEqual(len(vec), 128)
        self.assertEqual(sum(vec), 0.0)

    def test_semantic_similarity(self):
        text_ai_1 = "Artificial intelligence and machine learning neural networks."
        text_ai_2 = "Deep learning models, algorithms, and AI systems."
        text_unrelated = "Cooking recipes for chocolate cake and vanilla cupcakes."

        vec_ai_1 = self.embedding_service.embed_text(text_ai_1)
        vec_ai_2 = self.embedding_service.embed_text(text_ai_2)
        vec_unrelated = self.embedding_service.embed_text(text_unrelated)

        sim_ai = VectorStore._cosine_similarity(vec_ai_1, vec_ai_2)
        sim_diff = VectorStore._cosine_similarity(vec_ai_1, vec_unrelated)

        self.assertGreater(sim_ai, sim_diff)

    def test_vector_store_retrieval(self):
        chunks = [
            DocumentChunk(chunk_id=1, text="FastAPI backend with Server-Sent Events streaming.", char_count=52),
            DocumentChunk(chunk_id=2, text="Delicious Italian pasta with tomato and basil sauce.", char_count=54),
            DocumentChunk(chunk_id=3, text="Docker containerization and automated deployments on cloud.", char_count=60),
        ]
        self.store.add_chunks(chunks)
        self.assertEqual(len(self.store), 3)

        results = self.store.similarity_search("Docker container cloud hosting", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, 3)

        results_code = self.store.similarity_search("FastAPI SSE event stream", top_k=1)
        self.assertEqual(len(results_code), 1)
        self.assertEqual(results_code[0].chunk.chunk_id, 1)


if __name__ == "__main__":
    unittest.main()
