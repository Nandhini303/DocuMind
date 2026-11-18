"""
Unit tests for TextChunker service.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.text_chunker import TextChunker
from app.services.document_parser import ExtractedDocument


class TestTextChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    def test_short_text_single_chunk(self):
        text = "This is a short text that fits in one chunk."
        chunks = self.chunker.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, 0)
        self.assertEqual(chunks[0].text, text)

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(self.chunker.split_text(""), [])
        self.assertEqual(self.chunker.split_text("   "), [])

    def test_large_text_multi_chunks_with_overlap(self):
        text = (
            "Paragraph one is discussing machine learning fundamentals. "
            "It explains vector embeddings and similarity search. "
            "Paragraph two goes into transformer architectures, attention mechanisms, "
            "and long context window handling in large language models. "
            "Paragraph three concludes with practical deployment on cloud platforms."
        )
        chunks = self.chunker.split_text(text)
        self.assertGreater(len(chunks), 1)
        # Verify chunk IDs are sequential
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_id, idx)
            self.assertLessEqual(len(chunk.text), 120)  # within margin of chunk size

    def test_chunk_document_preserves_page_numbers(self):
        doc = ExtractedDocument(
            full_text="Page 1 text.\n\nPage 2 text.",
            pages=[(1, "Page 1 content here with details."), (2, "Page 2 content here with details.")],
        )
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[1].page_number, 2)


if __name__ == "__main__":
    unittest.main()
