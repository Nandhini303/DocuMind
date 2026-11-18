"""
Unit tests for DocumentParser service.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_parser import DocumentParser, ExtractedDocument
from app.core.exceptions import DocumentParsingError


class TestDocumentParser(unittest.TestCase):

    def test_clean_text_ligatures(self):
        raw = "The \ufb01rst \ufb02oating o\ufb03ce had an e\ufb00ect."
        cleaned = DocumentParser.clean_text(raw)
        self.assertIn("first", cleaned)
        self.assertIn("floating", cleaned)
        self.assertIn("office", cleaned)
        self.assertIn("effect", cleaned)

    def test_clean_text_hyphenation(self):
        raw = "This is an infor-\n  mation document with experi-\n   ments."
        cleaned = DocumentParser.clean_text(raw)
        self.assertIn("information", cleaned)
        self.assertIn("experiments", cleaned)

    def test_clean_text_whitespace(self):
        raw = "Line 1   with   spaces\n\n\n\nLine 2"
        cleaned = DocumentParser.clean_text(raw)
        self.assertEqual(cleaned, "Line 1 with spaces\n\nLine 2")

    def test_parse_plain_text_utf8(self):
        content = "DocuMind analyzes artificial intelligence research papers.".encode("utf-8")
        doc = DocumentParser.parse_plain_text(content)
        self.assertIsInstance(doc, ExtractedDocument)
        self.assertEqual(doc.page_count, 1)
        self.assertIn("DocuMind", doc.full_text)
        self.assertGreater(doc.word_count, 5)

    def test_parse_plain_text_empty(self):
        with self.assertRaises(DocumentParsingError):
            DocumentParser.parse_plain_text(b"")

    def test_parse_pdf_invalid_bytes(self):
        with self.assertRaises(DocumentParsingError):
            DocumentParser.parse_pdf(b"not a real pdf content")

    def test_parse_bytes_dispatch(self):
        content = "Sample report text".encode("utf-8")
        doc = DocumentParser.parse_bytes(content, filename="test.txt")
        self.assertEqual(doc.full_text, "Sample report text")


if __name__ == "__main__":
    unittest.main()
