"""
Unit tests for Security utilities and validation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.security import sanitize_text, validate_file_upload
from app.core.exceptions import InvalidFileError, FileSizeLimitExceededError


class TestSecurity(unittest.TestCase):

    def test_sanitize_text_strips_null_bytes(self):
        dirty = "Hello\x00World!\x01\x02Test"
        cleaned = sanitize_text(dirty)
        self.assertEqual(cleaned, "HelloWorld!Test")

    def test_sanitize_text_normalizes_line_endings(self):
        dirty = "Line 1\r\nLine 2\rLine 3\n"
        cleaned = sanitize_text(dirty)
        self.assertEqual(cleaned, "Line 1\nLine 2\nLine 3")

    def test_validate_file_upload_allowed_extensions(self):
        # Should not raise
        validate_file_upload("report.pdf", 1024, b"%PDF-1.5")
        validate_file_upload("notes.txt", 1024, b"plain text")

    def test_validate_file_upload_forbidden_extensions(self):
        with self.assertRaises(InvalidFileError):
            validate_file_upload("malware.exe", 1024)
        with self.assertRaises(InvalidFileError):
            validate_file_upload("script.py", 1024)

    def test_validate_file_upload_invalid_pdf_magic_bytes(self):
        with self.assertRaises(InvalidFileError):
            validate_file_upload("fake.pdf", 1024, b"Not a real PDF stream")

    def test_validate_file_upload_size_limit(self):
        with self.assertRaises(FileSizeLimitExceededError):
            # 20 MB file exceeds default 15 MB limit
            validate_file_upload("huge.txt", 20 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
