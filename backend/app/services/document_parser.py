"""
Document parser service.
Robust extraction and normalization for PDF documents and plain text files.
"""

import io
import re
from typing import List, Tuple, Dict, Any, Optional
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import DocumentParsingError


class ExtractedDocument:
    """Container for extracted document contents and metadata."""

    def __init__(
        self,
        full_text: str,
        pages: List[Tuple[int, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.full_text = full_text
        self.pages = pages  # List of (page_number, text)
        self.metadata = metadata or {}
        self.page_count = len(pages)
        self.char_count = len(full_text)
        self.word_count = len(full_text.split()) if full_text else 0

    def __repr__(self) -> str:
        return (
            f"<ExtractedDocument pages={self.page_count} "
            f"words={self.word_count} chars={self.char_count}>"
        )


class DocumentParser:
    """Service to safely parse and normalize documents."""

    # Common unicode ligature mappings
    LIGATURES = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
        "\u2014": " — ",
        "\u2013": " – ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Cleans extracted text:
        - Resolves ligatures
        - Normalizes line breaks and whitespace
        - Reconstructs words broken by line-break hyphens
        - Cleans non-printable artifacts
        """
        if not text:
            return ""

        # Replace ligatures and common unicode replacements
        for lig, repl in cls.LIGATURES.items():
            text = text.replace(lig, repl)

        # Re-join hyphenated linebreaks: e.g. "experi-\nment" -> "experiment"
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excessive spaces within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

        # Collapse more than 2 consecutive newlines into 2
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @classmethod
    def parse_pdf(cls, raw_bytes: bytes) -> ExtractedDocument:
        """
        Extracts structured text from PDF raw bytes.
        Handles encrypted (blank password), multi-page layout, and metadata.
        """
        if not raw_bytes:
            raise DocumentParsingError("Empty PDF file.")

        try:
            stream = io.BytesIO(raw_bytes)
            reader = PdfReader(stream)

            # Handle blank-password encrypted files
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise DocumentParsingError("PDF is encrypted with a password and cannot be read.")

            pages: List[Tuple[int, str]] = []
            metadata: Dict[str, Any] = {}

            # Read document info if available
            if reader.metadata:
                metadata = {
                    "title": str(reader.metadata.title or ""),
                    "author": str(reader.metadata.author or ""),
                    "creator": str(reader.metadata.creator or ""),
                }

            page_texts: List[str] = []
            for idx, page in enumerate(reader.pages, start=1):
                try:
                    extracted = page.extract_text() or ""
                except Exception:
                    # Graceful degradation per page
                    extracted = ""

                cleaned_page = cls.clean_text(extracted)
                if cleaned_page:
                    pages.append((idx, cleaned_page))
                    page_texts.append(cleaned_page)

            full_text = "\n\n".join(page_texts).strip()

            if not full_text:
                raise DocumentParsingError(
                    "The PDF appears to be empty or contains only scanned images without selectable text."
                )

            return ExtractedDocument(full_text=full_text, pages=pages, metadata=metadata)

        except DocumentParsingError:
            raise
        except PdfReadError as exc:
            raise DocumentParsingError(f"Corrupt or invalid PDF file: {exc}")
        except Exception as exc:
            raise DocumentParsingError(f"Failed to process PDF: {exc}")

    @classmethod
    def parse_plain_text(cls, raw_bytes: bytes) -> ExtractedDocument:
        """
        Decodes plain text with automatic fallback encodings.
        """
        if not raw_bytes:
            raise DocumentParsingError("Empty text file.")

        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "ascii"]
        decoded_text = ""

        for enc in encodings:
            try:
                decoded_text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if not decoded_text:
            decoded_text = raw_bytes.decode("utf-8", errors="replace")

        cleaned = cls.clean_text(decoded_text)
        if not cleaned:
            raise DocumentParsingError("Text file contains no readable content.")

        pages = [(1, cleaned)]
        return ExtractedDocument(full_text=cleaned, pages=pages)

    @classmethod
    def parse_bytes(cls, raw_bytes: bytes, filename: Optional[str] = None) -> ExtractedDocument:
        """
        Dispatch parser according to filename extension or magic bytes.
        """
        if filename and filename.lower().endswith(".pdf"):
            return cls.parse_pdf(raw_bytes)
        elif raw_bytes.startswith(b"%PDF"):
            return cls.parse_pdf(raw_bytes)
        else:
            return cls.parse_plain_text(raw_bytes)
