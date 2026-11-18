"""
Text chunker service.
Recursively segments documents into semantically coherent chunks with overlap.
"""

import re
from typing import List, Optional
from app.models.schemas import DocumentChunk
from app.services.document_parser import ExtractedDocument


class TextChunker:
    """Recursively splits text into overlapping chunks respecting linguistic boundaries."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, page_number: Optional[int] = None, start_chunk_id: int = 0) -> List[DocumentChunk]:
        """
        Splits a single block of text into chunks.
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [
                DocumentChunk(
                    chunk_id=start_chunk_id,
                    text=text,
                    page_number=page_number,
                    char_count=len(text),
                )
            ]

        chunks: List[DocumentChunk] = []
        current_idx = 0
        chunk_counter = start_chunk_id

        # Candidate separators ordered by semantic strength
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

        while current_idx < len(text):
            end_idx = min(current_idx + self.chunk_size, len(text))
            candidate = text[current_idx:end_idx]

            # If we reached the end of the text, just take the rest
            if end_idx == len(text):
                chunk_str = candidate.strip()
                if chunk_str:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_counter,
                            text=chunk_str,
                            page_number=page_number,
                            char_count=len(chunk_str),
                        )
                    )
                break

            # Find best split point near the end of candidate
            split_pos = -1
            for sep in separators:
                # Look for separator in the last 40% of the candidate slice
                search_start = int(len(candidate) * 0.6)
                last_pos = candidate.rfind(sep, search_start)
                if last_pos != -1:
                    split_pos = last_pos + len(sep)
                    break

            if split_pos == -1:
                # No natural separator found in window; take hard split at chunk_size
                split_pos = len(candidate)

            chunk_str = text[current_idx : current_idx + split_pos].strip()
            if chunk_str:
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_counter,
                        text=chunk_str,
                        page_number=page_number,
                        char_count=len(chunk_str),
                    )
                )
                chunk_counter += 1

            # Advance with overlap
            advance = max(1, split_pos - self.chunk_overlap)
            current_idx += advance

        return chunks

    def chunk_document(self, doc: ExtractedDocument) -> List[DocumentChunk]:
        """
        Chunks an ExtractedDocument page-by-page while tracking page numbers.
        """
        all_chunks: List[DocumentChunk] = []
        chunk_id = 0

        if doc.pages:
            for page_num, page_text in doc.pages:
                page_chunks = self.split_text(page_text, page_number=page_num, start_chunk_id=chunk_id)
                all_chunks.extend(page_chunks)
                chunk_id += len(page_chunks)
        else:
            all_chunks = self.split_text(doc.full_text, page_number=None, start_chunk_id=0)

        return all_chunks
