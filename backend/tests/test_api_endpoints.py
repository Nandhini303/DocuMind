"""
Integration tests for FastAPI endpoints:
- GET /
- GET /api/health
- POST /api/analyze (SSE streaming)
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from app.core.config import settings
from app.api.routes.health import health_check
from app.api.routes.analyze import analyze_document
from app.services.rag_engine import RAGEngine
from app.api.dependencies import get_rag_engine


def run_coroutine(coro):
    """Executes a coroutine without requiring a socket-based event loop."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


async def collect_async_generator(async_gen):
    """Collects all items yielded from an asynchronous generator."""
    items = []
    async for item in async_gen:
        items.append(item)
    return items


class MockUploadFile:
    """Mock UploadFile for testing multipart uploads."""
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


class MockRequest:
    """Mock Request for SSE streaming disconnect checks."""
    def __init__(self, is_disconnected: bool = False):
        self._disconnected = is_disconnected

    async def is_disconnected(self) -> bool:
        return self._disconnected


class TestAPIEndpoints(unittest.TestCase):

    def test_root_index_exists(self):
        static_dir = Path(__file__).resolve().parent.parent / "static"
        index_file = static_dir / "index.html"
        self.assertTrue(index_file.exists(), "index.html must exist in static directory")
        content = index_file.read_text(encoding="utf-8")
        self.assertIn("<title>DocuMind", content)
        self.assertIn("AI reading desk", content)

    def test_health_check_endpoint(self):
        res = run_coroutine(health_check())
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.model, settings.active_model)
        self.assertIsInstance(res.api_key_configured, bool)
        self.assertEqual(res.version, settings.APP_VERSION)
        self.assertTrue(res.rag_enabled)

    def test_analyze_invalid_mode(self):
        rag = get_rag_engine()
        req = MockRequest()
        with self.assertRaises(HTTPException) as ctx:
            run_coroutine(analyze_document(request=req, mode="invalid_mode", text="Hello", rag_engine=rag))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Unknown mode", ctx.exception.detail)

    def test_analyze_empty_input(self):
        rag = get_rag_engine()
        req = MockRequest()
        with self.assertRaises(HTTPException) as ctx:
            run_coroutine(analyze_document(request=req, mode="summarize", text="   ", rag_engine=rag))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("No document text provided", ctx.exception.detail)

    def test_analyze_unsupported_file_extension(self):
        rag = get_rag_engine()
        req = MockRequest()
        fake_file = MockUploadFile(filename="malicious.exe", content=b"fake exe data")
        with self.assertRaises(HTTPException) as ctx:
            run_coroutine(analyze_document(request=req, mode="summarize", file=fake_file, rag_engine=rag))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Unsupported file type", ctx.exception.detail)

    def test_analyze_streaming_mock_mode(self):
        original_mock = settings.ENABLE_MOCK_FALLBACK
        settings.ENABLE_MOCK_FALLBACK = True
        try:
            rag = get_rag_engine()
            req = MockRequest()
            response: StreamingResponse = run_coroutine(
                analyze_document(request=req, mode="summarize", text="Sample document text.", rag_engine=rag)
            )
            self.assertEqual(response.media_type, "text/event-stream")
            self.assertEqual(response.headers["Cache-Control"], "no-cache")
            self.assertEqual(response.headers["X-Accel-Buffering"], "no")
        finally:
            settings.ENABLE_MOCK_FALLBACK = original_mock

    def test_analyze_file_upload_mock_mode(self):
        original_mock = settings.ENABLE_MOCK_FALLBACK
        settings.ENABLE_MOCK_FALLBACK = True
        try:
            rag = get_rag_engine()
            req = MockRequest()
            fake_txt = MockUploadFile(filename="sample.txt", content=b"Valid plain text for testing.")
            response: StreamingResponse = run_coroutine(
                analyze_document(request=req, mode="key_points", file=fake_txt, rag_engine=rag)
            )
            self.assertEqual(response.media_type, "text/event-stream")
        finally:
            settings.ENABLE_MOCK_FALLBACK = original_mock


if __name__ == "__main__":
    unittest.main()
