"""
Document analysis route with Server-Sent Events (SSE) streaming.
Preserves exact frontend contract and handles file uploads and text inputs.
"""

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.security import sanitize_text, validate_file_upload
from app.core.exceptions import (
    DocumentParsingError,
    FileSizeLimitExceededError,
    InvalidFileError,
    MissingAPIKeyError,
)
from app.models.schemas import AnalysisMode
from app.services.document_parser import DocumentParser, ExtractedDocument
from app.services.rag_engine import RAGEngine
from app.api.dependencies import get_rag_engine
from app.utils.sse import sse_event_stream

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post("/analyze")
async def analyze_document(
    request: Request,
    mode: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Analyzes document text or uploaded PDF/TXT files using Claude and RAG.
    Streams progress and text deltas back via Server-Sent Events.
    """
    # 1. Validate API Key configuration early
    if not settings.is_api_key_configured and not settings.ENABLE_MOCK_FALLBACK:
        raise HTTPException(
            status_code=500,
            detail="Server is missing ANTHROPIC_API_KEY. Set it as an environment variable.",
        )

    # 2. Validate Analysis Mode
    try:
        analysis_mode = AnalysisMode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'.")

    # 3. Process Input (File upload or direct text)
    extracted_doc: Optional[ExtractedDocument] = None

    if file is not None and hasattr(file, "filename") and file.filename:
        # Read file contents with size guard
        raw_bytes = await file.read()
        file_size = len(raw_bytes)

        try:
            validate_file_upload(
                filename=file.filename,
                file_size=file_size,
                content_preview=raw_bytes[:8],
            )
            extracted_doc = DocumentParser.parse_bytes(raw_bytes, filename=file.filename)
        except (InvalidFileError, DocumentParsingError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileSizeLimitExceededError as exc:
            raise HTTPException(status_code=413, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to process file: {exc}")

    elif text and isinstance(text, str):
        clean = sanitize_text(text)
        if clean:
            extracted_doc = ExtractedDocument(full_text=clean, pages=[(1, clean)])

    # 4. Check for presence of document content
    if not extracted_doc or not extracted_doc.full_text.strip():
        raise HTTPException(status_code=400, detail="No document text provided.")

    # 5. Guardrail against runaway document size (e.g. > 500k characters)
    if extracted_doc.char_count > settings.MAX_DOCUMENT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Document exceeds maximum length of {settings.MAX_DOCUMENT_CHARS} characters.",
        )

    # 6. Stream analysis via RAG engine and SSE formatter
    token_generator = rag_engine.process_and_stream(doc=extracted_doc, mode=analysis_mode)
    event_stream = sse_event_stream(generator=token_generator, request=request)

    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables proxy buffering on Nginx/Render
        },
    )
