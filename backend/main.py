"""
DocuMind AI — backend server
-----------------------------
FastAPI app that:
  1. Serves the static frontend (index.html)
  2. Accepts pasted text OR an uploaded .txt/.pdf file
  3. Streams a live AI analysis (summary / key points / simplify) back to
     the browser using Server-Sent Events (SSE)

The Gemini API key is read from an environment variable only —
it never appears in frontend code or in this repo.
"""

import os
import io
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
MAX_INPUT_CHARS = 20000  # simple guardrail so requests stay cheap & fast

if not API_KEY:
    # We don't crash on import so the container can still start and show a
    # clear error in the UI/logs — but every /api/analyze call will fail
    # loudly until the env var is set.
    print("WARNING: GEMINI_API_KEY is not set. Set it as an environment "
          "variable before making requests.")

client = genai.Client(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="DocuMind AI")

# CORS is permissive here because the frontend is served from the same
# origin in production; this just protects local dev with a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODES = {
    "summarize": (
        "You are an expert editor. Read the document below and produce a "
        "concise, well-structured summary. Use short paragraphs or bullet "
        "points. Capture the main argument, key facts, and conclusion. "
        "Do not add outside information."
    ),
    "key_points": (
        "You are an analyst. Read the document below and extract the most "
        "important points as a clean bulleted list, ordered by importance. "
        "Each bullet should be one clear sentence. Include a one-line "
        "'Bottom line' at the end."
    ),
    "simplify": (
        "You are a patient teacher. Rewrite the document below in plain, "
        "simple language a general reader (no specialist background) could "
        "understand. Keep it faithful to the original meaning, avoid "
        "jargon, and keep roughly the same length as the original."
    ),
}


def extract_text_from_pdf(raw_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL, "api_key_configured": bool(API_KEY)}


@app.post("/api/analyze")
async def analyze(
    mode: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Server is missing GEMINI_API_KEY. Set it as an environment variable.",
        )

    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'.")

    document_text = ""
    if file is not None:
        raw = await file.read()
        if file.filename.lower().endswith(".pdf"):
            document_text = extract_text_from_pdf(raw)
        else:
            document_text = raw.decode("utf-8", errors="ignore")
    elif text:
        document_text = text

    document_text = (document_text or "").strip()
    if not document_text:
        raise HTTPException(status_code=400, detail="No document text provided.")

    document_text = document_text[:MAX_INPUT_CHARS]
    system_prompt = MODES[mode]

    def event_stream():
        try:
            response = client.models.generate_content_stream(
                model=MODEL,
                contents=document_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1500,
                ),
            )
            for chunk in response:
                if chunk.text:
                    payload = json.dumps({"delta": chunk.text})
                    yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            payload = json.dumps({"error": str(exc)})
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
