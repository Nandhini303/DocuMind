# DocuMind AI — Setup, Architecture & Deployment Guide
*Developed by Nandhini*

An AI-powered document intelligence platform. Paste raw text or upload a `.txt` / `.pdf` document, select **Summarize**, **Key Points**, or **Simplify**, and watch the AI's response stream in live.

- **Frontend:** Pure HTML5 / CSS3 / Vanilla JS (no framework, no build step, 100% untouched UI contract) — served as static assets by FastAPI at `/` and `/static`.
- **Backend:** Python + FastAPI engineered with Clean Architecture and modular services.
- **LLM Engine:** Anthropic Claude (`claude-3-5-sonnet-20241022` primary) with OpenAI (`gpt-4o`) and Google Gemini fallback.
- **Vector Database & Search:** Embedded vector store with ChromaDB architecture, 256-dimensional normalized dense semantic embeddings, and cosine similarity ranking.
- **Document Chunking & RAG:** Recursive semantic text chunker with overlap and multi-query context retrieval for long documents.
- **PDF Processing:** Advanced extraction with unicode ligature resolution, line-break hyphen reconstruction, password-free decryption, and page metadata preservation.
- **Streaming:** Asynchronous Server-Sent Events (SSE) with non-blocking client disconnection detection and proxy buffer prevention (`X-Accel-Buffering: no`).
- **Security:** Pure ASGI security headers, strict file validation, magic byte checking, and upload size guardrails.
- **Deployment:** Multi-stage, non-root production Dockerfile with automated health checks. Optimized for Render.com ($0 free tier) and AWS App Runner.

---

## Cost Summary (Designed to Run at $0)

| Piece | Provider | Tier / Cost |
|---|---|---|
| **LLM Inference** | Anthropic Console | Free trial credits (new accounts receive free credits on signup) |
| **Hosting** | Render.com | Free Web Service tier (no credit card required) |
| **Domain & SSL** | Render.com | Automated public `https://*.onrender.com` with TLS certificate |
| **Vector Indexing** | Local / Embedded | Zero cloud database cost ($0) |

> **Note on Assignment Brief:** The original brief suggests AWS deployment. Render was configured as the default zero-cost cloud host to ensure students can test, demonstrate, and deploy without incurring AWS charges. A production-ready Dockerfile is included for seamless deployment to AWS App Runner, ECS, or Render.

---

## 1. Quickstart & Local Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11 & 3.12)
- Anthropic Claude API Key (`ANTHROPIC_API_KEY`) or OpenAI API Key (`OPENAI_API_KEY` / `OPENAI_API`)

### Installation

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

Open `.env` and set your API key:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### Running the Server

```bash
# Start with Uvicorn (hot-reload enabled)
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# Or run the entrypoint script directly
python main.py
```

Open **http://localhost:8080** in your browser. The footer will display `backend connected`.

---

## 2. System Architecture & RAG Pipeline

DocuMind uses a clean, decoupled service architecture:

```
DocuMind/
├── .env.example
├── README.md
└── backend/
    ├── app/
    │   ├── main.py                  # FastAPI factory, ASGI middleware, lifespan, router mounts
    │   ├── core/
    │   │   ├── config.py            # Pydantic Settings, multi-dir .env auto-discovery
    │   │   ├── security.py          # Pure ASGI security headers, text sanitization, upload guard
    │   │   └── exceptions.py        # Domain exceptions (DocumentParsingError, etc.)
    │   ├── models/
    │   │   └── schemas.py           # Pydantic schemas: AnalysisMode, HealthResponse, DocumentChunk
    │   ├── services/
    │   │   ├── document_parser.py   # PDF & TXT parser: ligature resolution, line-hyphen joining
    │   │   ├── text_chunker.py      # Recursive semantic chunker with boundary overlap
    │   │   ├── embeddings.py        # 256-D normalized dense semantic vector embedding engine
    │   │   ├── vector_store.py      # Vector index (ChromaDB architecture) with cosine similarity
    │   │   ├── rag_engine.py        # Adaptive vector RAG pipeline with multi-query retrieval
    │   │   └── llm_service.py       # Async SSE client: Claude (primary), OpenAI, Gemini, & mock
    │   ├── api/
    │   │   ├── dependencies.py      # Dependency injection providers
    │   │   └── routes/
    │   │       ├── health.py        # GET /api/health endpoint
    │   │       └── analyze.py       # POST /api/analyze SSE streaming endpoint
    │   └── utils/
    │       └── sse.py               # SSE packet formatting & connection abort handling
    ├── static/
    │   └── index.html               # 100% UNCHANGED Frontend UI
    ├── tests/                       # Complete automated test suite (31 tests)
    │   ├── test_document_parser.py
    │   ├── test_text_chunker.py
    │   ├── test_vector_store.py
    │   ├── test_rag_engine.py
    │   ├── test_security.py
    │   └── test_api_endpoints.py
    ├── Dockerfile                   # Multi-stage production container
    ├── requirements.txt
    ├── render.yaml                  # Render deployment blueprint
    ├── run_tests.py                 # Master test runner
    └── main.py                      # Root entrypoint
```

### Retrieval-Augmented Generation (RAG) Flow

1. **Document Intake & Normalization:** `DocumentParser` cleans unicode ligatures (`ﬁ`, `ﬂ`, `ﬀ`), resolves hyphenated line breaks (`infor-\nmation` $\rightarrow$ `information`), strips non-printable control characters, and extracts text page-by-page.
2. **Adaptive Context Routing:**
   - **Compact Documents ($\le$ 12,000 characters):** Streamed directly to Claude for holistic context.
   - **Large Documents ($>$ 12,000 characters):** Automatically routed to the Vector RAG Pipeline.
3. **Semantic Chunking:** `TextChunker` segments documents into 1,000-character chunks with 150-character overlap, respecting paragraphs and sentence boundaries.
4. **Vector Database & Embeddings:** `SemanticEmbeddingService` generates 256-dimensional L2-normalized dense feature vectors. Chunks are indexed into `VectorStore` (ChromaDB vector database architecture).
5. **Multi-Query Semantic Search:** `RAGEngine` generates mode-specific retrieval queries (e.g. for `summarize`, `key_points`, or `simplify`), computes cosine similarity:
   $$\text{Similarity}(A, B) = A \cdot B$$
   and retrieves top-$K$ deduplicated chunks ordered chronologically by page number.
6. **Optimized SSE Streaming:** `LLMService` streams tokens asynchronously using Server-Sent Events with client disconnect detection (`request.is_disconnected()`) to conserve API tokens if the client closes the connection.

---

## 3. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *None* | Anthropic Claude API Key (**Primary provider**) |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Claude model name |
| `OPENAI_API_KEY` / `OPENAI_API` | *None* | OpenAI API Key (Supported provider fallback) |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `GEMINI_API_KEY` | *None* | Google Gemini API Key (Optional fallback) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `PORT` | `8080` | Port for the HTTP server |
| `HOST` | `0.0.0.0` | Host binding interface |
| `ENVIRONMENT` | `production` | Deployment environment (`development` / `production`) |
| `MAX_UPLOAD_SIZE_BYTES` | `15728640` (15 MB) | Maximum permitted file upload size |
| `MAX_DOCUMENT_CHARS` | `500000` | Maximum character boundary guardrail (~100k words) |
| `DIRECT_PROCESSING_CHAR_LIMIT`| `12000` | Threshold above which Vector RAG is triggered |
| `CHUNK_SIZE` | `1000` | Character size per semantic chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between adjacent chunks |
| `TOP_K_CHUNKS` | `5` | Number of chunks retrieved per query |
| `ENABLE_MOCK_FALLBACK` | `false` | Enable simulated streaming for offline tests without API keys |

---

## 4. API Specification

### `GET /`
Serves the responsive static frontend (`static/index.html`).

### `GET /api/health`
Returns system status, active LLM model, and API configuration state.
```json
{
  "status": "ok",
  "model": "claude-3-5-sonnet-20241022",
  "api_key_configured": true,
  "provider": "anthropic",
  "version": "2.0.0",
  "rag_enabled": true
}
```

### `GET /favicon.ico`
Returns `204 No Content` or static icon to prevent browser console 404 noise.

### `POST /api/analyze`
Accepts `multipart/form-data`:
- `mode` *(string, required)*: `summarize` | `key_points` | `simplify`
- `text` *(string, optional)*: Raw text content
- `file` *(binary, optional)*: Uploaded `.pdf` or `.txt` file

Returns an asynchronous `text/event-stream` (Server-Sent Events):
- `data: {"delta": "..."}\n\n` — Real-time streamed text token
- `data: {"error": "..."}\n\n` — Error notification
- `data: {"done": true}\n\n` — Stream completion signal

---

## 5. Automated Testing

The backend includes a comprehensive test suite covering all modules:
- Document parser & ligature cleaning
- Text chunker & boundary overlap
- Semantic embeddings & vector store retrieval
- RAG pipeline routing & context assembly
- Security sanitization & upload guardrails
- API route health & SSE streaming endpoints

To execute the test suite:
```bash
python run_tests.py
```

Expected output:
```
======================================================================
                     DocuMind AI Backend Test Suite                   
======================================================================
Results: 31/31 passed | 0 failures | 0 errors
======================================================================
```

---

## 6. Docker Deployment

### Build the Image
```bash
docker build -t documind-ai:latest -f Dockerfile .
```

### Run the Container Locally
```bash
docker run -p 8080:8080 --env-file .env documind-ai:latest
```

The container runs as an unprivileged user (`appuser` UID 1000) and includes an integrated `HEALTHCHECK` probing `/api/health`.

---

## 7. Cloud Deployment (Render.com)

1. Commit and push your code to a GitHub repository:
   ```bash
   git add .
   git commit -m "DocuMind upgraded backend with FastAPI, RAG, and Claude AI"
   git push origin main
   ```
2. Log in to [render.com](https://render.com) and click **New + → Web Service**.
3. Connect your GitHub repository.
4. Configure service settings:
   - **Root Directory:** `backend`
   - **Runtime:** `Docker`
   - **Instance Type:** `Free`
5. Under **Environment Variables**, add:
   - `ANTHROPIC_API_KEY`: your Anthropic API key (or `OPENAI_API`)
   - `PORT`: `8080`
6. Click **Create Web Service**. Render builds the multi-stage Docker container and provides an HTTPS URL (e.g. `https://documind-ai.onrender.com`).

*Tip:* You can also use **New + → Blueprint** and select `backend/render.yaml` for automatic configuration.

---

## 8. Security Checklist

- [x] `.env` is listed in `.gitignore` and never committed to source control.
- [x] No API keys appear in frontend files or repository source code.
- [x] `/api/health` indicates whether a key is configured without exposing the secret key value.
- [x] Upload size limit (15 MB) and file extension verification prevent memory exhaustion.
- [x] Pure ASGI security headers (CSP, HSTS, X-Content-Type-Options, X-Frame-Options) applied to all HTTP responses.

---

## 9. Author

Developed by **Nandhini** ([@Nandhini303](https://github.com/Nandhini303)).
