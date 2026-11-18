# DocuMind  AI Document Intelligence Platform

An AI-powered document analysis platform that allows users to upload
PDF/TXT files or paste text and perform three operations: **Summarize,
Extract Key Points, and Simplify**. Responses are streamed in real time
using the OpenAI API.

## Tech Stack

-   **Frontend:** HTML5, CSS3, Vanilla JavaScript
-   **Backend:** Python, FastAPI
-   **AI Engine:** OpenAI API
-   **Vector Database:** ChromaDB
-   **Embeddings:** Semantic text embeddings
-   **RAG:** Document chunking, vector retrieval, contextual generation
-   **PDF Processing:** Python-based PDF text extraction
-   **Streaming:** Server-Sent Events (SSE)
-   **Deployment:** Docker, Render

## Features

-   PDF and TXT document uploads
-   Direct text input
-   AI-powered summarization
-   Key point extraction
-   Text simplification
-   Semantic document search
-   RAG-based contextual processing
-   Real-time response streaming
-   Secure API key management
-   Automated backend tests

## Installation

``` bash
git clone <repository-url>
cd DocuMind/backend

python -m venv venv

# Windows
venv\\Scripts\\activate

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

## Run Locally

#### Option A: FastAPI Web App (HTML5 Frontend)
```bash
# Start with Uvicorn (hot-reload enabled)
uvicorn main:app --reload --port 8080
```
Open **http://localhost:8080** in your browser.  
API documentation: http://localhost:8080/docs

#### Option B: Streamlit Interactive Desk
```bash
# Launch the Streamlit dashboard
streamlit run streamlit_app.py
```
Open **http://localhost:8501** in your browser to use the interactive reading desk with live streaming, model provider selection, and vector chunk inspector.

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

## Docker Deployment

``` bash
docker build -t documind .
docker run -p 8080:8080 --env-file .env documind
```

## Deployment

Deploy using Docker on Render or another supported cloud platform.
Configure `OPENAI_API_KEY` as a secure environment variable.

## Security

-   API keys stored in environment variables.
-   Input and file validation.
-   Upload size restrictions.
-   Secure HTTP response headers.
-   Backend-controlled OpenAI API requests.

## Author

Developed by **Nandhini**  [GitHub](https://github.com/Nandhini303).
