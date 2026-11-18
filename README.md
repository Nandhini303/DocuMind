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
python -m pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file:

``` env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=your_supported_model
PORT=8080
```

Never expose API keys in frontend code or commit `.env` to GitHub.

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

## Architecture

1.  **Document Intake:** Upload PDF/TXT or paste text.
2.  **Text Extraction:** Extract and normalize document content.
3.  **Chunking:** Split large documents into manageable chunks.
4.  **Embeddings:** Generate vector representations.
5.  **Vector Storage:** Store document vectors in ChromaDB.
6.  **Retrieval:** Retrieve relevant document chunks using semantic
    similarity.
7.  **AI Processing:** Send relevant context and user instructions to
    the OpenAI API.
8.  **Streaming:** Return generated responses through SSE.

## API Endpoints

  Method   Endpoint         Description
  -------- ---------------- ----------------------------------
  GET      `/`              Frontend application
  GET      `/api/health`    Backend health status
  POST     `/api/analyze`   Document analysis with streaming

## Testing

``` bash
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
