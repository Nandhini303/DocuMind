"""
DocuMind AI — Streamlit Reading Desk
------------------------------------
Styled to replicate the vintage editorial aesthetic of the original HTML reading desk.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import streamlit as st
from app.core.config import settings

# Ingest Streamlit Cloud secrets into environment and settings
try:
    if hasattr(st, "secrets"):
        for key in ["OPENAI_API_KEY", "OPENAI_API", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
            if key in st.secrets:
                val = str(st.secrets[key]).strip()
                os.environ[key] = val
                if key in ("OPENAI_API_KEY", "OPENAI_API"):
                    settings.OPENAI_API_KEY = val
                elif key == "ANTHROPIC_API_KEY":
                    settings.ANTHROPIC_API_KEY = val
                elif key == "GEMINI_API_KEY":
                    settings.GEMINI_API_KEY = val
except Exception:
    pass

from app.models.schemas import AnalysisMode
from app.services.document_parser import DocumentParser, ExtractedDocument
from app.services.text_chunker import TextChunker
from app.services.embeddings import SemanticEmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.services.rag_engine import RAGEngine
from app.core.security import sanitize_text

# Page Configuration
st.set_page_config(
    page_title="DocuMind — read less, know more",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------- Exact Vintage Paper Theme Styling ----------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');

    :root {
        --paper: #f2ede3;
        --paper-line: #e3dbca;
        --ink: #2b2620;
        --ink-soft: #6b6255;
        --margin: #fffaf0;
        --highlight: #c96f34;
        --highlight-soft: #f0d9c4;
        --rule: #c9bfa8;
        --mono: "IBM Plex Mono", "Courier New", monospace;
        --serif: "Iowan Old Style", "Palatino Linotype", "Newsreader", Georgia, serif;
    }

    /* Force Light Theme & Paper Background across full app */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: var(--paper) !important;
        background-color: var(--paper) !important;
        background-image: repeating-linear-gradient(180deg, transparent 0 27px, var(--paper-line) 27px 28px) !important;
        color: var(--ink) !important;
        font-family: var(--serif) !important;
    }

    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        background: transparent !important;
    }

    /* Hide default top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #eae4d8 !important;
        border-right: 1px solid var(--rule) !important;
    }
    [data-testid="stSidebar"] * {
        font-family: var(--mono) !important;
        color: var(--ink) !important;
    }

    /* Header matching original HTML */
    .brand-wrap {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 3px double var(--ink);
        padding-bottom: 14px;
        margin-bottom: 24px;
        flex-wrap: wrap;
        gap: 10px;
    }
    .brand-title-row {
        display: flex;
        align-items: baseline;
        gap: 12px;
    }
    .brand-title-row h1 {
        font-family: var(--serif);
        font-size: 32px;
        margin: 0;
        font-weight: 600;
        letter-spacing: 0.2px;
        color: var(--ink);
    }
    .brand-mark {
        font-family: var(--mono);
        font-size: 13px;
        letter-spacing: 2px;
        color: var(--highlight);
        border: 1px solid var(--highlight);
        padding: 2px 8px;
        text-transform: uppercase;
        font-weight: 500;
    }
    .brand-subtitle {
        margin: 0;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--ink-soft);
        letter-spacing: 0.5px;
    }

    /* Pane labels */
    .pane-label {
        font-family: var(--mono);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--ink-soft);
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Textarea */
    .stTextArea textarea {
        background: var(--margin) !important;
        border: 1px solid var(--rule) !important;
        border-radius: 2px !important;
        padding: 16px 18px !important;
        font-family: var(--serif) !important;
        font-size: 15.5px !important;
        line-height: 1.65 !important;
        color: var(--ink) !important;
        min-height: 280px !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--highlight) !important;
        box-shadow: 0 0 0 3px var(--highlight-soft) !important;
    }
    .stTextArea textarea::placeholder {
        color: #9c9282 !important;
    }

    /* File uploader container */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        padding: 0 !important;
        margin-top: 4px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--margin) !important;
        border: 1px dashed var(--rule) !important;
        border-radius: 2px !important;
        padding: 10px 14px !important;
        color: var(--ink-soft) !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        font-family: var(--mono) !important;
        font-size: 12px !important;
        color: var(--ink-soft) !important;
    }

    /* Mode Buttons */
    .mode-btn-container {
        display: flex;
        gap: 8px;
        margin-top: 14px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }
    
    /* Buttons general */
    div.stButton > button {
        font-family: var(--mono) !important;
        font-size: 12px !important;
        letter-spacing: 0.5px !important;
        padding: 9px 14px !important;
        border: 1px solid var(--ink) !important;
        background: transparent !important;
        color: var(--ink) !important;
        border-radius: 2px !important;
        transition: all .15s ease !important;
    }
    div.stButton > button:hover {
        background: var(--ink) !important;
        color: var(--paper) !important;
    }

    /* Primary Run Button */
    div.stButton > button[kind="primary"] {
        font-family: var(--mono) !important;
        font-size: 13px !important;
        letter-spacing: 1px !important;
        padding: 12px 26px !important;
        background: var(--ink) !important;
        color: var(--paper) !important;
        border: none !important;
        border-radius: 2px !important;
        text-transform: uppercase !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #443c33 !important;
        color: #fff !important;
    }

    /* Output Margin & Box */
    .margin-pane {
        position: relative;
        padding-left: 18px;
    }
    .margin-pane::before {
        content: "";
        position: absolute;
        left: 0;
        top: 4px;
        bottom: 4px;
        width: 2px;
        background: repeating-linear-gradient(180deg, var(--highlight) 0 6px, transparent 6px 12px);
    }
    .output-box {
        background: var(--margin);
        border: 1px solid var(--rule);
        border-radius: 2px;
        min-height: 380px;
        padding: 20px 22px;
        font-size: 15.5px;
        line-height: 1.7;
        color: var(--ink);
        font-family: var(--serif);
    }
    .output-box p {
        margin-top: 0;
        margin-bottom: 1rem;
    }
    .output-box h1, .output-box h2, .output-box h3, .output-box h4 {
        font-family: var(--mono);
        color: var(--highlight);
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }
    .output-box h3 {
        font-size: 15px;
        text-transform: uppercase;
        border-bottom: 1px dashed var(--rule);
        padding-bottom: 4px;
    }
    .output-box ul, .output-box ol {
        padding-left: 22px;
        margin-bottom: 1rem;
    }
    .output-box li {
        margin-bottom: 6px;
    }
    .output-box strong {
        color: var(--ink);
        font-weight: 700;
    }
    .output-box code {
        font-family: var(--mono);
        background: var(--paper);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 13px;
    }
    .output-empty {
        color: #a89e8c;
        font-style: italic;
        font-family: var(--serif);
    }

    /* Badges */
    .metric-badge {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--ink-soft);
        margin-right: 8px;
    }
    .rag-badge {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--highlight);
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Footer */
    .documind-footer {
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid var(--rule);
        font-family: var(--mono);
        font-size: 11px;
        color: var(--ink-soft);
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session States
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "summarize"
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "extracted_doc" not in st.session_state:
    st.session_state.extracted_doc = None
if "doc_text_input" not in st.session_state:
    st.session_state.doc_text_input = ""
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []


# ---------------- Sidebar: Settings ---------------- #
with st.sidebar:
    st.markdown("### Settings & Providers")

    provider_options = ["OpenAI", "Anthropic Claude", "Google Gemini", "Offline Mock Mode"]
    default_idx = 0
    if settings.ENABLE_MOCK_FALLBACK:
        default_idx = 3
    elif settings.ANTHROPIC_API_KEY and not settings.OPENAI_API_KEY:
        default_idx = 1
    elif settings.GEMINI_API_KEY and not settings.OPENAI_API_KEY:
        default_idx = 2

    chosen_provider = st.selectbox("LLM Provider", options=provider_options, index=default_idx)

    if chosen_provider == "OpenAI":
        key = st.text_input("OpenAI API Key", value=settings.OPENAI_API_KEY or "", type="password")
        if key:
            settings.OPENAI_API_KEY = key
        model = st.selectbox("Model", options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], index=0)
        settings.OPENAI_MODEL = model
        settings.ENABLE_MOCK_FALLBACK = False

    elif chosen_provider == "Anthropic Claude":
        key = st.text_input("Claude API Key", value=settings.ANTHROPIC_API_KEY or "", type="password")
        if key:
            settings.ANTHROPIC_API_KEY = key
        model = st.selectbox("Model", options=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"], index=0)
        settings.ANTHROPIC_MODEL = model
        settings.ENABLE_MOCK_FALLBACK = False

    elif chosen_provider == "Google Gemini":
        key = st.text_input("Gemini API Key", value=settings.GEMINI_API_KEY or "", type="password")
        if key:
            settings.GEMINI_API_KEY = key
        model = st.selectbox("Model", options=["gemini-2.0-flash", "gemini-1.5-pro"], index=0)
        settings.GEMINI_MODEL = model
        settings.ENABLE_MOCK_FALLBACK = False

    else:
        st.info("Mock mode enabled — no API key required.")
        settings.ENABLE_MOCK_FALLBACK = True

    st.markdown("---")
    st.markdown("### RAG Settings")
    settings.CHUNK_SIZE = st.slider("Chunk Size", 400, 2500, settings.CHUNK_SIZE, 100)
    settings.CHUNK_OVERLAP = st.slider("Chunk Overlap", 50, 500, settings.CHUNK_OVERLAP, 25)
    settings.TOP_K_CHUNKS = st.slider("Top-K Chunks", 2, 10, settings.TOP_K_CHUNKS)


# ---------------- Header ---------------- #
st.markdown(
    """
    <div class="brand-wrap">
      <div class="brand-title-row">
        <h1>DocuMind</h1>
        <span class="brand-mark">AI reading desk</span>
      </div>
      <p class="brand-subtitle">paste text or upload a document — get it summarized, distilled, or simplified, live</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- Main Desk Layout ---------------- #
col_left, col_right = st.columns([1.1, 1.0], gap="large")

with col_left:
    # Calculate live charcount
    current_text = st.session_state.doc_text_input
    char_len = len(current_text) if current_text else 0

    st.markdown(
        f"""
        <div class="pane-label">
            <span>01 — the document</span>
            <span id="charcount">{char_len} chars</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    doc_text = st.text_area(
        label="01 — the document",
        placeholder="Paste an article, report, contract, or essay here…",
        height=320,
        label_visibility="collapsed",
        key="doc_text_area",
    )
    if doc_text != st.session_state.doc_text_input:
        st.session_state.doc_text_input = doc_text

    # File uploader
    uploaded_file = st.file_uploader(
        "or upload a file:",
        type=["txt", "pdf"],
        label_visibility="visible",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        try:
            parsed_doc = DocumentParser.parse_bytes(file_bytes, filename=uploaded_file.name)
            st.session_state.extracted_doc = parsed_doc
            st.markdown(
                f"""
                <div style="margin-top: 6px;">
                    <span class="metric-badge">📄 {uploaded_file.name}</span>
                    <span class="metric-badge">({parsed_doc.page_count} pages, {parsed_doc.char_count:,} chars)</span>
                    <span class="rag-badge">{'⚡ Vector RAG' if parsed_doc.char_count > settings.DIRECT_PROCESSING_CHAR_LIMIT else 'Direct'}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif doc_text.strip():
        cleaned = sanitize_text(doc_text)
        st.session_state.extracted_doc = ExtractedDocument(full_text=cleaned, pages=[(1, cleaned)])
    else:
        st.session_state.extracted_doc = None

    # Mode Buttons matching .mode-btn
    st.markdown('<div style="font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: var(--ink-soft); margin-top: 14px; margin-bottom: 6px;">Select Mode</div>', unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        sum_active = "background: var(--highlight) !important; color: #fff !important; border-color: var(--highlight) !important;" if st.session_state.current_mode == "summarize" else ""
        st.markdown(f"<style>div.row-widget.stButton:has(button#btn-sum) button {{ {sum_active} }}</style>", unsafe_allow_html=True)
        if st.button("Summarize", key="btn-sum", use_container_width=True):
            st.session_state.current_mode = "summarize"
            st.rerun()

    with m_col2:
        kp_active = "background: var(--highlight) !important; color: #fff !important; border-color: var(--highlight) !important;" if st.session_state.current_mode == "key_points" else ""
        st.markdown(f"<style>div.row-widget.stButton:has(button#btn-kp) button {{ {kp_active} }}</style>", unsafe_allow_html=True)
        if st.button("Key Points", key="btn-kp", use_container_width=True):
            st.session_state.current_mode = "key_points"
            st.rerun()

    with m_col3:
        simp_active = "background: var(--highlight) !important; color: #fff !important; border-color: var(--highlight) !important;" if st.session_state.current_mode == "simplify" else ""
        st.markdown(f"<style>div.row-widget.stButton:has(button#btn-simp) button {{ {simp_active} }}</style>", unsafe_allow_html=True)
        if st.button("Simplify", key="btn-simp", use_container_width=True):
            st.session_state.current_mode = "simplify"
            st.rerun()

    # Run Row
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Analyze →", type="primary", use_container_width=False)


with col_right:
    mode_display = st.session_state.current_mode.replace("_", " ")
    st.markdown(
        f"""
        <div class="margin-pane">
            <div class="pane-label">
                <span>02 — the reading</span>
                <span style="color: var(--highlight);">{mode_display}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    output_slot = st.empty()

    if not st.session_state.analysis_result and not run_clicked:
        output_slot.markdown(
            """
            <div class="margin-pane">
                <div class="output-box">
                    <span class="output-empty">Your analysis will appear here, streamed in as it's written…</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_clicked:
        doc = st.session_state.extracted_doc
        if not doc or not doc.full_text.strip():
            output_slot.markdown(
                """
                <div class="margin-pane">
                    <div class="output-box" style="border-color: #d9534f; color: #a94442;">
                        Paste some text or choose a file first.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Prepare services
            llm_service = LLMService()
            chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
            vector_store = VectorStore(embedding_service=SemanticEmbeddingService())
            rag_engine = RAGEngine(llm_service=llm_service, chunker=chunker, vector_store=vector_store)

            mode = AnalysisMode(st.session_state.current_mode)

            if rag_engine.is_large_document(doc.full_text):
                chunks = chunker.chunk_document(doc)
                vector_store.clear()
                vector_store.add_chunks(chunks)
                st.session_state.retrieved_chunks = rag_engine.retrieve_context(mode=mode, top_k_per_query=settings.TOP_K_CHUNKS)

            async def get_stream():
                async for token in rag_engine.process_and_stream(doc=doc, mode=mode):
                    yield token

            def sync_stream(async_generator):
                loop = asyncio.new_event_loop()
                try:
                    while True:
                        try:
                            token = loop.run_until_complete(async_generator.__anext__())
                            yield token
                        except StopAsyncIteration:
                            break
                finally:
                    loop.close()

            try:
                with output_slot.container():
                    st.markdown('<div class="margin-pane"><div class="output-box">', unsafe_allow_html=True)
                    result = st.write_stream(sync_stream(get_stream()))
                    st.markdown('</div></div>', unsafe_allow_html=True)
                    st.session_state.analysis_result = result
            except Exception as exc:
                output_slot.markdown(
                    f"""
                    <div class="margin-pane">
                        <div class="output-box" style="border-color: #d9534f; color: #a94442;">
                            Error: {exc}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    elif st.session_state.analysis_result:
        with output_slot.container():
            st.markdown(
                f"""
                <div class="margin-pane">
                    <div class="output-box">
                        {st.session_state.analysis_result}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Actions when result exists
    if st.session_state.analysis_result:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Download Markdown (.md)",
                data=st.session_state.analysis_result,
                file_name=f"documind_{st.session_state.current_mode}.md",
                mime="text/markdown",
            )
        with col_d2:
            st.download_button(
                "Download Text (.txt)",
                data=st.session_state.analysis_result,
                file_name=f"documind_{st.session_state.current_mode}.txt",
                mime="text/plain",
            )


# ---------------- Footer ---------------- #
st.markdown(
    """
    <footer class="documind-footer">
      <span>DocuMind AI — built with Vibe Coding</span>
      <span>reading desk ready</span>
    </footer>
    """,
    unsafe_allow_html=True,
)
