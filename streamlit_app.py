"""
DocuMind AI — Streamlit Reading Desk
------------------------------------
Styled to replicate the vintage editorial aesthetic of the original HTML reading desk,
with responsive layout, in-box output rendering, and multi-format export (PDF/MD/TXT).
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


def generate_pdf(content: str, mode_name: str) -> bytes:
    """Generates a clean PDF document from analysis text."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "DocuMind Analysis", new_x="LMARGIN", new_y="NEXT")

    # Mode header
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, f"Analysis Mode: {mode_name.upper().replace('_', ' ')}", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(8)

    # Body
    pdf.set_font("Helvetica", size=11)
    safe_text = (
        content
        .replace("—", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("•", "*")
    )
    safe_text = safe_text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, safe_text)
    return bytes(pdf.output())


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

    /* Base page container */
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

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1180px !important;
    }

    /* Prevent icon font collisions */
    [data-testid="stIcon"],
    [data-testid="stIcon"] *,
    .material-symbols-rounded,
    .material-symbols-outlined,
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] i,
    button[aria-label*="password"] span,
    button[aria-label*="password"] i {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
        font-style: normal !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ede7da !important;
        border-right: 1px solid var(--rule) !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: var(--serif) !important;
        font-size: 1.25rem !important;
        color: var(--ink) !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] .sidebar-section-tag {
        font-family: var(--mono);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--highlight);
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 4px;
    }
    [data-testid="stSidebar"] label p {
        font-family: var(--mono) !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        color: var(--ink-soft) !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: var(--margin) !important;
        border: 1px solid var(--rule) !important;
        border-radius: 3px !important;
        color: var(--ink) !important;
        font-family: var(--mono) !important;
        font-size: 12.5px !important;
    }
    [data-testid="stSidebar"] input:focus,
    [data-testid="stSidebar"] div[data-baseweb="select"]:focus-within {
        border-color: var(--highlight) !important;
        box-shadow: 0 0 0 2px var(--highlight-soft) !important;
    }
    .sidebar-info-card {
        background: var(--margin);
        border: 1px solid var(--rule);
        border-radius: 3px;
        padding: 10px 12px;
        margin-top: 16px;
        font-family: var(--mono);
        font-size: 11px;
        color: var(--ink-soft);
        line-height: 1.6;
    }

    /* Header matching original HTML */
    .brand-wrap {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 3px double var(--ink);
        padding-bottom: 12px;
        margin-bottom: 20px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .brand-title-row {
        display: flex;
        align-items: baseline;
        gap: 12px;
    }
    .brand-title-row h1 {
        font-family: var(--serif);
        font-size: 30px;
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
        padding: 14px 16px !important;
        font-family: var(--serif) !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        color: var(--ink) !important;
        resize: vertical !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--highlight) !important;
        box-shadow: 0 0 0 3px var(--highlight-soft) !important;
    }
    .stTextArea textarea::placeholder {
        color: #9c9282 !important;
    }

    /* Compact File Uploader */
    [data-testid="stFileUploader"] {
        margin-top: 4px !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stFileUploader"] label {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--margin) !important;
        border: 1px dashed var(--rule) !important;
        border-radius: 2px !important;
        padding: 6px 12px !important;
        min-height: 42px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-family: var(--mono) !important;
        font-size: 11px !important;
        border: 1px solid var(--rule) !important;
        background: var(--paper) !important;
        color: var(--ink) !important;
        padding: 4px 10px !important;
        border-radius: 2px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        font-family: var(--mono) !important;
        font-size: 11px !important;
        color: var(--ink-soft) !important;
    }

    /* Mode Radio Group styled exactly as .mode-btn */
    div[data-testid="stRadio"] > div {
        display: flex !important;
        gap: 8px !important;
        flex-wrap: wrap !important;
    }
    div[data-testid="stRadio"] label {
        font-family: var(--mono) !important;
        font-size: 12px !important;
        letter-spacing: 0.5px !important;
        padding: 8px 16px !important;
        border: 1px solid var(--ink) !important;
        background: transparent !important;
        color: var(--ink) !important;
        border-radius: 2px !important;
        cursor: pointer !important;
        transition: all .15s ease !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: var(--ink) !important;
        color: var(--paper) !important;
    }
    div[data-testid="stRadio"] label:hover p {
        color: var(--paper) !important;
    }
    /* Active highlighted mode */
    div[data-testid="stRadio"] label:has(input:checked) {
        background: var(--highlight) !important;
        border-color: var(--highlight) !important;
        color: #ffffff !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] input {
        display: none !important;
    }
    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        font-size: 12px !important;
        font-family: var(--mono) !important;
    }

    /* Primary Run Button */
    div.stButton > button[kind="primary"] {
        font-family: var(--mono) !important;
        font-size: 13px !important;
        letter-spacing: 1px !important;
        padding: 10px 24px !important;
        background: var(--ink) !important;
        color: var(--paper) !important;
        border: none !important;
        border-radius: 2px !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #443c33 !important;
        color: #fff !important;
    }

    /* Output Reading Box Container (guarantees text stays strictly inside) */
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--margin) !important;
        border: 1px solid var(--rule) !important;
        border-radius: 2px !important;
        min-height: 380px !important;
        padding: 18px 20px !important;
        position: relative !important;
        margin-left: 18px !important;
    }
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"]::before {
        content: "";
        position: absolute;
        left: -18px;
        top: 2px;
        bottom: 2px;
        width: 2px;
        background: repeating-linear-gradient(180deg, var(--highlight) 0 6px, transparent 6px 12px);
    }
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] p {
        font-family: var(--serif);
        font-size: 15px;
        line-height: 1.68;
        color: var(--ink);
        margin-bottom: 0.8rem;
    }
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] h1,
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] h2,
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        font-family: var(--mono);
        color: var(--highlight);
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        letter-spacing: 0.5px;
    }
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        font-size: 14px;
        text-transform: uppercase;
        border-bottom: 1px dashed var(--rule);
        padding-bottom: 4px;
    }
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] ul,
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] ol {
        padding-left: 20px;
        margin-bottom: 0.8rem;
        font-family: var(--serif);
        font-size: 15px;
        color: var(--ink);
    }
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] li {
        margin-bottom: 5px;
        line-height: 1.6;
    }
    div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] strong {
        color: var(--ink);
        font-weight: 700;
    }
    .output-empty {
        color: #a89e8c;
        font-style: italic;
        font-family: var(--serif);
        font-size: 15px;
    }

    /* Desktop Vertical Divider between panes */
    @media (min-width: 861px) {
        div[data-testid="column"]:first-child {
            border-right: 1px solid var(--rule);
            padding-right: 24px !important;
        }
        div[data-testid="column"]:last-child {
            padding-left: 4px !important;
        }
    }

    /* Mobile Responsive Adjustments */
    @media (max-width: 860px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"] {
            margin-left: 0 !important;
        }
        div[data-testid="column"]:last-child div[data-testid="stVerticalBlockBorderWrapper"]::before {
            display: none !important;
        }
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

    /* Download Buttons */
    div.stDownloadButton > button {
        font-family: var(--mono) !important;
        font-size: 11px !important;
        letter-spacing: 0.5px !important;
        padding: 6px 12px !important;
        border: 1px solid var(--ink) !important;
        background: var(--paper) !important;
        color: var(--ink) !important;
        border-radius: 2px !important;
        width: 100% !important;
    }
    div.stDownloadButton > button:hover {
        background: var(--ink) !important;
        color: var(--paper) !important;
    }

    /* Footer */
    .documind-footer {
        margin-top: 36px;
        padding-top: 14px;
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
    st.markdown("### Desk Settings")

    st.markdown('<div class="sidebar-section-tag">01 // LLM Engine</div>', unsafe_allow_html=True)
    provider_options = ["OpenAI", "Anthropic Claude", "Google Gemini", "Offline Mock Mode"]
    default_idx = 0
    if settings.ENABLE_MOCK_FALLBACK:
        default_idx = 3
    elif settings.ANTHROPIC_API_KEY and not settings.OPENAI_API_KEY:
        default_idx = 1
    elif settings.GEMINI_API_KEY and not settings.OPENAI_API_KEY:
        default_idx = 2

    chosen_provider = st.selectbox("Active Provider", options=provider_options, index=default_idx)

    if chosen_provider == "OpenAI":
        key = st.text_input("OpenAI API Key", value=settings.OPENAI_API_KEY or "", type="password", help="Stored safely in session")
        if key:
            settings.OPENAI_API_KEY = key
        model = st.selectbox("Model", options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], index=0)
        settings.OPENAI_MODEL = model
        settings.ENABLE_MOCK_FALLBACK = False

    elif chosen_provider == "Anthropic Claude":
        key = st.text_input("Claude API Key", value=settings.ANTHROPIC_API_KEY or "", type="password", help="Stored safely in session")
        if key:
            settings.ANTHROPIC_API_KEY = key
        model = st.selectbox("Model", options=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"], index=0)
        settings.ANTHROPIC_MODEL = model
        settings.ENABLE_MOCK_FALLBACK = False

    elif chosen_provider == "Google Gemini":
        key = st.text_input("Gemini API Key", value=settings.GEMINI_API_KEY or "", type="password", help="Stored safely in session")
        if key:
            settings.GEMINI_API_KEY = key
        model = st.selectbox("Model", options=["gemini-2.0-flash", "gemini-1.5-pro"], index=0)
        settings.GEMINI_MODEL = model
        settings.ENABLE_MOCK_FALLBACK = False

    else:
        st.info("Offline Mock Mode enabled — no API key needed.")
        settings.ENABLE_MOCK_FALLBACK = True

    st.markdown('<div class="sidebar-section-tag">02 // Vector RAG Parameters</div>', unsafe_allow_html=True)
    settings.CHUNK_SIZE = st.slider("Chunk Size (chars)", 400, 2500, settings.CHUNK_SIZE, 100)
    settings.CHUNK_OVERLAP = st.slider("Chunk Overlap (chars)", 50, 500, settings.CHUNK_OVERLAP, 25)
    settings.TOP_K_CHUNKS = st.slider("Top-K Retrieved Chunks", 2, 10, settings.TOP_K_CHUNKS)

    active_key_status = "Configured" if settings.is_api_key_configured or settings.ENABLE_MOCK_FALLBACK else "Key Missing"
    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div><strong>Engine:</strong> {chosen_provider}</div>
            <div><strong>Model:</strong> {settings.active_model}</div>
            <div><strong>Status:</strong> <span style="color: {'#2e7d32' if active_key_status == 'Configured' else '#c62828'}; font-weight:600;">{active_key_status}</span></div>
            <div><strong>RAG Routing:</strong> &gt; 12,000 chars</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        height=240,
        label_visibility="collapsed",
        key="doc_text_area",
    )
    if doc_text != st.session_state.doc_text_input:
        st.session_state.doc_text_input = doc_text

    # File bar
    st.markdown('<div style="font-family: var(--mono); font-size: 12px; color: var(--ink-soft); margin-top: 8px; margin-bottom: 2px;">or upload a file:</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="or upload a file:",
        type=["txt", "pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        try:
            parsed_doc = DocumentParser.parse_bytes(file_bytes, filename=uploaded_file.name)
            st.session_state.extracted_doc = parsed_doc
            st.markdown(
                f"""
                <div style="margin-top: 4px; margin-bottom: 8px;">
                    <span class="metric-badge">📄 {uploaded_file.name}</span>
                    <span class="metric-badge">({parsed_doc.page_count}p, {parsed_doc.char_count:,} chars)</span>
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

    # Mode Selector - Strongly works with native radio styled as .mode-btn
    st.markdown('<div class="pane-label" style="margin-top: 14px; margin-bottom: 6px;"><span>Select Mode</span></div>', unsafe_allow_html=True)
    
    mode_options = ["Summarize", "Key Points", "Simplify"]
    mode_map = {"Summarize": "summarize", "Key Points": "key_points", "Simplify": "simplify"}
    inv_mode_map = {"summarize": "Summarize", "key_points": "Key Points", "simplify": "Simplify"}

    default_mode_idx = mode_options.index(inv_mode_map.get(st.session_state.current_mode, "Summarize"))
    selected_mode_label = st.radio(
        label="Select Mode",
        options=mode_options,
        index=default_mode_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="mode_radio_selector",
    )
    st.session_state.current_mode = mode_map[selected_mode_label]

    # Run Row
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Analyze →", type="primary", use_container_width=False)


with col_right:
    mode_display = st.session_state.current_mode.replace("_", " ")

    st.markdown(
        f"""
        <div class="pane-label" style="padding-left: 18px;">
            <span>02 — the reading</span>
            <span style="color: var(--highlight);">{mode_display}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Everything renders STRICTLY inside this bordered paper container
    with st.container(border=True):
        output_slot = st.empty()

        if not st.session_state.analysis_result and not run_clicked:
            output_slot.markdown(
                '<div class="output-empty">Your analysis will appear here, streamed in as it\'s written…</div>',
                unsafe_allow_html=True,
            )

        if run_clicked:
            doc = st.session_state.extracted_doc
            if not doc or not doc.full_text.strip():
                output_slot.markdown(
                    '<div style="color: #a94442; font-family: var(--mono); font-size: 13px;">Paste some text or choose a file first.</div>',
                    unsafe_allow_html=True,
                )
            else:
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
                    result = output_slot.write_stream(sync_stream(get_stream()))
                    st.session_state.analysis_result = result
                except Exception as exc:
                    output_slot.markdown(
                        f'<div style="color: #a94442; font-family: var(--mono); font-size: 13px;">Error: {exc}</div>',
                        unsafe_allow_html=True,
                    )

        elif st.session_state.analysis_result:
            output_slot.markdown(st.session_state.analysis_result)

    # Download Actions (PDF, Markdown, Text)
    if st.session_state.analysis_result:
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            try:
                pdf_bytes = generate_pdf(st.session_state.analysis_result, st.session_state.current_mode)
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=f"documind_{st.session_state.current_mode}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.caption(f"PDF export error: {e}")

        with col_d2:
            st.download_button(
                label="📝 Download Markdown",
                data=st.session_state.analysis_result,
                file_name=f"documind_{st.session_state.current_mode}.md",
                mime="text/markdown",
            )

        with col_d3:
            st.download_button(
                label="📄 Download Text",
                data=st.session_state.analysis_result,
                file_name=f"documind_{st.session_state.current_mode}.txt",
                mime="text/plain",
            )


# ---------------- Footer ---------------- #
st.markdown(
    """
    <footer class="documind-footer">
      <span>DocuMind AI</span>
      <span>reading desk ready</span>
    </footer>
    """,
    unsafe_allow_html=True,
)
