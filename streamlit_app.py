"""
DocuMind AI — Streamlit Reading Desk
------------------------------------
Interactive document reading, analysis, and Q&A desk powered by
FastAPI / RAG backend components and multi-provider LLMs.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, List

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import streamlit as st
from app.core.config import settings
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
    page_title="DocuMind AI — Intelligent Reading Desk",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .main-header {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.4rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: #2b2620;
    }
    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #6b6255;
        letter-spacing: 0.5px;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        background-color: #f6f1e8;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #e3dbca;
        color: #6b6255;
        display: inline-block;
        margin-right: 6px;
    }
    .rag-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        background-color: #fdf5ec;
        color: #c96f34;
        border: 1px solid #f0d9c4;
        padding: 2px 8px;
        border-radius: 3px;
        font-weight: 500;
    }
    .stTextArea textarea {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session():
    """Initializes session state variables."""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = ""
    if "extracted_doc" not in st.session_state:
        st.session_state.extracted_doc = None
    if "retrieved_chunks" not in st.session_state:
        st.session_state.retrieved_chunks = []
    if "active_mode_label" not in st.session_state:
        st.session_state.active_mode_label = "Summarize"


initialize_session()


# ---------------- Sidebar: Settings & Model Configuration ---------------- #
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    # Provider Selection
    provider_options = ["Anthropic Claude", "OpenAI", "Google Gemini", "Offline Mock Mode"]
    default_provider_idx = 0
    if settings.ENABLE_MOCK_FALLBACK:
        default_provider_idx = 3
    elif settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY:
        default_provider_idx = 1
    elif settings.GEMINI_API_KEY and not settings.ANTHROPIC_API_KEY:
        default_provider_idx = 2

    chosen_provider = st.selectbox(
        "LLM Provider",
        options=provider_options,
        index=default_provider_idx,
        help="Select which LLM provider to stream responses from.",
    )

    # API Keys & Models based on provider
    if chosen_provider == "Anthropic Claude":
        api_key = st.text_input(
            "Anthropic API Key",
            value=settings.ANTHROPIC_API_KEY or "",
            type="password",
            help="Defaults to ANTHROPIC_API_KEY from .env",
        )
        if api_key:
            settings.ANTHROPIC_API_KEY = api_key

        model_name = st.selectbox(
            "Claude Model",
            options=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
            index=0,
        )
        settings.ANTHROPIC_MODEL = model_name
        settings.ENABLE_MOCK_FALLBACK = False

    elif chosen_provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            value=settings.OPENAI_API_KEY or "",
            type="password",
            help="Defaults to OPENAI_API_KEY from .env",
        )
        if api_key:
            settings.OPENAI_API_KEY = api_key

        model_name = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=0,
        )
        settings.OPENAI_MODEL = model_name
        settings.ENABLE_MOCK_FALLBACK = False

    elif chosen_provider == "Google Gemini":
        api_key = st.text_input(
            "Gemini API Key",
            value=settings.GEMINI_API_KEY or "",
            type="password",
            help="Defaults to GEMINI_API_KEY from .env",
        )
        if api_key:
            settings.GEMINI_API_KEY = api_key

        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=0,
        )
        settings.GEMINI_MODEL = model_name
        settings.ENABLE_MOCK_FALLBACK = False

    else:
        st.info("ℹ️ Running in Mock Mode. No API calls or keys required.")
        settings.ENABLE_MOCK_FALLBACK = True

    st.divider()

    st.markdown("### 🧠 RAG & Guardrails")
    chunk_size = st.slider("Chunk Size (characters)", min_value=400, max_value=2500, value=settings.CHUNK_SIZE, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=50, max_value=500, value=settings.CHUNK_OVERLAP, step=25)
    top_k = st.slider("Top-K Semantic Chunks", min_value=2, max_value=12, value=settings.TOP_K_CHUNKS)

    settings.CHUNK_SIZE = chunk_size
    settings.CHUNK_OVERLAP = chunk_overlap
    settings.TOP_K_CHUNKS = top_k

    st.divider()
    st.markdown(
        """
        **DocuMind AI** v2.0  
        *Retrieval-Augmented Intelligent Reader*
        """
    )


# ---------------- Header ---------------- #
st.markdown('<div class="main-header">DocuMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">READ LESS, KNOW MORE — LIVE SUMMARIZATION, ANALYSIS & SYNTHESIS DESK</div>',
    unsafe_allow_html=True,
)


# ---------------- Main Desk: Input & Output Columns ---------------- #
col_input, col_output = st.columns([1.05, 1.15], gap="large")

with col_input:
    st.markdown("#### 01 — The Document")

    input_method = st.radio(
        "Choose Input Method:",
        options=["Upload File (.pdf, .txt)", "Paste Text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    extracted_doc: Optional[ExtractedDocument] = None

    if input_method == "Upload File (.pdf, .txt)":
        uploaded_file = st.file_uploader(
            "Drop a document here:",
            type=["pdf", "txt"],
            help="Upload a PDF or TXT file up to 15MB.",
        )
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            with st.spinner(f"Extracting and normalizing {uploaded_file.name}..."):
                try:
                    extracted_doc = DocumentParser.parse_bytes(bytes_data, filename=uploaded_file.name)
                    st.session_state.extracted_doc = extracted_doc
                except Exception as exc:
                    st.error(f"Error parsing document: {exc}")
    else:
        text_input = st.text_area(
            "Paste document text:",
            placeholder="Paste an article, legal agreement, research paper, or essay here...",
            height=320,
        )
        if text_input.strip():
            clean_text = sanitize_text(text_input)
            extracted_doc = ExtractedDocument(full_text=clean_text, pages=[(1, clean_text)])
            st.session_state.extracted_doc = extracted_doc
        else:
            extracted_doc = None
            st.session_state.extracted_doc = None

    # Document Stats Badge
    if extracted_doc and extracted_doc.full_text.strip():
        char_count = extracted_doc.char_count
        word_count = extracted_doc.word_count
        page_count = extracted_doc.page_count
        read_time_min = max(1, round(word_count / 200))

        is_rag_required = char_count > settings.DIRECT_PROCESSING_CHAR_LIMIT
        rag_label = "⚡ Vector RAG Engine Active" if is_rag_required else "Direct Stream (< 12k chars)"

        st.markdown(
            f"""
            <div>
                <span class="metric-badge">📄 {page_count} page(s)</span>
                <span class="metric-badge">📝 {word_count:,} words</span>
                <span class="metric-badge">🔤 {char_count:,} chars</span>
                <span class="metric-badge">⏱️ ~{read_time_min} min read</span>
                <span class="rag-badge">{rag_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Analysis Objective")
    mode_selection = st.radio(
        "Mode:",
        options=["Summarize", "Key Points", "Simplify", "Ask Custom Question"],
        horizontal=True,
        label_visibility="collapsed",
    )

    custom_question = ""
    if mode_selection == "Ask Custom Question":
        custom_question = st.text_input(
            "Enter your question regarding the document:",
            placeholder="e.g., What are the key liabilities and termination terms in this agreement?",
        )

    run_clicked = st.button("Analyze Document →", type="primary", use_container_width=True)


with col_output:
    mode_title = mode_selection.upper() if mode_selection else "SUMMARY"
    st.markdown(f"#### 02 — The Reading ({mode_title})")

    output_container = st.empty()

    if not st.session_state.analysis_result and not run_clicked:
        output_container.markdown(
            """
            <div style="background-color: #fffaf0; border: 1px dashed #c9bfa8; border-radius: 4px; padding: 40px 24px; text-align: center; color: #a89e8c; font-style: italic;">
                Your analysis will appear here, streamed in live as it is generated...
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_clicked:
        doc_to_analyze = st.session_state.extracted_doc

        if not doc_to_analyze or not doc_to_analyze.full_text.strip():
            st.warning("⚠️ Please provide a document by uploading a file or pasting text.")
        elif mode_selection == "Ask Custom Question" and not custom_question.strip():
            st.warning("⚠️ Please enter a question to ask about the document.")
        else:
            st.session_state.active_mode_label = mode_selection
            st.session_state.retrieved_chunks = []

            # Instantiate services
            llm_service = LLMService()
            chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
            vector_store = VectorStore(embedding_service=SemanticEmbeddingService())
            rag_engine = RAGEngine(llm_service=llm_service, chunker=chunker, vector_store=vector_store)

            # Map mode
            if mode_selection == "Summarize":
                mode = AnalysisMode.SUMMARIZE
            elif mode_selection == "Key Points":
                mode = AnalysisMode.KEY_POINTS
            elif mode_selection == "Simplify":
                mode = AnalysisMode.SIMPLIFY
            else:
                mode = AnalysisMode.SUMMARIZE

            # Custom question handling
            if mode_selection == "Ask Custom Question":
                chunks = chunker.chunk_document(doc_to_analyze)
                vector_store.clear()
                vector_store.add_chunks(chunks)
                search_results = vector_store.similarity_search(query=custom_question, top_k=settings.TOP_K_CHUNKS)
                retrieved_chunks = [r.chunk for r in search_results]
                st.session_state.retrieved_chunks = retrieved_chunks

                context_sections = []
                for idx, c in enumerate(retrieved_chunks, 1):
                    p_info = f" (Page {c.page_number})" if c.page_number else ""
                    context_sections.append(f"--- Section {idx}{p_info} ---\n{c.text}")

                augmented_prompt = (
                    f"Document Context:\n\n"
                    + "\n\n".join(context_sections)
                    + f"\n\nQuestion: {custom_question}\n\n"
                    f"Answer the question thoroughly and concisely strictly based on the extracted context above."
                )

                custom_sys_prompt = (
                    "You are a precise document intelligence assistant. Answer user questions "
                    "accurately using only the provided context. Cite section or page numbers when available."
                )

                async def get_stream():
                    async for token in llm_service.stream_analysis(
                        mode=mode,
                        content=augmented_prompt,
                        custom_system_prompt=custom_sys_prompt,
                    ):
                        yield token

            else:
                # If RAG is triggered, retrieve chunks for transparency tab
                if rag_engine.is_large_document(doc_to_analyze.full_text):
                    chunks = chunker.chunk_document(doc_to_analyze)
                    vector_store.clear()
                    vector_store.add_chunks(chunks)
                    st.session_state.retrieved_chunks = rag_engine.retrieve_context(
                        mode=mode, top_k_per_query=settings.TOP_K_CHUNKS
                    )

                async def get_stream():
                    async for token in rag_engine.process_and_stream(doc=doc_to_analyze, mode=mode):
                        yield token

            def sync_stream_adapter(async_generator):
                """Bridge async generator to synchronous generator for st.write_stream."""
                loop = asyncio.new_event_loop()
                try:
                    while True:
                        try:
                            item = loop.run_until_complete(async_generator.__anext__())
                            yield item
                        except StopAsyncIteration:
                            break
                finally:
                    loop.close()

            # Stream response in Streamlit
            try:
                with output_container.container():
                    stream_generator = sync_stream_adapter(get_stream())
                    full_response = st.write_stream(stream_generator)
                    st.session_state.analysis_result = full_response
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # Persistent output view (when not streaming new run)
    elif st.session_state.analysis_result:
        with output_container.container():
            st.markdown(st.session_state.analysis_result)

    # Export & Download Actions
    if st.session_state.analysis_result:
        st.divider()
        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            st.download_button(
                label="📥 Download as Markdown (.md)",
                data=st.session_state.analysis_result,
                file_name=f"documind_{st.session_state.active_mode_label.lower().replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                label="📄 Download as Text (.txt)",
                data=st.session_state.analysis_result,
                file_name=f"documind_{st.session_state.active_mode_label.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # RAG Transparency & Retrieved Chunks Inspector
    if st.session_state.retrieved_chunks:
        with st.expander(f"🔍 Inspect Vector RAG Chunks ({len(st.session_state.retrieved_chunks)} sections retrieved)"):
            for i, chunk in enumerate(st.session_state.retrieved_chunks, 1):
                p_text = f" | Page {chunk.page_number}" if chunk.page_number else ""
                st.markdown(f"**Chunk #{chunk.chunk_id}** ({chunk.char_count} chars{p_text})")
                st.caption(chunk.text)
                st.divider()
