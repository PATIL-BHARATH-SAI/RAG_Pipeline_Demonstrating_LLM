import os
import sys
from pathlib import Path

# Ensure root directory is on python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Prevent script name 'app.py' from shadowing the top-level 'app' package
if "app" in sys.modules and not hasattr(sys.modules["app"], "config"):
    del sys.modules["app"]

import streamlit as st
import importlib
import html
from app.config import settings
import app.agents.planner
import app.agents.responder
import app.agents.retriever
import app.agents.graph

# Ensure latest module code is always loaded on rerun
importlib.reload(app.agents.planner)
importlib.reload(app.agents.responder)
importlib.reload(app.agents.retriever)
importlib.reload(app.agents.graph)

from app.agents.graph import agent_graph
from app.ingestion.processor import process_and_ingest
from app.services.retrieval.vectorstore.factory import get_vectorstore
from app.services.retrieval.embeddings.factory import get_embedding_provider

st.set_page_config(
    page_title="Omni-fetch — Local & Modular RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .badge-match {
        background-color: #10B981;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .source-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    
    selected_chunker = st.selectbox(
        "✂️ Chunking Strategy",
        options=["paragraph", "fixed_size", "recursive", "semantic"],
        index=0
    )
    
    selected_embedder = st.selectbox(
        "🧠 Embedding Provider",
        options=["local", "openai", "gemini", "ollama"],
        index=0
    )
    
    selected_vdb = st.selectbox(
        "🗄️ Vector Store",
        options=["chroma", "faiss", "qdrant"],
        index=0
    )
    
    selected_llm = st.selectbox(
        "⚡ LLM Provider",
        options=["groq", "openai", "gemini", "ollama", "local"],
        index=0
    )
    
    # Update runtime settings
    settings.CHUNK_STRATEGY = selected_chunker
    settings.EMBEDDING_PROVIDER = selected_embedder
    settings.VECTOR_DB = selected_vdb
    settings.LLM_PROVIDER = selected_llm
    
    st.divider()
    
    st.markdown("### 📁 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md", "pptx"],
        accept_multiple_files=True
    )
    
    wipe_existing = st.checkbox("🧹 Wipe index before ingesting", value=False)
    
    if st.button("🚀 Ingest Uploaded Files", use_container_width=True):
        if uploaded_files:
            upload_dir = BASE_DIR / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded_files:
                file_path = upload_dir / f.name
                with open(file_path, "wb") as dest:
                    dest.write(f.getbuffer())
                    
            with st.spinner("Indexing documents..."):
                count = process_and_ingest(
                    input_path=str(upload_dir),
                    chunk_strategy=selected_chunker,
                    embedding_provider_name=selected_embedder,
                    vector_db_name=selected_vdb,
                    wipe=wipe_existing
                )
                st.success(f"Indexed {count} chunks into {selected_vdb.upper()}!")
        else:
            st.warning("Please select at least one file.")

    st.divider()
    try:
        vs = get_vectorstore(selected_vdb)
        doc_count = vs.count()
        st.metric("📊 Total Chunks in DB", doc_count)
    except Exception:
        st.metric("📊 Total Chunks in DB", "0")

# Main Interface
st.markdown('<div class="main-header">⚡ Omni-fetch: Local & Modular RAG Assistant</div>', unsafe_allow_html=True)
st.caption(f"Active Architecture: **{selected_chunker}** chunker ➔ **{selected_embedder}** embeddings ➔ **{selected_vdb}** local store ➔ **{selected_llm}** LLM gateway")

# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Cited Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    safe_preview = html.escape(src.get('preview', '')).replace('\n', '<br>')
                    st.markdown(f"""
                    <div class="source-box">
                        <b>{src['filename']}</b> <span class="badge-match">{src['match_pct']}</span><br>
                        <small>{safe_preview}</small>
                    </div>
                    """, unsafe_allow_html=True)

# User query input
user_input = st.chat_input("Ask me your question or doubt about AI, ML, or anything else related to Gen AI...", accept_file=True)
if user_input:
    # In Streamlit 1.30+, accept_file returns a ChatInputValue object with .text and .files attributes
    if hasattr(user_input, 'text'):
        prompt = user_input.text
        if hasattr(user_input, 'files') and user_input.files:
            # We can process uploaded files here in the future
            st.toast(f"📎 Received {len(user_input.files)} file(s)! (Backend OCR integration needed for full processing)")
    elif isinstance(user_input, dict): # Fallback just in case
        prompt = user_input.get("text", "")
    else:
        prompt = str(user_input)

    if prompt:
        # Render user prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Execute Agent graph
        with st.chat_message("assistant"):
            with st.spinner("Analyzing knowledge base & generating answer..."):
                result = agent_graph.query(prompt)

                if not result.is_safe:
                    st.error(f"🛡️ Guardrail Intervention: {result.response}")
                    st.session_state.messages.append({"role": "assistant", "content": f"🛡️ Blocked: {result.response}"})
                else:
                    st.markdown(result.response)
                    
                    # Render source citations
                    if result.source_documents:
                        with st.expander(f"📚 Cited Sources ({len(result.source_documents)})", expanded=True):
                            for src in result.source_documents:
                                safe_preview = html.escape(src.get('preview', '')).replace('\n', '<br>')
                                st.markdown(f"""
                                <div class="source-box">
                                    <b>{src['filename']}</b> <span class="badge-match">{src['match_pct']}</span><br>
                                    <small>{safe_preview}</small>
                                </div>
                                """, unsafe_allow_html=True)
                                
                    st.caption(f"⚡ Latency: {result.latency_ms:.1f}ms | Provider: `{result.provider_used}` | Model: `{result.model_used}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.response,
                        "sources": result.source_documents,
                        "latency": result.latency_ms,
                        "provider": result.provider_used,
                        "model": result.model_used
                    })
