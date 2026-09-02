# ⚡ Omni-fetch: Local & Modular Enterprise RAG Architecture

Omni-fetch is a modular, extensible, and **100% free / local-first runnable** Retrieval-Augmented Generation (RAG) system built with Python, LangGraph, Streamlit, and modern vector databases.

---

## 🌟 Key Highlights

- **100% Free & Local-First**: Replaced expensive/proprietary cloud components (NeMo Guardrails, Portkey, Qdrant Cloud) with pure open-source, local equivalents:
  - **Local Safety Guardrails**: Heuristic and regex-based prompt injection detection and PII masking.
  - **Resilient LLM Gateway**: Multi-model fallback chain (`Groq` ➔ `OpenAI` ➔ `Gemini` ➔ `Ollama` ➔ `Local Deterministic Offline Synthesizer`) with SQLite audit logging.
  - **Local Vector Stores**: Embedded ChromaDB, FAISS FlatIP, and local Qdrant.
- **Pluggable Chunking Suite**: Switch effortlessly between:
  - `paragraph`: Semantic boundary paragraph chunking.
  - `fixed_size`: Fixed character windows with configurable overlap.
  - `recursive`: Smart structural hierarchy splitting (`\n\n`, `\n`, ` `, `""`).
  - `semantic`: Embedding cosine similarity breakpoint chunking.
- **Multi-Provider Embeddings**:
  - `local`: Offline `sentence-transformers/all-MiniLM-L6-v2` (zero cost, runs on CPU/GPU).
  - `openai`: `text-embedding-3-small`.
  - `gemini`: Google Gemini embeddings.
- **FlashRank Cross-Encoder Reranking**: Sub-5ms reranking improving retrieval precision.
- **Similarity Match Percentages**: Normalizes raw Cosine $[-1, 1]$ and Euclidean distances to human-readable percentages (e.g., `89.4% Match`).
- **Streamlit Web UI & CLI**: Beautiful dark-mode interface with live strategy switching, file uploading, and source citation inspector.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────┐
│  Local Safety Guardrails             │
│  (PII Masking, Prompt Injection)     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Query Decomposition & Acronym Exp.  │
│  (ml -> machine learning, etc.)      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Vector Store Retrieval              │
│  (ChromaDB / FAISS / Qdrant)         │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  FlashRank Cross-Encoder Reranking   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Resilient Multi-Model LLM Gateway   │
│  (Groq Qwen/Llama -> OpenAI -> Local)│
└──────────────────┬───────────────────┘
                   │
                   ▼
       Answer + Match % Citations
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/omni-fetch.git
cd omni-fetch

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file based on `.env.example`:
```ini
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here

# Strategy Defaults
CHUNK_STRATEGY=recursive
EMBEDDING_PROVIDER=local
VECTOR_DB=chroma
LLM_PROVIDER=groq
```

### 3. Ingest Documents
```bash
# Ingest single file or folder
python -m app.ingestion.processor data/sample_enterprise_doc.md --strategy recursive --db chroma --provider local
```

### 4. Run the Web Interface
```bash
streamlit run ui/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Testing

Run the full automated test suite:
```bash
pytest tests/ -v
```

Run strategy benchmarks:
```bash
# Benchmark chunking strategies
python evals/benchmark_chunking.py

# Benchmark embeddings
python evals/benchmark_embeddings.py

# Benchmark end-to-end RAG quality
python evals/rag_eval.py
```

---

## 📄 License
MIT License
