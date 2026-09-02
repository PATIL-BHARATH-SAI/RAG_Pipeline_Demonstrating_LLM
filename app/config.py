"""Unified configuration for Omni-fetch Local & Modular RAG."""
import os
import sys
from pathlib import Path
from dataclasses import dataclass

# Protect against Windows [Errno 22] Invalid argument on redirected stderr/stdout in background processes
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "flush"):
        _orig_flush = _stream.flush
        def _make_safe_flush(orig):
            def _safe():
                try:
                    orig()
                except OSError:
                    pass
            return _safe
        _stream.flush = _make_safe_flush(_orig_flush)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env manually if dotenv isn't installed
env_path = BASE_DIR / ".env"
if env_path.exists():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

@dataclass
class Settings:
    # App Settings
    APP_NAME: str = "Omni-fetch Modular RAG"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENCODE_API_KEY: str = os.getenv("OPENCODE_API_KEY", "")
    
    # Chunking Strategy: 'paragraph' | 'fixed_size' | 'recursive' | 'semantic'
    CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "paragraph")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    
    # Embedding Provider: 'local' | 'gemini' | 'openai' | 'ollama'
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    LOCAL_EMBED_MODEL: str = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    
    # Vector DB: 'chroma' | 'faiss' | 'qdrant'
    VECTOR_DB: str = os.getenv("VECTOR_DB", "chroma")
    CHROMA_PERSIST_DIR: str = str(DATA_DIR / "chroma_db")
    FAISS_PERSIST_DIR: str = str(DATA_DIR / "faiss_index")
    QDRANT_PERSIST_DIR: str = str(DATA_DIR / "qdrant_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "omnifetch_docs")
    
    # Gateway & LLM: 'groq' | 'openai' | 'gemini' | 'ollama' | 'local'
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
    GATEWAY_DB_PATH: str = str(DATA_DIR / "gateway_audit.db")
    
    # Reranking
    ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")

settings = Settings()
