"""Local embedding provider using SentenceTransformers (100% offline & free)."""
import os
import sys
from typing import List

# Suppress progress bars and broken pipe errors on Windows
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

if hasattr(sys.stderr, "flush"):
    _orig_err_flush = sys.stderr.flush
    def _safe_err_flush():
        try:
            _orig_err_flush()
        except OSError:
            pass
    sys.stderr.flush = _safe_err_flush

from sentence_transformers import SentenceTransformer
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider

class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local HuggingFace/SentenceTransformer provider running purely on local CPU/GPU."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        try:
            # Try loading from local HuggingFace cache first for 100% offline instant startup
            self._model = SentenceTransformer(self._model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(self._model_name)
            
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = self._model.get_embedding_dimension()
        else:
            self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_query(self, query: str) -> List[float]:
        q = str(query).strip() if query else "query"
        return self.embed_texts([q])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        all_embeddings = []
        batch_size = 128
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            clean_batch = []
            for t in batch:
                if t is None or not isinstance(t, str):
                    clean_batch.append("empty chunk")
                else:
                    s = t.strip()
                    clean_batch.append(s if len(s) > 0 else "empty chunk")
                    
            embs = self._model.encode(
                clean_batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.extend(embs.tolist())
            
        return all_embeddings
