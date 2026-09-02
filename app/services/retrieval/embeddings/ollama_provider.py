"""Ollama local embedding provider (100% free, local self-hosted API)."""
from typing import List
import requests
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.embeddings.local_provider import LocalEmbeddingProvider

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Local Ollama embeddings running through local Ollama instance (e.g. nomic-embed-text)."""
    
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._dim = 768
        self._fallback = LocalEmbeddingProvider()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            embeddings = []
            for text in texts:
                resp = requests.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model_name, "prompt": text},
                    timeout=5
                )
                if resp.status_code == 200:
                    data = resp.json()
                    vec = data.get("embedding", [])
                    if vec:
                        self._dim = len(vec)
                        embeddings.append(vec)
                    else:
                        raise ValueError("Empty embedding returned from Ollama")
                else:
                    raise ConnectionError(f"Ollama returned HTTP {resp.status_code}")
            return embeddings
        except Exception:
            # Automatic graceful fallback to local sentence-transformer
            return self._fallback.embed_texts(texts)
