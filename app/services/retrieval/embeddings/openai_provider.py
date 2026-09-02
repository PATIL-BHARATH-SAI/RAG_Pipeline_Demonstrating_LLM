"""OpenAI embedding provider with local fallback."""
import os
from typing import List
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.embeddings.local_provider import LocalEmbeddingProvider

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embeddings provider (e.g. text-embedding-3-small)."""
    
    def __init__(self, api_key: str = "", model_name: str = "text-embedding-3-small"):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model_name = model_name
        self._dim = 1536
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
        if not self._api_key:
            return self._fallback.embed_texts(texts)
            
        try:
            import requests
            headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
            resp = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json={"input": texts, "model": self._model_name},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                if embeddings:
                    self._dim = len(embeddings[0])
                return embeddings
            else:
                return self._fallback.embed_texts(texts)
        except Exception:
            return self._fallback.embed_texts(texts)
