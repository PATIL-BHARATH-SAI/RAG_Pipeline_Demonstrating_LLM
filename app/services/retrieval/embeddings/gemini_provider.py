"""Gemini embedding provider with automatic local fallback."""
import os
from typing import List
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.embeddings.local_provider import LocalEmbeddingProvider

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider with automatic fallback to local sentence-transformers."""
    
    def __init__(self, api_key: str = "", model_name: str = "models/text-embedding-004"):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model_name = model_name
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
            
        if not self._api_key:
            return self._fallback.embed_texts(texts)
            
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            results = []
            for text in texts:
                result = genai.embed_content(
                    model=self._model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                vec = result.get("embedding", [])
                if vec:
                    self._dim = len(vec)
                    results.append(vec)
                else:
                    raise ValueError("Empty embedding returned by Gemini")
            return results
        except Exception:
            # Graceful local fallback if offline or invalid key
            return self._fallback.embed_texts(texts)
