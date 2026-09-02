"""Embeddings package."""
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.embeddings.local_provider import LocalEmbeddingProvider
from app.services.retrieval.embeddings.gemini_provider import GeminiEmbeddingProvider
from app.services.retrieval.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.services.retrieval.embeddings.factory import get_embedding_provider

__all__ = [
    "BaseEmbeddingProvider",
    "LocalEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "get_embedding_provider",
]
