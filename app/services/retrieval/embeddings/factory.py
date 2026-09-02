"""Factory for creating embedding providers dynamically."""
from typing import Optional
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.embeddings.local_provider import LocalEmbeddingProvider
from app.services.retrieval.embeddings.gemini_provider import GeminiEmbeddingProvider
from app.services.retrieval.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.services.retrieval.embeddings.ollama_provider import OllamaEmbeddingProvider
from app.config import settings

def get_embedding_provider(provider_name: Optional[str] = None) -> BaseEmbeddingProvider:
    """Instantiate and return an embedding provider based on provider name.
    
    Supported providers:
      - 'local': 100% offline SentenceTransformers (default)
      - 'openai': OpenAI embeddings
      - 'gemini': Google Gemini embedding with local fallback
      - 'ollama': Local self-hosted Ollama server
    """
    name = (provider_name or settings.EMBEDDING_PROVIDER).lower().strip()
    
    if name == "local":
        return LocalEmbeddingProvider(model_name=settings.LOCAL_EMBED_MODEL)
    elif name == "openai":
        return OpenAIEmbeddingProvider(api_key=settings.OPENAI_API_KEY)
    elif name == "gemini":
        return GeminiEmbeddingProvider(api_key=settings.GEMINI_API_KEY)
    elif name == "ollama":
        return OllamaEmbeddingProvider(
            model_name=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
    else:
        raise ValueError(f"Unknown embedding provider: '{provider_name}'. Supported: 'local', 'openai', 'gemini', 'ollama'.")
