"""Factory for creating local vector stores."""
from typing import Optional
from app.services.retrieval.vectorstore.base import BaseVectorStore
from app.services.retrieval.vectorstore.chroma_store import ChromaVectorStore
from app.services.retrieval.vectorstore.faiss_store import FaissVectorStore
from app.services.retrieval.vectorstore.qdrant_store import QdrantVectorStore
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.embeddings.factory import get_embedding_provider
from app.config import settings

def get_vectorstore(
    store_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_provider: Optional[BaseEmbeddingProvider] = None
) -> BaseVectorStore:
    """Instantiate and return a vector store based on store name.
    
    Supported vector stores:
      - 'chroma': Local persistent ChromaDB (default)
      - 'faiss': High performance local FAISS index
      - 'qdrant': Local embedded Qdrant disk/memory engine (no cloud needed)
    """
    name = (store_name or settings.VECTOR_DB).lower().strip()
    coll = collection_name or settings.COLLECTION_NAME
    embedder = embedding_provider or get_embedding_provider()
    
    if name == "chroma":
        return ChromaVectorStore(collection_name=coll, embedding_provider=embedder)
    elif name == "faiss":
        return FaissVectorStore(collection_name=coll, embedding_provider=embedder)
    elif name == "qdrant":
        return QdrantVectorStore(collection_name=coll, embedding_provider=embedder)
    else:
        raise ValueError(f"Unknown vector store: '{store_name}'. Supported: 'chroma', 'faiss', 'qdrant'.")
