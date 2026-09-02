"""Base interface and data models for local vector stores."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from app.ingestion.chunking.base import Chunk
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.similarity import to_percentage

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_score: float = 0.0
    similarity_pct: float = 0.0
    rank: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "raw_score": self.raw_score,
            "similarity_pct": f"{self.similarity_pct:.2f}%",
            "similarity_pct_raw": self.similarity_pct,
            "rank": self.rank,
        }

class BaseVectorStore(ABC):
    """Abstract vector store interface with unified search and ingestion."""
    
    def __init__(self, collection_name: str, embedding_provider: BaseEmbeddingProvider):
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider

    @abstractmethod
    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """Upsert chunks into the vector store. Returns number of records inserted."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        """Perform similarity search for a query string."""
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """Wipe collection."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total document count in store."""
        pass
