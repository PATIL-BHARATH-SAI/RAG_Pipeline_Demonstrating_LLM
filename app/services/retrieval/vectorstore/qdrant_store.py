"""Embedded Local Qdrant vector store (100% free, local embedded disk engine)."""
import os
from typing import List, Optional
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.ingestion.chunking.base import Chunk
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.vectorstore.base import BaseVectorStore, SearchResult
from app.services.retrieval.similarity import to_percentage
from app.config import settings

class QdrantVectorStore(BaseVectorStore):
    """Local embedded Qdrant vector store using local disk storage (zero cloud required)."""
    
    # ponytail: Embedded Qdrant disk client, upgrade to QdrantServer container if concurrent worker clustering needed
    def __init__(self, collection_name: str, embedding_provider: BaseEmbeddingProvider, persist_dir: Optional[str] = None):
        super().__init__(collection_name, embedding_provider)
        self.persist_dir = persist_dir or settings.QDRANT_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self._dim = self.embedding_provider.dim
        self._client = QdrantClient(path=self.persist_dir)
        collections = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE)
            )

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        embeddings = self.embedding_provider.embed_texts([c.text for c in chunks])
        points = [
            PointStruct(
                id=c.chunk_id, vector=emb,
                payload={"text": c.text, "metadata": c.metadata, "chunk_index": c.chunk_index}
            ) for c, emb in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)
        return len(chunks)

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        query_emb = self.embedding_provider.embed_query(query)
        if hasattr(self._client, "query_points"):
            res = self._client.query_points(collection_name=self.collection_name, query=query_emb, limit=top_k)
            hits = res.points
        else:
            hits = self._client.search(collection_name=self.collection_name, query_vector=query_emb, limit=top_k)
        
        results = []
        for rank, hit in enumerate(hits, start=1):
            raw_score = float(hit.score)
            pct = to_percentage(raw_score, metric="cosine")
            if pct >= score_threshold:
                payload = hit.payload or {}
                results.append(SearchResult(
                    chunk_id=str(hit.id), text=payload.get("text", ""),
                    metadata=payload.get("metadata", {}), raw_score=round(raw_score, 4),
                    similarity_pct=pct, rank=rank
                ))
        return results

    def delete_collection(self) -> bool:
        self._client.delete_collection(self.collection_name)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE)
        )
        return True

    def count(self) -> int:
        return self._client.get_collection(self.collection_name).points_count or 0
