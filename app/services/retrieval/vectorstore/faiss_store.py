"""FAISS local vector store (100% free, high-performance in-memory & disk persistence)."""
import os
import json
import faiss
import numpy as np
from typing import List, Optional
from app.ingestion.chunking.base import Chunk
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.vectorstore.base import BaseVectorStore, SearchResult
from app.services.retrieval.similarity import to_percentage
from app.config import settings

class FaissVectorStore(BaseVectorStore):
    """Local FAISS vector store with local disk persistence."""
    
    # ponytail: IndexFlatIP for exact cosine search with local disk JSON metadata, upgrade to IndexHNSWFlat if docs > 500k
    def __init__(self, collection_name: str, embedding_provider: BaseEmbeddingProvider, persist_dir: Optional[str] = None):
        super().__init__(collection_name, embedding_provider)
        self.persist_dir = persist_dir or settings.FAISS_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_file = os.path.join(self.persist_dir, f"{self.collection_name}.index")
        self.meta_file = os.path.join(self.persist_dir, f"{self.collection_name}_meta.json")
        self._dim = self.embedding_provider.dim
        self._chunks_data: List[dict] = []
        self._index = self._load_or_create_index()

    def _load_or_create_index(self) -> faiss.Index:
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self._chunks_data = json.load(f)
            return faiss.read_index(self.index_file)
        return faiss.IndexFlatIP(self._dim)

    def _save(self):
        faiss.write_index(self._index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self._chunks_data, f, indent=2)

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        embeddings = self.embedding_provider.embed_texts([c.text for c in chunks])
        vecs = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vecs)
        self._index.add(vecs)
        for c in chunks:
            self._chunks_data.append({"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata})
        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        if not self._chunks_data or self._index.ntotal == 0:
            return []
        q_vec = np.array([self.embedding_provider.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        distances, indices = self._index.search(q_vec, min(top_k, self._index.ntotal))
        
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if 0 <= idx < len(self._chunks_data):
                item = self._chunks_data[idx]
                pct = to_percentage(float(dist), metric="cosine")
                if pct >= score_threshold:
                    results.append(SearchResult(
                        chunk_id=item["chunk_id"], text=item["text"],
                        metadata=item.get("metadata", {}), raw_score=round(float(dist), 4),
                        similarity_pct=pct, rank=rank
                    ))
        return results

    def delete_collection(self) -> bool:
        self._chunks_data = []
        self._index = faiss.IndexFlatIP(self._dim)
        for f in [self.index_file, self.meta_file]:
            if os.path.exists(f):
                os.remove(f)
        return True

    def count(self) -> int:
        return self._index.ntotal
