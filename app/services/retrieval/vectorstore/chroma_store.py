"""ChromaDB local persistent vector store (100% free, local disk storage)."""
import os
import chromadb
from typing import List, Optional
from app.ingestion.chunking.base import Chunk
from app.services.retrieval.embeddings.base import BaseEmbeddingProvider
from app.services.retrieval.vectorstore.base import BaseVectorStore, SearchResult
from app.services.retrieval.similarity import to_percentage
from app.config import settings

class ChromaVectorStore(BaseVectorStore):
    """Local persistent ChromaDB vector store."""
    
    # ponytail: PersistentClient with cosine metric, upgrade to remote HTTP Chroma if distributed cluster needed
    def __init__(self, collection_name: str, embedding_provider: BaseEmbeddingProvider, persist_dir: Optional[str] = None):
        super().__init__(collection_name, embedding_provider)
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        valid_chunks = [c for c in chunks if isinstance(c.text, str) and c.text.strip()]
        if not valid_chunks:
            return 0
            
        batch_size = 500
        total_inserted = 0
        
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedding_provider.embed_texts(texts)
            ids = [c.chunk_id for c in batch]
            metadatas = [{k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in c.metadata.items()} for c in batch]
            self._collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
            total_inserted += len(batch)
            
        return total_inserted

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        query_emb = self.embedding_provider.embed_query(query)
        res = self._collection.query(query_embeddings=[query_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
        
        results = []
        docs, metas, dists, ids = res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("distances", [[]])[0], res.get("ids", [[]])[0]
        for rank, (doc_id, doc_text, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            cosine_sim = 1.0 - dist
            pct = to_percentage(cosine_sim, metric="cosine")
            if pct >= score_threshold:
                results.append(SearchResult(
                    chunk_id=doc_id, text=doc_text, metadata=meta or {},
                    raw_score=round(cosine_sim, 4), similarity_pct=pct, rank=rank
                ))
        return results

    def delete_collection(self) -> bool:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        return True

    def count(self) -> int:
        return self._collection.count()
