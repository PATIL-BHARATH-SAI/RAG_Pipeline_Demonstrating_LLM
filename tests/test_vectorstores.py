"""Unit tests for local vector stores (Chroma, FAISS, Qdrant)."""
import pytest
from app.ingestion.chunking.base import Chunk
from app.services.retrieval.embeddings.local_provider import LocalEmbeddingProvider
from app.services.retrieval.vectorstore.factory import get_vectorstore

SAMPLE_CHUNKS = [
    Chunk(text="Vector databases store embeddings for similarity search.", metadata={"category": "ai"}),
    Chunk(text="Local guardrails inspect user prompts for safety and security.", metadata={"category": "security"}),
    Chunk(text="Chunking strategies break large documents into coherent sections.", metadata={"category": "rag"})
]

@pytest.mark.parametrize("store_name", ["chroma", "faiss", "qdrant"])
def test_vector_store_crud(store_name):
    embedder = LocalEmbeddingProvider()
    store = get_vectorstore(store_name, collection_name="test_collection", embedding_provider=embedder)
    
    # Wipe
    store.delete_collection()
    
    # Insert
    inserted = store.upsert_chunks(SAMPLE_CHUNKS)
    assert inserted == len(SAMPLE_CHUNKS)
    
    # Count
    count = store.count()
    assert count >= len(SAMPLE_CHUNKS)
    
    # Search
    results = store.search("similarity search embeddings", top_k=2)
    assert len(results) > 0
    assert results[0].similarity_pct >= 0.0
    assert "%" in results[0].to_dict()["similarity_pct"]
