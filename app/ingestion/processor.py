"""Document ingestion processor pipeline."""
import argparse
import sys
from pathlib import Path
from typing import Optional, List
from app.ingestion.loaders.document_loader import document_loader
from app.ingestion.chunking.factory import get_chunker
from app.services.retrieval.embeddings.factory import get_embedding_provider
from app.services.retrieval.vectorstore.factory import get_vectorstore
from app.config import settings

def process_and_ingest(
    input_path: str,
    chunk_strategy: Optional[str] = None,
    embedding_provider_name: Optional[str] = None,
    vector_db_name: Optional[str] = None,
    wipe: bool = False
) -> int:
    """Load documents, chunk them, embed them, and index into vector store."""
    strat = chunk_strategy or settings.CHUNK_STRATEGY
    embed_prov_name = embedding_provider_name or settings.EMBEDDING_PROVIDER
    v_db = vector_db_name or settings.VECTOR_DB

    print(f"\n=======================================================")
    print(f"🚀 Ingestion Pipeline: Starting")
    print(f"📂 Source: {input_path}")
    print(f"✂️  Chunking Strategy: {strat}")
    print(f"🧠 Embedding Provider: {embed_prov_name}")
    print(f"🗄️  Vector Store: {v_db}")
    print(f"=======================================================\n")

    # 1. Initialize components
    embedder = get_embedding_provider(embed_prov_name)
    vector_store = get_vectorstore(v_db, embedding_provider=embedder)
    chunker = get_chunker(strat, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP, embedding_provider=embedder)

    if wipe:
        print("🧹 Wiping existing collection...")
        vector_store.delete_collection()

    # 2. Load documents
    path = Path(input_path)
    if path.is_file():
        docs = [document_loader.load(path)]
    elif path.is_dir():
        docs = document_loader.load_directory(path)
    else:
        print(f"❌ Error: Path not found: {input_path}")
        return 0

    print(f"📄 Loaded {len(docs)} document(s).")

    # 3. Chunk documents
    all_chunks = []
    for doc in docs:
        if not doc.content.strip():
            continue
        chunks = chunker.chunk(doc.content, metadata=doc.metadata)
        all_chunks.extend(chunks)

    print(f"🧩 Generated {len(all_chunks)} chunks using strategy '{strat}'.")
    if not all_chunks:
        print("⚠️  No text found to index.")
        return 0

    # 4. Upsert chunks into vector store
    inserted = vector_store.upsert_chunks(all_chunks)
    total_count = vector_store.count()
    print(f"✅ Successfully indexed {inserted} chunks. Total records in store: {total_count}")
    return inserted

def main():
    parser = argparse.ArgumentParser(description="Omni-fetch Ingestion Pipeline")
    parser.add_argument("path", help="Path to a file or directory of documents to ingest")
    parser.add_argument("--wipe", action="store_true", help="Wipe collection before ingestion")
    parser.add_argument("--strategy", default=None, help="Chunking strategy (paragraph, fixed_size, recursive, semantic)")
    parser.add_argument("--provider", default=None, help="Embedding provider (local, openai, gemini, ollama)")
    parser.add_argument("--db", default=None, help="Vector store (chroma, faiss, qdrant)")
    
    args = parser.parse_args()
    process_and_ingest(
        input_path=args.path,
        chunk_strategy=args.strategy,
        embedding_provider_name=args.provider,
        vector_db_name=args.db,
        wipe=args.wipe
    )

if __name__ == "__main__":
    main()
