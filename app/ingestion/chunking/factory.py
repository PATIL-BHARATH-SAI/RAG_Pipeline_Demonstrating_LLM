"""Factory for creating chunkers by strategy name."""
from typing import Optional
from app.ingestion.chunking.base import BaseChunker
from app.ingestion.chunking.paragraph import ParagraphChunker
from app.ingestion.chunking.fixed_size import FixedSizeChunker
from app.ingestion.chunking.recursive import RecursiveChunker
from app.ingestion.chunking.semantic import SemanticChunker

def get_chunker(strategy: str = "paragraph", chunk_size: int = 1000, chunk_overlap: int = 150, embedding_provider = None) -> BaseChunker:
    """Instantiate and return a chunker based on strategy name."""
    strat = strategy.lower().strip()
    if strat == "paragraph":
        return ParagraphChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strat == "fixed_size":
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strat == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strat == "semantic":
        return SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, embedding_provider=embedding_provider)
    else:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'. Supported strategies: 'paragraph', 'fixed_size', 'recursive', 'semantic'.")
