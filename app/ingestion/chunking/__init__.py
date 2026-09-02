"""Chunking strategies module."""
from app.ingestion.chunking.base import BaseChunker, Chunk
from app.ingestion.chunking.paragraph import ParagraphChunker
from app.ingestion.chunking.fixed_size import FixedSizeChunker
from app.ingestion.chunking.recursive import RecursiveChunker
from app.ingestion.chunking.semantic import SemanticChunker
from app.ingestion.chunking.factory import get_chunker

__all__ = [
    "BaseChunker",
    "Chunk",
    "ParagraphChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "get_chunker",
]
