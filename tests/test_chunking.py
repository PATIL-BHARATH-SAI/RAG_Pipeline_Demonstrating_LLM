"""Unit tests for all pluggable chunking strategies."""
import pytest
from app.ingestion.chunking.factory import get_chunker

SAMPLE_TEXT = """Paragraph 1 is about artificial intelligence and deep neural networks.
It explains the core mechanisms of attention and backpropagation.

Paragraph 2 discusses Retrieval-Augmented Generation architectures.
It highlights how vector stores index semantic embeddings.

Paragraph 3 focuses on local LLM gateways and prompt safety guardrails."""

def test_paragraph_chunker():
    chunker = get_chunker("paragraph", chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) >= 2
    assert all(c.text for c in chunks)
    assert all(c.chunk_index is not None for c in chunks)

def test_fixed_size_chunker():
    chunker = get_chunker("fixed_size", chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)

def test_recursive_chunker():
    chunker = get_chunker("recursive", chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) > 1
    assert all(len(c.text) > 0 for c in chunks)

def test_semantic_chunker():
    chunker = get_chunker("semantic", chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk(SAMPLE_TEXT)
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)

def test_invalid_chunker_strategy():
    with pytest.raises(ValueError):
        get_chunker("unsupported_strategy")
