"""Benchmark chunking strategies across text length, chunk count, and throughput."""
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.ingestion.chunking.factory import get_chunker

SAMPLE_CORPUS = """
Artificial Intelligence (AI) is transforming enterprise operations across global industries.
Retrieval-Augmented Generation (RAG) combines dense semantic retrieval with generative LLMs to produce grounded, hallucination-resistant answers.

In traditional retrieval systems, keyword matching algorithms like BM25 were standard. Modern semantic search utilizes vector embeddings in high-dimensional spaces to capture context, intent, and subtle conceptual relationships.

Chunking is the critical first stage in any RAG data pipeline. The choice of chunking strategy directly dictates the granularity of indexed context.
If chunks are too small, critical context is fragmented. If chunks are too large, irrelevant noise dilutes the semantic signal and consumes excessive context window space.

Guardrails provide real-time runtime safety filters, inspecting incoming queries for adversarial jailbreak attacks, prompt injections, and PII leakage before data reaches the model.
"""

def benchmark_chunking_strategies():
    strategies = ["paragraph", "fixed_size", "recursive", "semantic"]
    print("="*65)
    print("📊 Omni-fetch Chunking Strategy Benchmark")
    print("="*65)
    print(f"{'Strategy':<15} | {'Chunks':<8} | {'Avg Chars':<10} | {'Latency (ms)':<12}")
    print("-"*65)

    for strat in strategies:
        start = time.time()
        chunker = get_chunker(strat, chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(SAMPLE_CORPUS)
        elapsed_ms = (time.time() - start) * 1000.0
        
        avg_chars = sum(len(c.text) for c in chunks) / max(1, len(chunks))
        print(f"{strat:<15} | {len(chunks):<8} | {avg_chars:<10.1f} | {elapsed_ms:<12.3f}")

    print("="*65)

if __name__ == "__main__":
    benchmark_chunking_strategies()
