"""Benchmark embedding providers across dimensions, batch latency, and throughput."""
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.services.retrieval.embeddings.factory import get_embedding_provider

BENCHMARK_TEXTS = [
    "Retrieval-Augmented Generation architectures combine dense vector index with autoregressive language models.",
    "Local vector databases such as Chroma, FAISS, and Qdrant provide embedded persistence.",
    "Prompt injections and adversarial jailbreak attacks are neutralized by local guardrails.",
    "Cross-encoder rerankers dramatically boost top-k retrieval precision."
]

def benchmark_embedding_providers():
    providers = ["local", "openai", "gemini"]
    print("="*70)
    print("🧠 Omni-fetch Multi-Provider Embeddings Benchmark")
    print("="*70)
    print(f"{'Provider':<12} | {'Dimensions':<10} | {'Query Lat (ms)':<15} | {'Batch Lat (ms)':<15}")
    print("-"*70)

    for prov in providers:
        try:
            embedder = get_embedding_provider(prov)
            t0 = time.time()
            _ = embedder.embed_query("What is enterprise RAG?")
            q_lat = (time.time() - t0) * 1000.0

            t1 = time.time()
            _ = embedder.embed_texts(BENCHMARK_TEXTS)
            b_lat = (time.time() - t1) * 1000.0

            print(f"{prov:<12} | {embedder.dim:<10} | {q_lat:<15.2f} | {b_lat:<15.2f}")
        except Exception as e:
            print(f"{prov:<12} | {'Error':<10} | {str(e)[:30]:<30}")

    print("="*70)

if __name__ == "__main__":
    benchmark_embedding_providers()
