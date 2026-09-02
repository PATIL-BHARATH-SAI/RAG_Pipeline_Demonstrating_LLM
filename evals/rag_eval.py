"""End-to-end local evaluation metrics for RAG accuracy and faithfulness."""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.agents.graph import agent_graph

TEST_DATASET = [
    {
        "question": "What is the primary role of chunking in RAG pipelines?",
        "ground_truth_keywords": ["chunk", "granularity", "retrieval", "context"]
    },
    {
        "question": "How do local guardrails prevent prompt injection?",
        "ground_truth_keywords": ["injection", "jailbreak", "sanitize", "pattern", "filter"]
    }
]

def evaluate_pipeline():
    print("="*65)
    print("📈 Omni-fetch RAG Pipeline Quality Evaluation")
    print("="*65)

    scores = []
    for item in TEST_DATASET:
        q = item["question"]
        expected = item["ground_truth_keywords"]
        
        t0 = time.time()
        res = agent_graph.query(q)
        lat = (time.time() - t0) * 1000.0
        
        answer = res.response.lower()
        found_keywords = [k for k in expected if k in answer]
        recall = len(found_keywords) / len(expected)
        has_sources = len(res.source_documents) > 0
        
        scores.append({
            "question": q,
            "latency_ms": lat,
            "keyword_recall": recall,
            "has_sources": has_sources,
            "provider": res.provider_used
        })
        
        print(f"\n❓ Question: {q}")
        print(f"   Keyword Recall: {recall*100:.1f}% ({len(found_keywords)}/{len(expected)})")
        print(f"   Sources Attached: {has_sources} ({len(res.source_documents)} chunks)")
        print(f"   Latency: {lat:.1f}ms (Provider: {res.provider_used})")

    avg_recall = sum(s["keyword_recall"] for s in scores) / len(scores)
    avg_lat = sum(s["latency_ms"] for s in scores) / len(scores)
    print("\n" + "="*65)
    print(f"🏁 Overall Keyword Grounding Recall: {avg_recall*100:.1f}%")
    print(f"⚡ Average End-to-End Latency: {avg_lat:.1f}ms")
    print("="*65)

if __name__ == "__main__":
    evaluate_pipeline()
