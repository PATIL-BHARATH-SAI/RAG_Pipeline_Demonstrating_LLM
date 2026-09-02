"""Omni-fetch Interactive CLI & Query Interface."""
import sys
import argparse
from app.agents.graph import agent_graph
from app.config import settings

def run_cli():
    print("\n" + "="*60)
    print("🤖 Omni-fetch — Local & Modular RAG Assistant")
    print(f"⚙️  Chunker: {settings.CHUNK_STRATEGY} | Embedder: {settings.EMBEDDING_PROVIDER} | Vector DB: {settings.VECTOR_DB} | LLM: {settings.LLM_PROVIDER}")
    print("="*60)
    print("Type your question below (or 'exit' to quit):\n")

    while True:
        try:
            query = input("\n🧑 Question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break

            print("\n⏳ Processing...")
            result = agent_graph.query(query)

            if not result.is_safe:
                print(f"\n🛡️ Guardrail Blocked: {result.guardrail_violations}")
                print(f"⚠️ {result.response}")
                continue

            print("\n" + "-"*60)
            print("📝 Answer:")
            print(result.response)
            print("-"*60)
            
            if result.source_documents:
                print("\n📚 Cited Sources:")
                for i, src in enumerate(result.source_documents, start=1):
                    print(f"  {i}. [{src['filename']}] (Match: {src['match_pct']})")
                    print(f"     Preview: {src['preview']}")

            print(f"\n⚡ Latency: {result.latency_ms:.1f}ms | Provider: {result.provider_used} | Model: {result.model_used}")
            print("="*60)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Omni-fetch RAG Query CLI")
    parser.add_argument("--query", "-q", help="Run a single query and exit")
    args = parser.parse_args()

    if args.query:
        result = agent_graph.query(args.query)
        print(result.response)
    else:
        run_cli()

if __name__ == "__main__":
    main()
