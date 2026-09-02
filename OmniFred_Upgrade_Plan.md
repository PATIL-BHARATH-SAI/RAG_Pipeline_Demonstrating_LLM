# Omni-fetch — Upgrade Plan (built on 8hr-MARATHON codebase)

## What you already have (verified from the zip)

| Layer | Current implementation | File |
|---|---|---|
| Loaders | PDF/HTML/TXT/DOCX/PPTX, local parsing | `app/ingestion/loaders/*.py` |
| Chunking | **One strategy only** — paragraph split, 1500-char cap | `app/ingestion/chunking/splitter.py` |
| Embeddings | Gemini `gemini-embedding-2-preview` (3072-dim), auto-fallback to `all-mpnet-base-v2` (768-dim) if Gemini probe fails | `app/services/retrieval/embedding.py` |
| Vector DB | Qdrant Cloud only | `app/services/retrieval/qdrant_service.py` |
| Reranking | FlashRank local cross-encoder | `app/services/retrieval/ranking_service.py` |
| Similarity score | Raw Qdrant cosine score returned, **not normalized to %** | `search_enterprise_knowledge()` in `qdrant_service.py` |
| Agent | LangGraph planner → retriever → responder | `app/agents/` |
| Guardrails | NeMo Guardrails | `app/guardrails/` |
| Evals | RAGAS, 6 metrics, golden dataset | `evals/` |

This is already a strong single-path pipeline. Your ask (multiple chunking strategies, multi-platform embedding comparison, multi vector-DB, real % similarity) means turning three hard-coded modules into **pluggable, swappable strategies**, config-selected — not replacing the working agent/guardrails/eval layers, which stay as-is.

---

## Step 1 — Chunking: turn `splitter.py` into a strategy package

Replace the single `chunk_text()` function with a package:

```
app/ingestion/chunking/
├── base.py              # Chunker interface: chunk(text, metadata) -> list[Chunk]
├── paragraph.py          # your existing logic, moved here unchanged
├── fixed_size.py          # naive token/char window + overlap
├── recursive.py           # LangChain RecursiveCharacterTextSplitter wrapper
├── semantic.py             # embedding-similarity breakpoint chunking
└── factory.py               # get_chunker(strategy_name) from config.py
```

`processor.py` currently calls `chunk_text(text)` directly — change that one call site to `factory.get_chunker(settings.CHUNK_STRATEGY).chunk(text, metadata)`. Add `CHUNK_STRATEGY` to `app/config.py`. This is the only integration point that needs touching.

## Step 2 — Embeddings: generalize `embedding.py` into a provider registry

Your fallback logic (Gemini → sentence-transformers) is already provider-aware — extend that same pattern instead of rewriting it:

```
app/services/retrieval/embeddings/
├── base.py            # EmbeddingProvider: embed_query(), embed_texts(), dim
├── gemini_provider.py  # your existing code, moved here
├── openai_provider.py
├── local_provider.py    # your existing sentence-transformers fallback, promoted to a first-class option
└── benchmark.py           # run same chunk set through each provider, log dim/latency/recall
```

`get_embedding_dim()` becomes `provider.dim` — needed because your Qdrant collection is created with a fixed vector size, so switching providers means either separate collections per provider or a re-index. Note this explicitly in `factory` docstring so Antigravity doesn't silently break the collection schema.

## Step 3 — Vector store: abstract `qdrant_service.py` behind an interface

```
app/services/retrieval/vectorstore/
├── base.py           # VectorStore: upsert(), search(), delete()
├── qdrant_store.py    # your existing qdrant_service.py logic, moved here
├── faiss_store.py       # new — local, fast, good for offline dev/testing
├── chroma_store.py        # new — persistent local option with metadata filtering
└── factory.py                # get_vectorstore(name) from config.py
```

`main.py` and `graph.py` currently import `search_enterprise_knowledge` from `qdrant_service` directly — repoint that import to `factory.get_vectorstore(settings.VECTOR_DB).search(...)`.

## Step 4 — Similarity scoring: fix the missing % conversion

This is the smallest but most visible gap. In `qdrant_service.py`, `res.score` is returned raw. Add a `similarity.py`:

```python
def to_percentage(raw_score: float, metric: str) -> float:
    # cosine (Qdrant default): score is already in [-1, 1] -> rescale to 0-100
    # dot product / euclidean: needs different normalization — document the formula used
    ...
```

Every result dict from every vector store's `search()` should include both `raw_score` and `similarity_pct`, so the UI can display "87% match" next to each retrieved chunk. Update `ui/app.py` to render it.

## Step 5 — Config wiring

Add to `app/config.py`:
```python
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "paragraph")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
VECTOR_DB = os.getenv("VECTOR_DB", "qdrant")
```
This is what lets you A/B different combinations without touching code — and what your `benchmark.py` notebooks sweep over.

## Step 6 — Re-index and evaluate

Because chunking and embedding strategy both change vector output, re-run:
```
python -m app.ingestion.processor DATA --wipe
```
per strategy/provider/store combination, then run `evals/pipeline.py` against each to get RAGAS scores per combination — that's your evidence for which setup is actually "advanced" vs just different.

---

## What NOT to touch

Leave these exactly as they are — they're already solid and unrelated to this upgrade:
- `app/guardrails/` (NeMo Guardrails)
- `app/gateway/` (Portkey + Groq routing)
- `app/agents/` (LangGraph planner/retriever/responder)
- `ranking_service.py` (FlashRank) — reranking sits after vector search regardless of which store/embedding you pick

## Handoff to Antigravity

Give Antigravity this file plus the unzipped repo, and ask it to:
1. Do Step 1 (chunking) fully, keep everything else running on the old single-path code.
2. Confirm existing tests/evals still pass with `CHUNK_STRATEGY=paragraph` (should be a no-op).
3. Only then move to Step 2, then 3, then 4 — each step should leave the app in a runnable state before starting the next.
