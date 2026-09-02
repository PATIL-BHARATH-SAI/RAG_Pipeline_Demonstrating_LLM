"""Retriever and Reranker node for agent workflow."""
from typing import Dict, Any, List
from app.agents.state import AgentState
from app.services.retrieval.vectorstore.factory import get_vectorstore
from app.services.retrieval.ranking_service import ranking_service
from app.config import settings

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Execute vector search across all planned search queries and rerank results."""
    if not state.is_safe or not state.search_queries:
        return {"retrieved_chunks": [], "reranked_chunks": []}

    vector_store = get_vectorstore(settings.VECTOR_DB)
    
    seen_ids = set()
    raw_results = []
    
    for q in state.search_queries:
        results = vector_store.search(q, top_k=5)
        for r in results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                raw_results.append(r)

    # Rerank combined results
    reranked = ranking_service.rerank(state.sanitized_query, raw_results, top_k=4)
    
    return {
        "retrieved_chunks": [r.to_dict() for r in raw_results],
        "reranked_chunks": [r.to_dict() for r in reranked]
    }
