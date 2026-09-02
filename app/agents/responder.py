"""Responder node synthesizing validated answers with match percentages and citations."""
import time
from typing import Dict, Any
from app.agents.state import AgentState
from app.gateway.local_gateway import gateway
from app.gateway.base import GatewayMessage
from app.guardrails.local_guard import guardrails

SYSTEM_PROMPT = """You are Omni-fetch, an advanced enterprise RAG assistant.
Synthesize a comprehensive, clear, structured explanation addressing the user's question using the provided context excerpts.
Cite the relevant source document filename and similarity match percentage for each key point (e.g., [filename | Match: XX.XX%]).
Connect all relevant concepts found in the context (such as models, algorithms, architectures, metrics, or training) to thoroughly answer the query.
Do NOT output <think> or internal reasoning tags. Directly provide your structured, polished answer."""

def respond_node(state: AgentState) -> Dict[str, Any]:
    """Synthesize final response from reranked context and check output guardrails."""
    if not state.is_safe:
        return {"response": state.response}
        
    start_time = time.time()
    
    # Format context
    context_blocks = []
    sources = []
    
    for r in state.reranked_chunks:
        filename = r.get("metadata", {}).get("filename", "Document")
        pct = r.get("similarity_pct", "0.00%")
        text = r.get("text", "")
        context_blocks.append(f"[{filename} | Match: {pct}]\n{text}")
        sources.append({
            "filename": filename,
            "match_pct": pct,
            "chunk_id": r.get("chunk_id"),
            "preview": text[:150] + "..." if len(text) > 150 else text
        })
        
    if not context_blocks:
        return {
            "response": "No relevant documents found in the vector store matching your query. Please index documents first.",
            "source_documents": [],
            "latency_ms": (time.time() - start_time) * 1000.0,
            "model_used": "none",
            "provider_used": "none"
        }
        
    context_str = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"Context:\n{context_str}\n\nUser Question: {state.sanitized_query}\n\nAnswer:"
    
    messages = [
        GatewayMessage(role="system", content=SYSTEM_PROMPT),
        GatewayMessage(role="user", content=user_prompt)
    ]
    
    gw_resp = gateway.generate(messages, temperature=0.3, max_tokens=2048)
    
    # Strip internal reasoning tokens (<think>...</think>) from reasoning models
    raw_content = gw_resp.content
    import re
    cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    if not cleaned_content and raw_content:
        cleaned_content = raw_content
    
    # Run output guardrail validation
    validated_out = guardrails.validate_output(cleaned_content, context=context_str)
    
    total_lat = (time.time() - start_time) * 1000.0 + gw_resp.latency_ms
    
    return {
        "response": validated_out.sanitized_text,
        "source_documents": sources,
        "latency_ms": round(total_lat, 2),
        "model_used": gw_resp.model,
        "provider_used": gw_resp.provider
    }
