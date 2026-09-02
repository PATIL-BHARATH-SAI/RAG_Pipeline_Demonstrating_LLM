"""Planner node for query inspection and decomposition."""
from typing import Dict, Any
from app.agents.state import AgentState
from app.guardrails.local_guard import guardrails

def plan_node(state: AgentState) -> Dict[str, Any]:
    """Inspect query through guardrails and extract search terms."""
    guard_res = guardrails.validate_input(state.query)
    
    if not guard_res.is_safe:
        return {
            "is_safe": False,
            "guardrail_violations": guard_res.violations,
            "response": guard_res.sanitized_text,
            "search_queries": []
        }
        
    sanitized = guard_res.sanitized_text
    search_queries = [sanitized]

    # Acronym expansion for dense vector semantic matching
    acronyms = {
        r"\bml\b": "machine learning",
        r"\bdl\b": "deep learning",
        r"\bai\b": "artificial intelligence",
        r"\brag\b": "retrieval augmented generation",
        r"\bllm\b": "large language model",
        r"\bllms\b": "large language models",
        r"\bnlp\b": "natural language processing"
    }
    expanded = sanitized
    import re
    for pattern, full_term in acronyms.items():
        if re.search(pattern, sanitized, re.IGNORECASE):
            expanded = re.sub(pattern, full_term, expanded, flags=re.IGNORECASE)
            
    if expanded.lower() != sanitized.lower():
        search_queries.append(expanded)
        
    # Sub-query generation for multi-part questions
    if " and " in sanitized.lower() or " vs " in sanitized.lower() or " compare " in sanitized.lower():
        parts = [p.strip() for p in sanitized.replace("?", "").split(" and ") if p.strip()]
        if len(parts) > 1:
            search_queries.extend(parts)
            
    return {
        "is_safe": True,
        "sanitized_query": sanitized,
        "guardrail_violations": guard_res.violations,
        "search_queries": search_queries[:3]
    }
