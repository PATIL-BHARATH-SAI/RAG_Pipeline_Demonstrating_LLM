"""Agent State schema."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    query: str
    sanitized_query: str = ""
    is_safe: bool = True
    guardrail_violations: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    response: str = ""
    source_documents: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    model_used: str = ""
    provider_used: str = ""
