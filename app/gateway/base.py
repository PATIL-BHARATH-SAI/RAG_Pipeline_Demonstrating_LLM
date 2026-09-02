"""Base interface and data structures for LLM Gateway."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class GatewayMessage:
    role: str # 'system', 'user', 'assistant'
    content: str

@dataclass
class GatewayResponse:
    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None

class BaseLLMGateway(ABC):
    """Abstract base class for LLM gateway router."""

    @abstractmethod
    def generate(
        self,
        messages: List[GatewayMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        provider: Optional[str] = None
    ) -> GatewayResponse:
        """Generate a completion with automatic fallback and local logging."""
        pass
