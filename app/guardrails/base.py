"""Guardrails base interfaces and response models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class GuardrailResult:
    is_safe: bool
    sanitized_text: str
    violations: List[str] = field(default_factory=list)
    risk_score: float = 0.0 # 0.0 to 1.0

class BaseGuardrail(ABC):
    """Abstract guardrail engine interface."""
    
    @abstractmethod
    def validate_input(self, user_input: str) -> GuardrailResult:
        """Inspect and sanitize incoming user prompt."""
        pass

    @abstractmethod
    def validate_output(self, generated_text: str, context: Optional[str] = None) -> GuardrailResult:
        """Inspect and sanitize generated LLM response."""
        pass
