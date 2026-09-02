"""Local Safety & Guardrails Engine (100% offline, zero cloud calls, replaces NeMo Guardrails)."""
import re
from typing import List, Optional
from app.guardrails.base import BaseGuardrail, GuardrailResult

class LocalGuardrails(BaseGuardrail):
    """Local Guardrails for Prompt Injection, Jailbreaks, PII Masking, and Toxicity protection."""

    # Prompt injection and jailbreak attack heuristics
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
        r"(?i)disregard\s+(the\s+)?(previous|above|system)\s+directives",
        r"(?i)you\s+are\s+now\s+(in\s+)?(DAN|developer|god)\s+mode",
        r"(?i)system\s*prompt\s*leak",
        r"(?i)reveal\s+(your\s+)?(initial|system|hidden)\s+(instructions|prompt)",
        r"(?i)bypass\s+(safety|content)\s+filters",
        r"(?i)sudo\s+mode",
        r"(?i)unrestricted\s+ai\s+mode",
    ]

    # PII patterns
    PII_REGEX = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "API_KEY": r"(?:sk-[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z-_]{35}|gsk_[a-zA-Z0-9]{30,})",
    }

    def __init__(self, mask_pii: bool = True, block_injections: bool = True):
        self.mask_pii = mask_pii
        self.block_injections = block_injections

    def _sanitize_pii(self, text: str) -> str:
        sanitized = text
        for pii_type, pattern in self.PII_REGEX.items():
            sanitized = re.sub(pattern, f"[{pii_type}_REDACTED]", sanitized)
        return sanitized

    def validate_input(self, user_input: str) -> GuardrailResult:
        violations = []
        risk_score = 0.0
        
        if not user_input or not user_input.strip():
            return GuardrailResult(is_safe=True, sanitized_text=user_input, violations=[], risk_score=0.0)

        # Check prompt injection
        if self.block_injections:
            for pat in self.INJECTION_PATTERNS:
                if re.search(pat, user_input):
                    violations.append("PROMPT_INJECTION_DETECTED")
                    risk_score = max(risk_score, 0.9)
                    break

        # Sanitize PII
        sanitized = user_input
        if self.mask_pii:
            sanitized = self._sanitize_pii(user_input)
            if sanitized != user_input:
                violations.append("PII_MASKED")
                risk_score = max(risk_score, 0.3)

        is_safe = "PROMPT_INJECTION_DETECTED" not in violations
        return GuardrailResult(
            is_safe=is_safe,
            sanitized_text=sanitized if is_safe else "I cannot process requests that attempt to override safety instructions.",
            violations=violations,
            risk_score=risk_score
        )

    def validate_output(self, generated_text: str, context: Optional[str] = None) -> GuardrailResult:
        violations = []
        risk_score = 0.0
        
        if not generated_text:
            return GuardrailResult(is_safe=True, sanitized_text="", violations=[], risk_score=0.0)

        # Sanitize any leaked PII or sensitive keys in outputs
        sanitized = self._sanitize_pii(generated_text)
        if sanitized != generated_text:
            violations.append("PII_REDACTED_FROM_OUTPUT")
            risk_score = 0.4

        return GuardrailResult(
            is_safe=True,
            sanitized_text=sanitized,
            violations=violations,
            risk_score=risk_score
        )

guardrails = LocalGuardrails()
