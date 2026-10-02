"""Guardrails for RAG input and output."""

import re
from app.core.logging import get_logger

logger = get_logger(__name__)

class InputGuardrails:
    @staticmethod
    def check_query(query: str, max_length: int) -> tuple[str, list[str]]:
        warnings = []
        sanitized_query = query.strip()
        
        # Check length
        if len(sanitized_query) > max_length:
            warnings.append(f"Query exceeded max length of {max_length} characters. Truncated.")
            sanitized_query = sanitized_query[:max_length]
            
        # Check prompt injection
        injection_patterns = [
            r"ignore previous",
            r"forget your instructions",
            r"you are now",
            r"system prompt",
            r"disregard",
        ]
        
        lower_query = sanitized_query.lower()
        for pattern in injection_patterns:
            if re.search(pattern, lower_query):
                msg = f"Potential prompt injection detected: matched pattern '{pattern}'"
                warnings.append(msg)
                logger.warning(msg)
                
        return sanitized_query, warnings


class OutputGuardrails:
    @staticmethod
    def check_response(answer: str, grounded: bool) -> dict:
        result = {
            "low_confidence": False,
            "warnings": []
        }
        
        if grounded:
            # Check for citations [1], [2], etc.
            if not re.search(r"\[\d+\]", answer):
                result["low_confidence"] = True
                result["warnings"].append("Response is marked as grounded but contains no citations.")
                
        return result
