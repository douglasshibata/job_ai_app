"""Backward compatibility layer for llm_client."""
from services.llm_provider import (
    UnifiedLLMGateway as UnifiedLLMClient,
    GroqProvider,
    GeminiProvider,
    OllamaProvider,
    extract_json_from_text
)

__all__ = [
    "UnifiedLLMClient",
    "GroqProvider",
    "GeminiProvider",
    "OllamaProvider",
    "extract_json_from_text"
]
