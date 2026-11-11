"""Domain-specific typed exceptions for job_ai_app."""

class JobAIException(Exception):
    """Base exception for all domain errors."""
    pass

class ScrapingError(JobAIException):
    """Raised when job board scraping encounters an unrecoverable failure."""
    pass

class ScrapingRateLimitError(ScrapingError):
    """Raised when rate-limited or blocked (429/403) by anti-bot measures."""
    pass

class LLMProviderError(JobAIException):
    """Raised when an LLM provider fails."""
    pass

class LLMQuotaError(LLMProviderError):
    """Raised when API quota or rate limits are exceeded."""
    pass

class FactAuditingError(JobAIException):
    """Raised when hallucinated facts are detected during validation."""
    pass

class VectorSearchError(JobAIException):
    """Raised when vector indexing or similarity search fails."""
    pass
