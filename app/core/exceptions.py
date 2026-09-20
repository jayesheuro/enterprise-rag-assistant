"""Custom exception hierarchy for the Enterprise RAG Assistant.

All application exceptions inherit from AppError. FastAPI exception handlers
in app/main.py catch these and return structured JSON error responses.
"""


class AppError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ConfigError(AppError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class ProviderError(AppError):
    """Raised when an LLM/embedding provider call fails."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}", status_code=502)


class ProviderTimeoutError(ProviderError):
    """Raised when a provider call times out after retries."""

    def __init__(self, provider: str, timeout_seconds: int) -> None:
        super().__init__(
            f"Request timed out after {timeout_seconds}s",
            provider=provider,
        )


class ProviderAuthError(ProviderError):
    """Raised when provider authentication fails (missing/invalid API key)."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            "Authentication failed. Check your API key in .env",
            provider=provider,
        )

