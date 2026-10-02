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


class IngestionError(AppError):
    """General error during document ingestion."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message, status_code=status_code)


class UnsupportedFileTypeError(IngestionError):
    """Raised when an unsupported file type is ingested."""

    def __init__(self, extension: str) -> None:
        super().__init__(
            f"Unsupported file extension: '{extension}'. "
            f"Supported types: .txt, .md, .pdf",
            status_code=400,
        )


class EmbeddingModelMismatchError(IngestionError):
    """Raised when appending to a vector store with a different embedding model.

    Mixed embeddings in one table produce meaningless similarity scores because
    different models map text to incompatible vector spaces.
    """

    def __init__(self, current_model: str, new_model: str) -> None:
        super().__init__(
            f"Embedding model mismatch: table uses '{current_model}', "
            f"but '{new_model}' was requested. You must either re-ingest with "
            f"--reset to switch models, or keep using '{current_model}'.",
            status_code=409,
        )

