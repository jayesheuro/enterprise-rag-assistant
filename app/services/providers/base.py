"""Abstract base classes for LLM and embedding providers.

This is the core of the provider abstraction (Strategy pattern). All concrete
providers (Gemini, Ollama, Bedrock) implement these ABCs, and the factory
returns the right one based on config. Switching providers = changing one line
in configs/app.yaml.

Key design decisions:
- Retry logic with exponential backoff lives HERE (base layer), not in each
  provider. Providers implement _generate_impl / _embed_impl.
- LLMResponse carries text, token usage, latency, and model name — everything
  needed for evaluation and logging.
- Message is a simple dataclass, not tied to any provider's format.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.exceptions import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Data models ─────────────────────────────────────────────────────────────


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: The speaker — "user", "assistant", or "system".
        content: The text content of the message.
    """

    role: str
    content: str


class TokenUsage(BaseModel):
    """Token counts for an LLM request."""

    prompt_tokens: int = Field(default=0, description="Tokens in the input")
    completion_tokens: int = Field(default=0, description="Tokens in the output")
    total_tokens: int = Field(default=0, description="Total tokens used")


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider.

    Every provider returns this same shape, making downstream code
    (evaluation, logging, API responses) provider-agnostic.
    """

    text: str = Field(description="The generated text response")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, description="Request duration in milliseconds")
    model_name: str = Field(default="unknown", description="Model that generated the response")


# ── Retry configuration ────────────────────────────────────────────────────

# Shared retry decorator for all provider calls.
# Retries on ProviderError (network issues, rate limits, transient failures).
# Does NOT retry on auth errors (those fail immediately).
_provider_retry = retry(
    retry=retry_if_exception_type(ProviderError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


# ── Abstract Base Classes ───────────────────────────────────────────────────


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    Subclasses implement _generate_impl with provider-specific API calls.
    The public generate() method adds retry logic, timing, and logging.
    """

    @abstractmethod
    def _generate_impl(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Provider-specific generation logic. Implemented by each provider.

        Args:
            messages: Conversation history as a list of Message objects.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated text and metadata.

        Raises:
            ProviderError: On API errors (will be retried).
            ProviderAuthError: On authentication failures (not retried).
        """
        ...

    @_provider_retry
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Generate a response from the LLM with retry and timing.

        Args:
            messages: Conversation history as a list of Message objects.
            **kwargs: Provider-specific overrides.

        Returns:
            LLMResponse with text, token usage, latency, and model name.
        """
        start = time.perf_counter()
        response = self._generate_impl(messages, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = elapsed_ms
        logger.info(
            "LLM response: model=%s tokens=%d latency=%.0fms",
            response.model_name,
            response.token_usage.total_tokens,
            elapsed_ms,
        )
        return response


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers.

    Subclasses implement _embed_impl for the actual embedding call.
    The public methods add retry logic and logging.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors produced by this model.

        This is critical metadata — you must never mix embeddings from
        models with different dimensions in the same vector store table.
        """
        ...

    @abstractmethod
    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Provider-specific embedding logic.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...

    @_provider_retry
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents with retry logic.

        Args:
            texts: Document texts to embed.

        Returns:
            List of embedding vectors.
        """
        start = time.perf_counter()
        embeddings = self._embed_impl(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Embedded %d documents: dim=%d latency=%.0fms",
            len(texts),
            self.dimension,
            elapsed_ms,
        )
        return embeddings

    @_provider_retry
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text with retry logic.

        Args:
            text: Query text to embed.

        Returns:
            Single embedding vector.
        """
        result = self._embed_impl([text])
        return result[0]

