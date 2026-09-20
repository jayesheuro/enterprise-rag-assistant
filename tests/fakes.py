"""Test fakes — deterministic test doubles for LLM and embedding providers.

These fakes implement the same ABCs as real providers but return canned
responses with no network calls. Use them in unit tests to verify business
logic without depending on external APIs.
"""

from app.services.providers.base import (
    BaseEmbedder,
    BaseLLM,
    LLMResponse,
    Message,
    TokenUsage,
)


class FakeLLM(BaseLLM):
    """Fake LLM that returns a configurable canned response.

    Args:
        response_text: The text to return from generate(). Defaults to "Fake response".
        model_name: Model name to include in response. Defaults to "fake-model".
    """

    def __init__(
        self,
        response_text: str = "Fake response",
        model_name: str = "fake-model",
    ) -> None:
        self._response_text = response_text
        self._model_name = model_name
        self.last_messages: list[Message] | None = None
        self.call_count: int = 0

    def _generate_impl(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Return a canned response and record the call for assertions."""
        self.last_messages = messages
        self.call_count += 1
        return LLMResponse(
            text=self._response_text,
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            model_name=self._model_name,
        )


class FakeEmbedder(BaseEmbedder):
    """Fake embedder that returns deterministic embedding vectors.

    Returns vectors of the specified dimension where each value is
    a simple function of the input text (for determinism in tests).

    Args:
        dim: Embedding dimension. Defaults to 768.
    """

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim
        self.call_count: int = 0
        self.last_texts: list[str] | None = None

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dim

    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings based on text length.

        Each embedding is a vector where all values are derived from
        the hash of the input text, ensuring different texts produce
        different (but deterministic) embeddings.
        """
        self.last_texts = texts
        self.call_count += 1
        embeddings = []
        for text in texts:
            # Use hash for deterministic but varied values
            h = hash(text) % 10000
            base_val = h / 10000.0
            embedding = [base_val + (i * 0.0001) for i in range(self._dim)]
            embeddings.append(embedding)
        return embeddings

