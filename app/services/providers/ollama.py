"""Ollama LLM and embedding provider using its local HTTP API.

Ollama exposes a REST API at http://localhost:11434 for running models locally.
No API key needed — just have Ollama running. Great for development without
burning cloud API credits.

API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
- POST /api/chat → chat completions
- POST /api/embed → text embeddings
"""

import httpx

from app.core.config import ProviderConfig
from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.services.providers.base import (
    BaseEmbedder,
    BaseLLM,
    LLMResponse,
    Message,
    TokenUsage,
)

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama LLM provider using its local HTTP API.

    Calls POST /api/chat with the conversation messages.
    Ollama returns token counts in the response metadata.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._model = config.model
        self._base_url = config.base_url or "http://localhost:11434"
        self._temperature = config.temperature
        self._max_output_tokens = config.max_output_tokens
        self._timeout = config.timeout
        logger.info("OllamaLLM initialized: model=%s url=%s", self._model, self._base_url)

    def _generate_impl(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Send messages to Ollama's /api/chat endpoint."""
        try:
            payload = {
                "model": kwargs.get("model", self._model),
                "messages": [
                    {"role": msg.role, "content": msg.content} for msg in messages
                ],
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self._temperature),
                    "num_predict": kwargs.get("max_output_tokens", self._max_output_tokens),
                },
            }

            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

            # Extract token usage from Ollama response
            usage = TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=(
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            )

            return LLMResponse(
                text=data.get("message", {}).get("content", ""),
                token_usage=usage,
                model_name=data.get("model", self._model),
            )

        except httpx.ConnectError as e:
            raise ProviderError(
                "Cannot connect to Ollama. Is it running? Start with: ollama serve",
                provider="ollama",
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Ollama API error: {e.response.status_code}",
                provider="ollama",
            ) from e
        except Exception as e:
            raise ProviderError(str(e), provider="ollama") from e


class OllamaEmbedder(BaseEmbedder):
    """Ollama embedding provider using its local HTTP API.

    Calls POST /api/embed which accepts multiple texts at once.
    nomic-embed-text produces 768-dimensional vectors by default.
    """

    # Known dimensions for common Ollama embedding models
    _KNOWN_DIMENSIONS: dict[str, int] = {
        "nomic-embed-text": 768,
        "all-minilm": 384,
        "mxbai-embed-large": 1024,
    }

    def __init__(self, config: ProviderConfig) -> None:
        self._model = config.embedding_model
        self._base_url = config.base_url or "http://localhost:11434"
        self._timeout = config.timeout
        self._dimension = self._KNOWN_DIMENSIONS.get(self._model, 768)
        logger.info(
            "OllamaEmbedder initialized: model=%s dim=%d url=%s",
            self._model,
            self._dimension,
            self._base_url,
        )

    @property
    def dimension(self) -> int:
        """Embedding dimension for the active model."""
        return self._dimension

    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Ollama's /api/embed endpoint."""
        try:
            payload = {
                "model": self._model,
                "input": texts,
            }

            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()

            embeddings = data.get("embeddings", [])

            # Auto-detect dimension from first response if not known
            if embeddings and len(embeddings[0]) != self._dimension:
                self._dimension = len(embeddings[0])
                logger.info("Auto-detected embedding dimension: %d", self._dimension)

            return embeddings

        except httpx.ConnectError as e:
            raise ProviderError(
                "Cannot connect to Ollama. Is it running? Start with: ollama serve",
                provider="ollama",
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Ollama API error: {e.response.status_code}",
                provider="ollama",
            ) from e
        except Exception as e:
            raise ProviderError(str(e), provider="ollama") from e

