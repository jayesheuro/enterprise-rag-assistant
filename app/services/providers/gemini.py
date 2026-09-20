"""Gemini LLM and embedding provider using the google-genai SDK.

Uses the official google-genai SDK (not the deprecated google-generativeai).
API key is loaded from .env via the config system.

SDK pattern:
    client = genai.Client(api_key="...")
    response = client.models.generate_content(model="...", contents="...")
    embeddings = client.models.embed_content(model="...", contents=[...])
"""

from google import genai
from google.genai import types

from app.core.config import ProviderConfig, SecretsConfig
from app.core.exceptions import ProviderAuthError, ProviderError
from app.core.logging import get_logger
from app.services.providers.base import (
    BaseEmbedder,
    BaseLLM,
    LLMResponse,
    Message,
    TokenUsage,
)

logger = get_logger(__name__)


class GeminiLLM(BaseLLM):
    """Gemini LLM provider using google-genai SDK.

    Wraps client.models.generate_content() with our standardized interface.
    Token usage and model name are extracted from the response metadata.
    """

    def __init__(self, config: ProviderConfig, secrets: SecretsConfig) -> None:
        if not secrets.google_api_key:
            raise ProviderAuthError(provider="gemini")

        self._model = config.model
        self._temperature = config.temperature
        self._max_output_tokens = config.max_output_tokens
        self._client = genai.Client(api_key=secrets.google_api_key)
        logger.info("GeminiLLM initialized: model=%s", self._model)

    def _generate_impl(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Send messages to Gemini and return a standardized response.

        Converts our Message objects to Gemini's content format,
        handles system instructions separately, and extracts token usage.
        """
        try:
            # Separate system message from conversation
            system_instruction = None
            contents: list[types.Content] = []

            for msg in messages:
                if msg.role == "system":
                    system_instruction = msg.content
                else:
                    role = "user" if msg.role == "user" else "model"
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=msg.content)],
                        )
                    )

            # Build generation config
            temperature = kwargs.get("temperature", self._temperature)
            max_tokens = kwargs.get("max_output_tokens", self._max_output_tokens)

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
            )

            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            # Extract token usage from response metadata
            usage = TokenUsage()
            if response.usage_metadata:
                usage = TokenUsage(
                    prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                    completion_tokens=response.usage_metadata.candidates_token_count or 0,
                    total_tokens=response.usage_metadata.total_token_count or 0,
                )

            return LLMResponse(
                text=response.text or "",
                token_usage=usage,
                model_name=self._model,
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "unauthorized" in error_msg or "403" in error_msg:
                raise ProviderAuthError(provider="gemini") from e
            raise ProviderError(str(e), provider="gemini") from e


class GeminiEmbedder(BaseEmbedder):
    """Gemini embedding provider using google-genai SDK.

    Uses client.models.embed_content() which supports batching natively.
    The text-embedding-004 model produces 768-dimensional vectors.
    """

    # Known dimensions for Gemini embedding models
    _KNOWN_DIMENSIONS: dict[str, int] = {
        "text-embedding-004": 768,
        "embedding-001": 768,
        "gemini-embedding-001": 3072,
        "gemini-embedding-002": 3072,
    }

    def __init__(self, config: ProviderConfig, secrets: SecretsConfig) -> None:
        if not secrets.google_api_key:
            raise ProviderAuthError(provider="gemini")

        self._model = config.embedding_model
        self._client = genai.Client(api_key=secrets.google_api_key)
        self._dimension = self._KNOWN_DIMENSIONS.get(self._model, 768)
        logger.info(
            "GeminiEmbedder initialized: model=%s dim=%d",
            self._model,
            self._dimension,
        )

    @property
    def dimension(self) -> int:
        """Embedding dimension for the active model."""
        return self._dimension

    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Gemini's embed_content API.

        The API supports passing a list of strings directly, returning
        one embedding vector per input text.
        """
        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=texts,
            )

            # response.embeddings is a list of ContentEmbedding objects
            if response.embeddings:
                return [emb.values for emb in response.embeddings]
            return []

        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "unauthorized" in error_msg or "403" in error_msg:
                raise ProviderAuthError(provider="gemini") from e
            raise ProviderError(str(e), provider="gemini") from e

