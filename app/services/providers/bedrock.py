"""AWS Bedrock LLM and embedding provider (UNTESTED STUB).

This provider is fully implemented but has NOT been tested against a live
Bedrock endpoint. It exists to demonstrate:
1. The provider abstraction works with a third cloud provider
2. Lazy imports — boto3 is only imported when this provider is instantiated,
   so it's not a required dependency for Gemini/Ollama users

To use: pip install boto3, set AWS credentials in .env, and change
configs/app.yaml to provider: bedrock
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import boto3  # noqa: F811

logger = get_logger(__name__)


def _get_boto3_client(service: str, config: ProviderConfig, secrets: SecretsConfig):
    """Lazily import boto3 and create a client.

    This ensures boto3 is only required when Bedrock is actually used,
    not when the module is imported.
    """
    try:
        import boto3
    except ImportError as e:
        raise ProviderError(
            "boto3 is required for the Bedrock provider. "
            "Install it with: pip install boto3",
            provider="bedrock",
        ) from e

    region = config.region or secrets.aws_region or "us-east-1"

    kwargs: dict = {"region_name": region}
    if secrets.aws_access_key_id:
        kwargs["aws_access_key_id"] = secrets.aws_access_key_id
        kwargs["aws_secret_access_key"] = secrets.aws_secret_access_key

    return boto3.client(service, **kwargs)


class BedrockLLM(BaseLLM):
    """AWS Bedrock LLM provider (UNTESTED STUB).

    Uses the Bedrock Runtime invoke_model API with Claude's message format.
    Supports Claude 3 models via the Messages API.
    """

    def __init__(self, config: ProviderConfig, secrets: SecretsConfig) -> None:
        if not secrets.aws_access_key_id and not self._has_default_credentials():
            logger.warning("No AWS credentials found. Bedrock calls may fail.")

        self._model = config.model
        self._temperature = config.temperature
        self._max_output_tokens = config.max_output_tokens
        self._client = _get_boto3_client("bedrock-runtime", config, secrets)
        logger.info("BedrockLLM initialized (UNTESTED): model=%s", self._model)

    @staticmethod
    def _has_default_credentials() -> bool:
        """Check if default AWS credentials are available (e.g., IAM role)."""
        try:
            import boto3

            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials is not None
        except Exception:
            return False

    def _generate_impl(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Send messages to Bedrock's invoke_model API (Claude format)."""
        try:
            # Convert to Claude Messages API format
            system_text = None
            api_messages = []
            for msg in messages:
                if msg.role == "system":
                    system_text = msg.content
                else:
                    api_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": kwargs.get("max_output_tokens", self._max_output_tokens),
                "temperature": kwargs.get("temperature", self._temperature),
                "messages": api_messages,
            }
            if system_text:
                body["system"] = system_text

            response = self._client.invoke_model(
                modelId=self._model,
                body=json.dumps(body),
                contentType="application/json",
            )

            result = json.loads(response["body"].read())

            text = ""
            if result.get("content"):
                text = result["content"][0].get("text", "")

            usage_data = result.get("usage", {})
            usage = TokenUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=(
                    usage_data.get("input_tokens", 0)
                    + usage_data.get("output_tokens", 0)
                ),
            )

            return LLMResponse(
                text=text,
                token_usage=usage,
                model_name=self._model,
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "credential" in error_msg or "access denied" in error_msg:
                raise ProviderAuthError(provider="bedrock") from e
            raise ProviderError(str(e), provider="bedrock") from e


class BedrockEmbedder(BaseEmbedder):
    """AWS Bedrock embedding provider (UNTESTED STUB).

    Uses Titan Embeddings V2 via invoke_model. Processes texts one at a time
    since Bedrock's Titan embed API doesn't support batching natively.
    """

    _KNOWN_DIMENSIONS: dict[str, int] = {
        "amazon.titan-embed-text-v2:0": 1024,
        "amazon.titan-embed-text-v1": 1536,
    }

    def __init__(self, config: ProviderConfig, secrets: SecretsConfig) -> None:
        self._model = config.embedding_model
        self._client = _get_boto3_client("bedrock-runtime", config, secrets)
        self._dimension = self._KNOWN_DIMENSIONS.get(self._model, 1024)
        logger.info(
            "BedrockEmbedder initialized (UNTESTED): model=%s dim=%d",
            self._model,
            self._dimension,
        )

    @property
    def dimension(self) -> int:
        """Embedding dimension for the active model."""
        return self._dimension

    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Bedrock's Titan Embeddings API.

        Titan doesn't support batch embedding, so we call invoke_model
        once per text. In production, you'd want to parallelize this.
        """
        try:
            embeddings = []
            for text in texts:
                body = {
                    "inputText": text,
                }
                response = self._client.invoke_model(
                    modelId=self._model,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                result = json.loads(response["body"].read())
                embedding = result.get("embedding", [])
                embeddings.append(embedding)

                # Auto-detect dimension
                if embedding and len(embedding) != self._dimension:
                    self._dimension = len(embedding)

            return embeddings

        except Exception as e:
            raise ProviderError(str(e), provider="bedrock") from e

