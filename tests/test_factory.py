"""Tests for the provider factory.

Verifies that the factory creates the correct provider class for each
provider name, and raises on unknown providers. Uses monkeypatching to
avoid real API calls — no network needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import AppConfig, ProviderConfig, SecretsConfig
from app.core.exceptions import ConfigError
from app.services.providers.factory import create_embedder, create_llm


def _make_config(provider: str = "gemini") -> AppConfig:
    """Create a test AppConfig with the given provider."""
    return AppConfig(
        provider=provider,
        provider_config=ProviderConfig(
            model="test-model",
            embedding_model="test-embed-model",
            temperature=0.5,
            max_output_tokens=1024,
            timeout=10,
            base_url="http://localhost:11434" if provider == "ollama" else None,
            region="us-east-1" if provider == "bedrock" else None,
        ),
        secrets=SecretsConfig(
            google_api_key="fake-key" if provider == "gemini" else "",
            aws_access_key_id="fake-aws-key" if provider == "bedrock" else "",
            aws_secret_access_key="fake-aws-secret" if provider == "bedrock" else "",
        ),
    )


class TestCreateLLM:
    """Tests for create_llm factory function."""

    def test_creates_gemini_llm(self):
        """Factory should create GeminiLLM when provider=gemini."""
        config = _make_config("gemini")
        with patch("app.services.providers.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            llm = create_llm(config)
        assert llm.__class__.__name__ == "GeminiLLM"

    def test_creates_ollama_llm(self):
        """Factory should create OllamaLLM when provider=ollama."""
        config = _make_config("ollama")
        llm = create_llm(config)
        assert llm.__class__.__name__ == "OllamaLLM"

    def test_creates_bedrock_llm(self):
        """Factory should create BedrockLLM when provider=bedrock."""
        config = _make_config("bedrock")
        with patch("app.services.providers.bedrock._get_boto3_client") as mock_boto:
            mock_boto.return_value = MagicMock()
            llm = create_llm(config)
        assert llm.__class__.__name__ == "BedrockLLM"

    def test_unknown_provider_raises(self):
        """Factory should raise ConfigError for unknown providers."""
        config = _make_config("gemini")
        config.provider = "unknown_provider"
        with pytest.raises(ConfigError, match="Unknown LLM provider"):
            create_llm(config)


class TestCreateEmbedder:
    """Tests for create_embedder factory function."""

    def test_creates_gemini_embedder(self):
        """Factory should create GeminiEmbedder when provider=gemini."""
        config = _make_config("gemini")
        with patch("app.services.providers.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            embedder = create_embedder(config)
        assert embedder.__class__.__name__ == "GeminiEmbedder"

    def test_creates_ollama_embedder(self):
        """Factory should create OllamaEmbedder when provider=ollama."""
        config = _make_config("ollama")
        embedder = create_embedder(config)
        assert embedder.__class__.__name__ == "OllamaEmbedder"

    def test_creates_bedrock_embedder(self):
        """Factory should create BedrockEmbedder when provider=bedrock."""
        config = _make_config("bedrock")
        with patch("app.services.providers.bedrock._get_boto3_client") as mock_boto:
            mock_boto.return_value = MagicMock()
            embedder = create_embedder(config)
        assert embedder.__class__.__name__ == "BedrockEmbedder"

    def test_unknown_provider_raises(self):
        """Factory should raise ConfigError for unknown providers."""
        config = _make_config("gemini")
        config.provider = "nonexistent"
        with pytest.raises(ConfigError, match="Unknown embedder provider"):
            create_embedder(config)


class TestFakesWork:
    """Verify our test doubles work correctly with the ABCs."""

    def test_fake_llm(self):
        from tests.fakes import FakeLLM
        from app.services.providers.base import Message

        llm = FakeLLM(response_text="Hello!")
        response = llm.generate([Message(role="user", content="Hi")])
        assert response.text == "Hello!"
        assert response.token_usage.total_tokens == 15
        assert llm.call_count == 1
        assert llm.last_messages[0].content == "Hi"

    def test_fake_embedder(self):
        from tests.fakes import FakeEmbedder

        embedder = FakeEmbedder(dim=384)
        assert embedder.dimension == 384

        vectors = embedder.embed_documents(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        assert vectors[0] != vectors[1]  # Different texts → different vectors

        single = embedder.embed_query("test")
        assert len(single) == 384

