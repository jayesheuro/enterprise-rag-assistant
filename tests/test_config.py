"""Tests for the configuration system.

Verifies that configs load correctly from YAML files and .env,
validates defaults, and catches invalid configurations.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from app.core.config import (
    CONFIGS_DIR,
    ChunkingConfig,
    ProviderConfig,
    RetrievalConfig,
    SecretsConfig,
    load_app_config,
)


class TestChunkingConfig:
    """Tests for ChunkingConfig defaults and validation."""

    def test_defaults(self):
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.strategy == "recursive"

    def test_custom_values(self):
        config = ChunkingConfig(chunk_size=500, chunk_overlap=100, strategy="markdown_aware")
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.strategy == "markdown_aware"


class TestRetrievalConfig:
    """Tests for RetrievalConfig defaults and validation."""

    def test_defaults(self):
        config = RetrievalConfig()
        assert config.top_k == 12
        assert config.final_k == 4
        assert config.similarity_threshold == 0.01
        assert config.mode == "hybrid"
        assert config.reranker == "none"

    def test_custom_values(self):
        config = RetrievalConfig(top_k=10, similarity_threshold=0.5, reranker="llm", mode="vector")
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.reranker == "llm"
        assert config.mode == "vector"


class TestProviderConfig:
    """Tests for ProviderConfig parsing."""

    def test_gemini_config(self):
        config = ProviderConfig(
            model="gemini-3.8-flash",
            embedding_model="gemini-embedding-001",
            temperature=0.7,
            max_output_tokens=2048,
        )
        assert config.model == "gemini-3.8-flash"
        assert config.base_url is None  # Not set for cloud providers

    def test_ollama_config(self):
        config = ProviderConfig(
            model="llama3.2",
            embedding_model="nomic-embed-text",
            base_url="http://localhost:11434",
        )
        assert config.base_url == "http://localhost:11434"


class TestSecretsConfig:
    """Tests for secrets loading (without real .env)."""

    def test_defaults_when_no_env(self, monkeypatch):
        """Secrets should default to empty strings when env vars are absent."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        # Patch the env_file to a nonexistent path so .env is not loaded
        config = SecretsConfig(_env_file="nonexistent.env")
        assert config.google_api_key == ""
        assert config.aws_access_key_id == ""

    def test_loads_from_env_vars(self, monkeypatch):
        """Secrets can be loaded from environment variables."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
        config = SecretsConfig()
        assert config.google_api_key == "test-key-123"


class TestLoadAppConfig:
    """Integration tests for the full config loading pipeline."""

    def test_loads_from_real_yaml_files(self):
        """Verify config loads from the actual YAML files in the repo."""
        config = load_app_config()
        assert config.provider in ("gemini", "ollama", "bedrock")
        assert config.chunking.chunk_size > 0
        assert config.provider_config is not None
        assert config.provider_config.model != ""

    def test_provider_config_matches_provider_name(self):
        """The loaded provider config should match the provider in app.yaml."""
        config = load_app_config()
        provider_yaml_path = CONFIGS_DIR / "providers" / f"{config.provider}.yaml"
        assert provider_yaml_path.exists()

        with open(provider_yaml_path) as f:
            expected = yaml.safe_load(f)

        assert config.provider_config is not None
        assert config.provider_config.model == expected["model"]

    def test_missing_provider_yaml_raises(self, tmp_path):
        """Should raise FileNotFoundError for unknown provider."""
        # Create a minimal app.yaml pointing to a nonexistent provider
        app_yaml = tmp_path / "app.yaml"
        app_yaml.write_text("provider: nonexistent\n")

        providers_dir = tmp_path / "providers"
        providers_dir.mkdir()

        with patch("app.core.config.CONFIGS_DIR", tmp_path), \
             patch("app.core.config.PROVIDERS_DIR", providers_dir):
            with pytest.raises(FileNotFoundError):
                load_app_config()

