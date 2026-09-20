"""Integration tests for live provider APIs.

These tests make REAL API calls and are skipped when credentials
are not available. Run them with:
    uv run pytest tests/test_providers_integration.py -v -m integration

Skip control:
- Gemini: skipped if GOOGLE_API_KEY is not set in .env
- Ollama: skipped if Ollama is not running at localhost:11434
"""

import os

import httpx
import pytest

from app.core.config import ProviderConfig, SecretsConfig
from app.services.providers.base import Message

# ── Skip conditions ─────────────────────────────────────────────────────────

_has_gemini_key = bool(os.environ.get("GOOGLE_API_KEY", ""))


def _ollama_is_running() -> bool:
    """Check if Ollama is running locally."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


_has_ollama = _ollama_is_running()

skip_no_gemini = pytest.mark.skipif(
    not _has_gemini_key,
    reason="GOOGLE_API_KEY not set — skipping Gemini integration test",
)

skip_no_ollama = pytest.mark.skipif(
    not _has_ollama,
    reason="Ollama not running at localhost:11434 — skipping Ollama integration test",
)


# ── Gemini tests ────────────────────────────────────────────────────────────


@pytest.mark.integration
@skip_no_gemini
class TestGeminiIntegration:
    """Live integration tests for the Gemini provider."""

    def _make_config(self):
        return (
            ProviderConfig(
                model="gemini-3.8-flash",
                embedding_model="text-embedding-004",
                temperature=0.1,
                max_output_tokens=100,
                timeout=30,
            ),
            SecretsConfig(),
        )

    def test_gemini_llm_generate(self):
        """GeminiLLM should generate a response for a simple prompt."""
        from app.services.providers.gemini import GeminiLLM

        config, secrets = self._make_config()
        llm = GeminiLLM(config=config, secrets=secrets)

        response = llm.generate([Message(role="user", content="Say OK")])

        assert response.text.strip() != ""
        assert response.model_name == "gemini-3.8-flash"
        assert response.latency_ms > 0
        assert response.token_usage.total_tokens > 0

    def test_gemini_embedder(self):
        """GeminiEmbedder should produce a vector of the expected dimension."""
        from app.services.providers.gemini import GeminiEmbedder

        config, secrets = self._make_config()
        embedder = GeminiEmbedder(config=config, secrets=secrets)

        vector = embedder.embed_query("Hello world")

        assert len(vector) == embedder.dimension
        assert all(isinstance(v, float) for v in vector)

    def test_gemini_embedder_batch(self):
        """GeminiEmbedder should handle batched embeddings."""
        from app.services.providers.gemini import GeminiEmbedder

        config, secrets = self._make_config()
        embedder = GeminiEmbedder(config=config, secrets=secrets)

        vectors = embedder.embed_documents(["Hello", "World", "Test"])

        assert len(vectors) == 3
        assert all(len(v) == embedder.dimension for v in vectors)


# ── Ollama tests ────────────────────────────────────────────────────────────


@pytest.mark.integration
@skip_no_ollama
class TestOllamaIntegration:
    """Live integration tests for the Ollama provider."""

    def _make_config(self):
        return ProviderConfig(
            model="llama3.2",
            embedding_model="nomic-embed-text",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_output_tokens=100,
            timeout=120,
        )

    def test_ollama_llm_generate(self):
        """OllamaLLM should generate a response from a local model."""
        from app.services.providers.ollama import OllamaLLM

        config = self._make_config()
        llm = OllamaLLM(config=config)

        response = llm.generate([Message(role="user", content="Say OK")])

        assert response.text.strip() != ""
        assert response.latency_ms > 0

    def test_ollama_embedder(self):
        """OllamaEmbedder should produce embedding vectors."""
        import httpx
        try:
            resp = httpx.post("http://localhost:11434/api/show", json={"model": "nomic-embed-text"})
            if resp.status_code != 200:
                pytest.skip("nomic-embed-text model not pulled")
        except Exception:
            pytest.skip("Ollama not running")

        from app.services.providers.ollama import OllamaEmbedder

        config = self._make_config()
        embedder = OllamaEmbedder(config=config)

        vector = embedder.embed_query("Hello world")

        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

