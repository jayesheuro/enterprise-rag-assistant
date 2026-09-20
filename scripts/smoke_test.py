"""Smoke test CLI — verifies provider setup end-to-end.

Usage:
    uv run python scripts/smoke_test.py

What it does:
    1. Loads config from configs/app.yaml + .env
    2. Creates LLM and embedder via the factory
    3. Sends "Say OK" to the LLM
    4. Embeds a single sentence
    5. Prints model name, latency, token usage, embedding dimension

If this script passes, your provider setup is working correctly.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_app_config
from app.core.logging import setup_logging
from app.services.providers.factory import create_embedder, create_llm
from app.services.providers.base import Message


def main() -> None:
    """Run the smoke test."""
    setup_logging(level="INFO")

    print("=" * 60)
    print("Enterprise RAG Assistant — Smoke Test")
    print("=" * 60)

    # 1. Load config
    config = load_app_config()
    print(f"\n✓ Config loaded: provider={config.provider}")
    assert config.provider_config is not None, "Provider config failed to load"
    print(f"  Model: {config.provider_config.model}")
    print(f"  Embedding model: {config.provider_config.embedding_model}")

    # 2. Create LLM via factory
    print("\n--- LLM Test ---")
    llm = create_llm(config)
    print(f"✓ LLM created: {llm.__class__.__name__}")

    # 3. Send a simple prompt
    messages = [Message(role="user", content="Say OK")]
    response = llm.generate(messages)
    print(f"✓ LLM response: \"{response.text.strip()}\"")
    print(f"  Model: {response.model_name}")
    print(f"  Latency: {response.latency_ms:.0f}ms")
    print(f"  Tokens: {response.token_usage.prompt_tokens} prompt "
          f"+ {response.token_usage.completion_tokens} completion "
          f"= {response.token_usage.total_tokens} total")

    # 4. Create embedder via factory
    print("\n--- Embedding Test ---")
    embedder = create_embedder(config)
    print(f"✓ Embedder created: {embedder.__class__.__name__}")

    # 5. Embed a sentence
    test_text = "The quick brown fox jumps over the lazy dog."
    embedding = embedder.embed_query(test_text)
    print(f"✓ Embedding generated for: \"{test_text}\"")
    print(f"  Dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Embedder reports dimension: {embedder.dimension}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("✓ ALL SMOKE TESTS PASSED")
    print(f"  Provider: {config.provider}")
    print(f"  LLM model: {response.model_name}")
    print(f"  Embedding dimension: {len(embedding)}")
    print("=" * 60)

    # Reminder for free tier users
    print("\n💡 Reminder: You're on Gemini free tier. Monitor your usage at:")
    print("   https://aistudio.google.com/apikey")


if __name__ == "__main__":
    main()

