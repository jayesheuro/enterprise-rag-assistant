"""Application configuration system.

Loads configuration from three layers:
1. .env file → secrets (API keys) via Pydantic Settings
2. configs/app.yaml → application tunables (chunking, retrieval, provider selection)
3. configs/providers/<name>.yaml → provider-specific settings (model names, URLs)

Usage:
    from app.core.config import get_settings
    settings = get_settings()
    print(settings.provider)          # "gemini"
    print(settings.chunking.chunk_size)  # 1000
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Path constants ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
PROVIDERS_DIR = CONFIGS_DIR / "providers"


# ── Secret settings (from .env) ────────────────────────────────────────────

class SecretsConfig(BaseSettings):
    """Secrets loaded exclusively from .env file. Never log these values."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = Field(
        default="",
        description="Google AI Studio API key for Gemini provider",
    )
    aws_access_key_id: str = Field(
        default="",
        description="AWS access key for Bedrock provider",
    )
    aws_secret_access_key: str = Field(
        default="",
        description="AWS secret key for Bedrock provider",
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock provider",
    )


# ── YAML-backed config models ──────────────────────────────────────────────

class ChunkingConfig(BaseModel):
    """Document chunking parameters."""

    chunk_size: int = Field(default=1000, description="Characters per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    strategy: str = Field(default="fixed", description="Chunking strategy: fixed | semantic")


class RetrievalConfig(BaseModel):
    """Vector search and retrieval parameters."""

    top_k: int = Field(default=5, description="Number of results to return")
    similarity_threshold: float = Field(
        default=0.7, description="Minimum similarity score (0.0-1.0)"
    )
    reranking_enabled: bool = Field(default=False, description="Apply reranking to results")


class ProviderConfig(BaseModel):
    """Provider-specific settings loaded from configs/providers/<name>.yaml."""

    model: str = Field(description="Chat/generation model identifier")
    embedding_model: str = Field(description="Embedding model identifier")
    temperature: float = Field(default=0.7, description="Generation temperature")
    max_output_tokens: int = Field(default=2048, description="Max tokens in response")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    # Ollama-specific
    base_url: str | None = Field(default=None, description="Base URL for local providers")
    # Bedrock-specific
    region: str | None = Field(default=None, description="AWS region for Bedrock")


# ── Main application config ────────────────────────────────────────────────

class AppConfig(BaseModel):
    """Top-level application configuration composing all sub-configs."""

    provider: str = Field(default="gemini", description="Active provider: gemini|ollama|bedrock")
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    provider_config: ProviderConfig | None = Field(
        default=None, description="Loaded from configs/providers/<provider>.yaml"
    )
    secrets: SecretsConfig = Field(
        default_factory=SecretsConfig, description="Secrets loaded from .env"
    )


# ── Loaders ─────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_app_config() -> AppConfig:
    """Load the full application configuration from YAML files and .env.

    Reads configs/app.yaml for tunables, then loads the matching
    configs/providers/<provider>.yaml for provider-specific settings,
    and merges in secrets from .env.
    """
    # 1. Load app.yaml
    app_yaml = _load_yaml(CONFIGS_DIR / "app.yaml")

    # 2. Determine active provider
    provider_name = app_yaml.get("provider", "gemini")

    # 3. Load provider-specific YAML
    provider_yaml_path = PROVIDERS_DIR / f"{provider_name}.yaml"
    provider_data = _load_yaml(provider_yaml_path)
    provider_config = ProviderConfig(**provider_data)

    # 4. Load secrets from .env
    secrets = SecretsConfig()

    # 5. Build chunking and retrieval configs
    chunking = ChunkingConfig(**app_yaml.get("chunking", {}))
    retrieval = RetrievalConfig(**app_yaml.get("retrieval", {}))

    return AppConfig(
        provider=provider_name,
        chunking=chunking,
        retrieval=retrieval,
        provider_config=provider_config,
        secrets=secrets,
    )


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Get cached application settings. Call once at startup."""
    return load_app_config()

