"""Provider factory — the single entry point for creating LLM and embedder instances.

This implements the Factory pattern. The rest of the application never imports
concrete providers directly; it calls create_llm() or create_embedder() and
gets back the right implementation based on configs/app.yaml.

Switching providers:
    1. Edit configs/app.yaml → provider: ollama
    2. Restart the app
    That's it. Zero code changes.
"""

from app.core.config import AppConfig, load_app_config
from app.core.exceptions import ConfigError
from app.core.logging import get_logger
from app.services.providers.base import BaseEmbedder, BaseLLM

logger = get_logger(__name__)

# Registry of provider names to their module paths and class names.
# This avoids importing all providers at module level — each is imported
# only when selected, keeping startup fast and dependencies minimal.
_LLM_REGISTRY: dict[str, tuple[str, str]] = {
    "gemini": ("app.services.providers.gemini", "GeminiLLM"),
    "ollama": ("app.services.providers.ollama", "OllamaLLM"),
    "bedrock": ("app.services.providers.bedrock", "BedrockLLM"),
}

_EMBEDDER_REGISTRY: dict[str, tuple[str, str]] = {
    "gemini": ("app.services.providers.gemini", "GeminiEmbedder"),
    "ollama": ("app.services.providers.ollama", "OllamaEmbedder"),
    "bedrock": ("app.services.providers.bedrock", "BedrockEmbedder"),
}


def _import_class(module_path: str, class_name: str) -> type:
    """Dynamically import a class from a module path."""
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_llm(config: AppConfig | None = None) -> BaseLLM:
    """Create an LLM instance based on the active provider in config.

    Args:
        config: Application config. If None, loads from configs/app.yaml.

    Returns:
        A BaseLLM implementation ready to use.

    Raises:
        ConfigError: If the provider name is unknown.
    """
    if config is None:
        config = load_app_config()

    provider = config.provider
    if provider not in _LLM_REGISTRY:
        raise ConfigError(
            f"Unknown LLM provider: '{provider}'. "
            f"Valid options: {', '.join(_LLM_REGISTRY.keys())}"
        )

    module_path, class_name = _LLM_REGISTRY[provider]
    llm_class = _import_class(module_path, class_name)

    logger.info("Creating LLM: provider=%s class=%s", provider, class_name)

    # Ollama doesn't need secrets; Gemini and Bedrock do
    if provider == "ollama":
        return llm_class(config=config.provider_config)
    else:
        return llm_class(config=config.provider_config, secrets=config.secrets)


def create_embedder(config: AppConfig | None = None) -> BaseEmbedder:
    """Create an embedder instance based on the active provider in config.

    Args:
        config: Application config. If None, loads from configs/app.yaml.

    Returns:
        A BaseEmbedder implementation ready to use.

    Raises:
        ConfigError: If the provider name is unknown.
    """
    if config is None:
        config = load_app_config()

    provider = config.provider
    if provider not in _EMBEDDER_REGISTRY:
        raise ConfigError(
            f"Unknown embedder provider: '{provider}'. "
            f"Valid options: {', '.join(_EMBEDDER_REGISTRY.keys())}"
        )

    module_path, class_name = _EMBEDDER_REGISTRY[provider]
    embedder_class = _import_class(module_path, class_name)

    logger.info("Creating embedder: provider=%s class=%s", provider, class_name)

    if provider == "ollama":
        return embedder_class(config=config.provider_config)
    else:
        return embedder_class(config=config.provider_config, secrets=config.secrets)

