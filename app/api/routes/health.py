"""Health check endpoint.

Returns the application status, active provider, and model name.
Useful for monitoring and verifying the deployment is working.
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Check application health and return provider info.

    Returns:
        JSON with status, provider name, and active model.
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "provider": settings.provider,
        "model": settings.provider_config.model if settings.provider_config else "unknown",
    }

