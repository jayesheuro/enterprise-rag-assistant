"""FastAPI application factory.

Creates and configures the FastAPI app with:
- Router registration
- Exception handlers for structured error responses
- Startup/shutdown events
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health
from app.core.exceptions import AppError
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    setup_logging()
    logger.info("Enterprise RAG Assistant starting up")
    yield
    logger.info("Enterprise RAG Assistant shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(
        title="Enterprise RAG Assistant",
        description="Document Q&A platform with built-in evaluation lab",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── Register routers ────────────────────────────────────────────────
    app.include_router(health.router, tags=["health"])
    
    from app.api.routes import ingest, chat
    app.include_router(ingest.router)
    app.include_router(chat.router)
    # ── Exception handlers ──────────────────────────────────────────────
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Handle all application errors with structured JSON response."""
        logger.error("AppError: %s (status=%d)", exc.message, exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all for unhandled exceptions. Log and return 500."""
        logger.exception("Unhandled exception: %s", str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
            },
        )

    return app


# Application instance for uvicorn
app = create_app()

