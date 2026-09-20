"""Structured logging configuration.

Sets up JSON-ish structured logging with module name, timestamp, and level.
Security rules:
- NEVER log secrets (API keys, tokens) at any level
- NEVER log full document contents at INFO level (use DEBUG)
- DO log: latency, token counts, model names, chunk counts
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)

