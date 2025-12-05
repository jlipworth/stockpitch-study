"""Logging configuration for stock pitch CLI.

Provides centralized logging setup with:
- Console handler (INFO by default, DEBUG with --verbose)
- Optional file handler (with --log-file)
- Consistent formatting across all modules
- Structured logging for key events
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        verbose: If True, set console to DEBUG level (default INFO)
        log_file: Optional path to log file (logs at DEBUG level)
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler - INFO or DEBUG based on verbose flag
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console format: simpler for readability
    console_format = logging.Formatter(
        "%(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler (if specified) - always DEBUG level
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # File format: more detailed with timestamps
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("edgar").setLevel(logging.WARNING)
    logging.getLogger("lancedb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
