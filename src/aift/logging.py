"""Logging configuration for AIFT using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure loguru to log to ~/aift/logs directory.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove default handler
    logger.remove()

    # Create log directory in user's home
    log_dir = Path.home() / "aift" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add console handler with rich formatting
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )

    # Add file handler with rotation
    logger.add(
        log_dir / "aift_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    logger.info(f"Logging initialized. Logs are stored in: {log_dir}")


# Initialize logging when module is imported
setup_logging()
