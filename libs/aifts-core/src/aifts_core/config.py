"""
Configuration management for AIFTS tools.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Config(BaseModel):
    """Base configuration for AIFTS tools."""

    debug: bool = False
    log_level: str = "INFO"
    config_dir: Path = Path.home() / ".aifts"

    class Config:
        env_prefix = "AIFTS_"

    def ensure_config_dir(self) -> None:
        """Ensure config directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)