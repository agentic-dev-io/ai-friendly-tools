"""Configuration management for AIFT tools."""

from pathlib import Path
from typing import Any, Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AIFTConfig(BaseSettings):
    """Global configuration for AIFT tools."""

    model_config = SettingsConfigDict(
        env_prefix="AIFT_",
        env_file=str(Path.home() / "aift" / "config.env"),
        env_file_encoding="utf-8",
        extra="allow",
    )

    log_level: str = Field(default="INFO", description="Logging level")
    config_dir: Path = Field(
        default_factory=lambda: Path.home() / "aift",
        description="Configuration directory"
    )

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary with tool-specific configuration
        """
        config_file = self.config_dir / f"{tool_name}.json"
        if config_file.exists():
            import json
            with open(config_file) as f:
                return json.load(f)
        return {}

    def save_tool_config(self, tool_name: str, config: Dict[str, Any]) -> None:
        """
        Save configuration for a specific tool.

        Args:
            tool_name: Name of the tool
            config: Configuration dictionary to save
        """
        self.config_dir.mkdir(parents=True, exist_ok=True)
        config_file = self.config_dir / f"{tool_name}.json"
        import json
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)


def get_config() -> AIFTConfig:
    """Get the global AIFT configuration."""
    return AIFTConfig()
