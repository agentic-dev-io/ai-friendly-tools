"""
Configuration for mcp-man tool.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class MCPConfig(BaseModel):
    """MCP Server configuration."""

    host: str = "127.0.0.1"
    port: int = 9999
    debug: bool = False


class DatabaseConfig(BaseModel):
    """Database configuration."""

    name: str
    path: Path
    mcp_enabled: bool = True


class MCPManagerConfig(BaseModel):
    """Overall configuration for mcp-man."""

    mcp: MCPConfig = MCPConfig()
    databases: list[DatabaseConfig] = []

    class Config:
        env_prefix = "MCP_MAN_"