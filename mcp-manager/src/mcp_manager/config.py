"""
Configuration for mcp-man tool.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from .security import SecurityConfig


class ConnectionConfig(BaseModel):
    """Configuration for a remote MCP server connection."""

    name: str = Field(description="Connection name identifier")
    transport: str = Field(
        description="Transport type: stdio, tcp, or websocket"
    )
    args: list[str] | dict = Field(
        default_factory=list,
        description="Transport-specific arguments"
    )
    auto_reconnect: bool = Field(
        default=True,
        description="Automatically reconnect on connection failure"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this connection is enabled"
    )


class ServerConfig(BaseModel):
    """MCP Server configuration."""

    host: str = Field(
        default="127.0.0.1",
        description="Host to bind MCP server to"
    )
    port: int = Field(
        default=9999,
        description="Port to bind MCP server to"
    )
    transport: str = Field(
        default="stdio",
        description="Server transport type"
    )
    enabled: bool = Field(
        default=False,
        description="Whether server mode is enabled"
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""

    name: str
    path: Path
    mcp_enabled: bool = True


class GatewayConfig(BaseModel):
    """Gateway configuration for MCP Manager."""

    databases: dict[str, Path] = Field(
        default_factory=dict,
        description="Named database paths"
    )
    connections: list[ConnectionConfig] = Field(
        default_factory=list,
        description="Remote MCP server connections"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="MCP server configuration"
    )
    default_database: str = Field(
        default="default",
        description="Default database name"
    )

    class Config:
        env_prefix = "MCP_MAN_"
