"""
AIFTS MCP Manager - DuckDB MCP Gateway

Centralized gateway for managing multiple MCP server connections
and exposing DuckDB databases via Model Context Protocol.
"""

__version__ = "0.1.0"

from .config import (
    ConnectionConfig,
    DatabaseConfig,
    GatewayConfig,
    ServerConfig,
)
from .gateway import MCPGateway
from .registry import ConnectionInfo, ConnectionRegistry, ConnectionStatus
from .security import SecurityConfig
from .workflows import (
    Pipeline,
    PipelineBuilder,
    PipelineStep,
    WorkflowRecommender,
)

__all__ = [
    # Gateway
    "MCPGateway",
    "GatewayConfig",
    "ConnectionConfig",
    "ServerConfig",
    "DatabaseConfig",
    "SecurityConfig",
    "ConnectionRegistry",
    "ConnectionInfo",
    "ConnectionStatus",
    # Workflows
    "Pipeline",
    "PipelineBuilder",
    "PipelineStep",
    "WorkflowRecommender",
]
