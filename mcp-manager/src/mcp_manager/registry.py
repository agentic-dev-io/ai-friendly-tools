"""Connection registry for tracking active MCP server connections."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConnectionStatus(str, Enum):
    """Status of an MCP connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class ConnectionInfo(BaseModel):
    """Information about an MCP server connection."""

    name: str = Field(description="Connection name identifier")
    transport: str = Field(description="Transport type: stdio, tcp, websocket")
    args: list[str] | dict = Field(description="Transport-specific arguments")
    status: ConnectionStatus = Field(
        default=ConnectionStatus.DISCONNECTED,
        description="Current connection status"
    )
    connected_at: datetime | None = Field(
        default=None,
        description="Timestamp when connection was established"
    )
    last_error: str | None = Field(
        default=None,
        description="Last error message if status is ERROR"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional connection metadata"
    )
    auto_reconnect: bool = Field(
        default=True,
        description="Whether to auto-reconnect on failure"
    )


class ConnectionRegistry:
    """Central registry for managing MCP server connections."""

    _instance: "ConnectionRegistry | None" = None

    def __new__(cls) -> "ConnectionRegistry":
        """Singleton pattern for connection registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connections = {}
        return cls._instance

    def __init__(self):
        """Initialize registry (only once due to singleton)."""
        if not hasattr(self, "_connections"):
            self._connections: dict[str, ConnectionInfo] = {}

    def register(
        self,
        name: str,
        transport: str,
        args: list[str] | dict,
        auto_reconnect: bool = True,
        metadata: dict[str, Any] | None = None
    ) -> ConnectionInfo:
        """
        Register a new MCP server connection.
        
        Args:
            name: Connection name identifier
            transport: Transport type (stdio, tcp, websocket)
            args: Transport-specific arguments
            auto_reconnect: Whether to auto-reconnect on failure
            metadata: Additional connection metadata
            
        Returns:
            ConnectionInfo object for the registered connection
            
        Raises:
            ValueError: If connection name already exists
        """
        if name in self._connections:
            raise ValueError(f"Connection '{name}' already exists")

        conn_info = ConnectionInfo(
            name=name,
            transport=transport,
            args=args,
            status=ConnectionStatus.DISCONNECTED,
            auto_reconnect=auto_reconnect,
            metadata=metadata or {}
        )

        self._connections[name] = conn_info
        return conn_info

    def get(self, name: str) -> ConnectionInfo | None:
        """
        Get connection info by name.
        
        Args:
            name: Connection name identifier
            
        Returns:
            ConnectionInfo if found, None otherwise
        """
        return self._connections.get(name)

    def update_status(
        self,
        name: str,
        status: ConnectionStatus,
        error: str | None = None
    ) -> None:
        """
        Update connection status.
        
        Args:
            name: Connection name identifier
            status: New connection status
            error: Optional error message
        """
        if name not in self._connections:
            return

        conn = self._connections[name]
        conn.status = status

        if status == ConnectionStatus.CONNECTED:
            conn.connected_at = datetime.now()
            conn.last_error = None
        elif status == ConnectionStatus.ERROR:
            conn.last_error = error

    def list_active(self) -> list[ConnectionInfo]:
        """
        List all connections with CONNECTED status.
        
        Returns:
            List of active ConnectionInfo objects
        """
        return [
            conn for conn in self._connections.values()
            if conn.status == ConnectionStatus.CONNECTED
        ]

    def list_all(self) -> list[ConnectionInfo]:
        """
        List all registered connections.
        
        Returns:
            List of all ConnectionInfo objects
        """
        return list(self._connections.values())

    def remove(self, name: str) -> bool:
        """
        Remove a connection from registry.
        
        Args:
            name: Connection name identifier
            
        Returns:
            True if connection was removed, False if not found
        """
        if name in self._connections:
            del self._connections[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all connections from registry."""
        self._connections.clear()

    def get_status_summary(self) -> dict[str, int]:
        """
        Get summary of connection statuses.
        
        Returns:
            Dictionary mapping status to count
        """
        summary: dict[str, int] = {}
        for conn in self._connections.values():
            status = conn.status.value
            summary[status] = summary.get(status, 0) + 1
        return summary

