"""MCP Server tool discovery and scanning."""

import asyncio
from datetime import datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, ToolListChangedNotification

from mcp_manager.registry import ConnectionInfo, ConnectionRegistry, ConnectionStatus


class ToolInfo(BaseModel):
    """Information about a tool discovered on an MCP server."""

    server: str = Field(description="Name of the MCP server")
    name: str = Field(description="Tool name identifier")
    description: str = Field(description="Tool description")
    input_schema: dict[str, Any] = Field(description="JSON schema for tool input")
    required_params: list[str] = Field(default_factory=list, description="List of required parameters")
    source_mcp_version: str = Field(description="MCP protocol version (e.g., '2024-11-05')")


class ToolRegistry:
    """Registry for storing and managing discovered tools."""

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: dict[str, list[ToolInfo]] = {}
        self._last_updated: dict[str, datetime] = {}

    def add_tools(self, server_name: str, tools: list[ToolInfo]) -> None:
        """
        Add tools for a server to the registry.

        Args:
            server_name: Name of the MCP server
            tools: List of ToolInfo objects
        """
        self._tools[server_name] = tools
        self._last_updated[server_name] = datetime.now()
        logger.info(f"Added {len(tools)} tools for server '{server_name}'")

    def get_tools(self, server_name: str) -> list[ToolInfo]:
        """
        Get all tools for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            List of ToolInfo objects for the server
        """
        return self._tools.get(server_name, [])

    def get_all_tools(self) -> dict[str, list[ToolInfo]]:
        """
        Get all tools from all servers.

        Returns:
            Dictionary mapping server name to list of tools
        """
        return self._tools.copy()

    def get_last_update(self, server_name: str) -> datetime | None:
        """
        Get the last update timestamp for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Datetime of last update or None if never updated
        """
        return self._last_updated.get(server_name)

    def clear(self) -> None:
        """Clear all tools from the registry."""
        self._tools.clear()
        self._last_updated.clear()
        logger.info("Tool registry cleared")

    def get_tool_count(self) -> int:
        """
        Get total number of tools in registry.

        Returns:
            Total count of tools across all servers
        """
        return sum(len(tools) for tools in self._tools.values())


class AsyncToolScanner:
    """Scanner for discovering and indexing tools from MCP servers."""

    def __init__(self, timeout: float = 10.0):
        """
        Initialize the tool scanner.

        Args:
            timeout: Connection timeout in seconds
        """
        self.timeout = timeout
        self.registry = ConnectionRegistry()

    async def scan_server(self, server_name: str) -> list[ToolInfo]:
        """
        Scan a single MCP server and extract tool information.

        Args:
            server_name: Name of the server to scan

        Returns:
            List of ToolInfo objects discovered on the server

        Raises:
            ConnectionError: If unable to connect to the server
        """
        logger.info(f"Starting scan of server '{server_name}'")

        conn_info = self.registry.get(server_name)
        if not conn_info:
            logger.warning(f"Server '{server_name}' not found in registry")
            return []

        # Only support stdio transport for now
        if conn_info.transport != "stdio":
            logger.warning(f"Server '{server_name}' uses unsupported transport: {conn_info.transport}")
            return []

        tools: list[ToolInfo] = []

        try:
            # Parse stdio arguments
            if isinstance(conn_info.args, dict):
                command = conn_info.args.get("command")
                args = conn_info.args.get("args", [])
            else:
                command = conn_info.args[0] if conn_info.args else None
                args = conn_info.args[1:] if len(conn_info.args) > 1 else []

            if not command:
                logger.error(f"No command specified for server '{server_name}'")
                return []

            # Connect to the server and list tools
            async with await asyncio.wait_for(
                stdio_client(command, args),
                timeout=self.timeout
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the session
                    await session.initialize()

                    # Get protocol version
                    protocol_version = getattr(session, "protocol_version", "2024-11-05")

                    # List all available tools
                    response = await session.list_tools()

                    if not response.tools:
                        logger.info(f"No tools found on server '{server_name}'")
                        return []

                    # Extract tool information
                    for tool in response.tools:
                        try:
                            # Extract required parameters from input schema
                            required_params: list[str] = []
                            input_schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}

                            if isinstance(input_schema, dict):
                                required_params = input_schema.get("required", [])

                            tool_info = ToolInfo(
                                server=server_name,
                                name=tool.name,
                                description=tool.description or "",
                                input_schema=input_schema,
                                required_params=required_params,
                                source_mcp_version=protocol_version,
                            )
                            tools.append(tool_info)
                            logger.debug(f"Discovered tool '{tool.name}' on server '{server_name}'")

                        except Exception as e:
                            logger.warning(
                                f"Failed to process tool on server '{server_name}': {e}",
                                extra={"tool_name": getattr(tool, "name", "unknown")}
                            )
                            continue

                    logger.info(f"Successfully scanned server '{server_name}': found {len(tools)} tools")

        except asyncio.TimeoutError:
            logger.error(f"Timeout while connecting to server '{server_name}' (>{self.timeout}s)")
            self.registry.update_status(
                server_name,
                ConnectionStatus.ERROR,
                f"Scan timeout after {self.timeout}s"
            )
        except ConnectionError as e:
            logger.error(f"Connection failed for server '{server_name}': {e}")
            self.registry.update_status(
                server_name,
                ConnectionStatus.ERROR,
                f"Connection failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error scanning server '{server_name}': {e}", exc_info=True)
            self.registry.update_status(
                server_name,
                ConnectionStatus.ERROR,
                f"Scan error: {str(e)}"
            )

        return tools

    async def scan_all_servers(self, registry: ConnectionRegistry) -> dict[str, list[ToolInfo]]:
        """
        Scan all registered servers in parallel.

        Args:
            registry: ConnectionRegistry with registered servers

        Returns:
            Dictionary mapping server name to list of ToolInfo objects
        """
        logger.info("Starting parallel scan of all registered servers")

        all_connections = registry.list_all()
        if not all_connections:
            logger.warning("No servers registered in connection registry")
            return {}

        # Create scan tasks for all servers
        scan_tasks = [
            self.scan_server(conn.name)
            for conn in all_connections
        ]

        # Execute all scans in parallel with timeout protection
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*scan_tasks, return_exceptions=True),
                timeout=self.timeout * len(all_connections) + 30  # Add buffer time
            )
        except asyncio.TimeoutError:
            logger.error("Overall scan timeout exceeded")
            results = []

        # Build result dictionary
        result: dict[str, list[ToolInfo]] = {}
        for conn, res in zip(all_connections, results):
            if isinstance(res, Exception):
                logger.warning(f"Scan of '{conn.name}' resulted in error: {res}")
                result[conn.name] = []
            else:
                result[conn.name] = res

        logger.info(f"Completed parallel scan of {len(all_connections)} servers")
        return result

    async def update_tool_index(
        self,
        tool_registry: ToolRegistry,
        servers: list[ConnectionInfo] | None = None
    ) -> dict[str, list[ToolInfo]]:
        """
        Scan all servers and update the tool index.

        Args:
            tool_registry: ToolRegistry to update with discovered tools
            servers: Optional list of specific servers to scan. If None, scans all registered servers.

        Returns:
            Dictionary of scanned servers and their tools
        """
        logger.info("Updating tool index")

        # Get servers to scan
        if servers is None:
            servers = self.registry.list_all()

        if not servers:
            logger.warning("No servers to scan for tool index update")
            return {}

        # Create scan tasks for specified servers
        scan_tasks = [
            self.scan_server(server.name)
            for server in servers
        ]

        # Execute all scans in parallel
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*scan_tasks, return_exceptions=True),
                timeout=self.timeout * len(servers) + 30
            )
        except asyncio.TimeoutError:
            logger.error("Tool index update timeout exceeded")
            results = []

        # Update tool registry with results
        scanned_servers: dict[str, list[ToolInfo]] = {}
        for server, res in zip(servers, results):
            if isinstance(res, Exception):
                logger.warning(f"Failed to update tools for '{server.name}': {res}")
                scanned_servers[server.name] = []
            else:
                tool_registry.add_tools(server.name, res)
                scanned_servers[server.name] = res

        logger.info(
            f"Tool index updated: {tool_registry.get_tool_count()} total tools "
            f"across {len(scanned_servers)} servers"
        )

        return scanned_servers


__all__ = [
    "ToolInfo",
    "ToolRegistry",
    "AsyncToolScanner",
]
