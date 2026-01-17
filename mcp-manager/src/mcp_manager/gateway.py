"""Central MCP Gateway coordinator managing multiple connections."""

import json
from pathlib import Path
from typing import Any

import duckdb

from .config import GatewayConfig
from .database import DatabaseConnectionPool, execute_query
from .registry import ConnectionInfo, ConnectionRegistry, ConnectionStatus


class MCPGateway:
    """Central gateway coordinator for managing MCP connections and operations."""

    def __init__(
        self,
        config: GatewayConfig | None = None,
        base_path: Path | None = None
    ):
        """
        Initialize MCP Gateway.
        
        Args:
            config: Gateway configuration
            base_path: Base directory for databases
        """
        self.config = config or GatewayConfig()
        self.db_pool = DatabaseConnectionPool(base_path)
        self.registry = ConnectionRegistry()
        self._default_db: str = self.config.default_database

    def get_default_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get connection to default database.
        
        Returns:
            DuckDB connection
        """
        db_path = self.config.databases.get(self._default_db)
        return self.db_pool.get_connection(
            self._default_db,
            db_path,
            self.config.security
        )

    def get_database_connection(
        self,
        name: str,
        path: Path | None = None
    ) -> duckdb.DuckDBPyConnection:
        """
        Get connection to named database.
        
        Args:
            name: Database name
            path: Optional custom path
            
        Returns:
            DuckDB connection with MCP extension
        """
        if path is None and name in self.config.databases:
            path = self.config.databases[name]

        return self.db_pool.get_connection(name, path, self.config.security)

    def connect_server(
        self,
        name: str,
        transport: str,
        args: list[str] | dict,
        auto_reconnect: bool = True,
        database: str | None = None
    ) -> None:
        """
        Connect to a remote MCP server.
        
        Args:
            name: Connection name identifier
            transport: Transport type (stdio, tcp, websocket)
            args: Transport-specific arguments
            auto_reconnect: Whether to auto-reconnect on failure
            database: Database to use (default: default database)
            
        Raises:
            ValueError: If command validation fails
            Exception: If connection fails
        """
        # Validate command if using stdio transport
        if transport == "stdio" and isinstance(args, list) and args:
            cmd = args[0]
            # Normalize paths using pathlib for cross-platform compatibility
            cmd_path = Path(cmd).as_posix()
            allowed_paths = [Path(c).as_posix() for c in self.config.security.allowed_commands]

            if cmd_path not in allowed_paths:
                raise ValueError(
                    f"Command '{cmd}' not in allowlist. "
                    f"Add it with: mcp-man security add-command {cmd}"
                )

        # Register connection
        try:
            conn_info = self.registry.register(
                name=name,
                transport=transport,
                args=args,
                auto_reconnect=auto_reconnect
            )
        except ValueError:
            # Already registered, get existing
            conn_info = self.registry.get(name)
            if not conn_info:
                raise

        self.registry.update_status(name, ConnectionStatus.CONNECTING)

        try:
            # Get database connection
            db_name = database or self._default_db
            conn = self.get_database_connection(db_name)

            # Build ATTACH statement
            args_json = json.dumps(args) if isinstance(args, list) else str(args)
            attach_sql = f"""
                ATTACH '{args[0] if isinstance(args, list) and args else ""}' AS {name} (
                    TYPE mcp,
                    TRANSPORT '{transport}',
                    ARGS '{args_json}'
                )
            """

            conn.execute(attach_sql)
            self.registry.update_status(name, ConnectionStatus.CONNECTED)

        except Exception as e:
            self.registry.update_status(name, ConnectionStatus.ERROR, str(e))
            raise Exception(f"Failed to connect to MCP server '{name}': {e}")

    def disconnect_server(self, name: str) -> None:
        """
        Disconnect from an MCP server.
        
        Args:
            name: Connection name identifier
        """
        conn_info = self.registry.get(name)
        if not conn_info:
            raise ValueError(f"Connection '{name}' not found")

        try:
            # Detach from DuckDB
            conn = self.get_default_connection()
            conn.execute(f"DETACH {name}")

            self.registry.update_status(name, ConnectionStatus.DISCONNECTED)
        except Exception as e:
            self.registry.update_status(name, ConnectionStatus.ERROR, str(e))
            raise

    def list_all_resources(self) -> dict[str, list[dict[str, Any]]]:
        """
        List resources from all active MCP connections.
        
        Returns:
            Dictionary mapping connection name to list of resources
        """
        resources: dict[str, list[dict[str, Any]]] = {}

        for conn_info in self.registry.list_active():
            try:
                conn = self.get_default_connection()
                result = execute_query(
                    conn,
                    f"SELECT mcp_list_resources('{conn_info.name}')"
                )
                resources[conn_info.name] = result
            except Exception as e:
                resources[conn_info.name] = [{"error": str(e)}]

        return resources

    def query_across_servers(self, sql: str, database: str | None = None) -> list[dict[str, Any]]:
        """
        Execute SQL query that can access resources from all connected servers.
        
        Args:
            sql: SQL query string
            database: Database to execute query on
            
        Returns:
            Query results as list of dictionaries
        """
        db_name = database or self._default_db
        conn = self.get_database_connection(db_name)
        return execute_query(conn, sql)

    def get_connection_status(self) -> dict[str, str]:
        """
        Get status of all registered connections.
        
        Returns:
            Dictionary mapping connection name to status string
        """
        return {
            conn.name: conn.status.value
            for conn in self.registry.list_all()
        }

    def get_connection_info(self, name: str) -> ConnectionInfo | None:
        """
        Get detailed information about a connection.
        
        Args:
            name: Connection name identifier
            
        Returns:
            ConnectionInfo object or None if not found
        """
        return self.registry.get(name)

    def shutdown(self) -> None:
        """Shutdown gateway and close all connections."""
        # Disconnect all MCP servers
        for conn_info in self.registry.list_all():
            try:
                if conn_info.status == ConnectionStatus.CONNECTED:
                    self.disconnect_server(conn_info.name)
            except Exception:
                pass

        # Close all database connections
        self.db_pool.close_all()

        # Clear registry
        self.registry.clear()

