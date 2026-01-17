"""Database schema management for MCP tools with FTS support."""

from datetime import datetime
from typing import Any, Optional
import json

import duckdb
from pydantic import BaseModel, Field

from ..database import DatabaseConnectionPool


class ToolSchema(BaseModel):
    """Schema for an MCP tool with metadata."""

    id: Optional[int] = Field(default=None, description="Auto-generated tool ID")
    server_name: str = Field(description="Name of the MCP server")
    tool_name: str = Field(description="Name of the tool")
    description: Optional[str] = Field(default=None, description="Tool description")
    input_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="JSON schema for tool input parameters"
    )
    required_params: list[str] = Field(
        default_factory=list,
        description="List of required parameter names"
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last update"
    )
    enabled: bool = Field(default=True, description="Whether tool is enabled")

    class Config:
        json_schema_extra = {
            "example": {
                "server_name": "example-server",
                "tool_name": "get_data",
                "description": "Retrieves data from the server",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    }
                },
                "required_params": ["id"],
                "enabled": True
            }
        }


class ServerSchema(BaseModel):
    """Schema for an MCP server configuration."""

    name: str = Field(description="Server name identifier")
    transport: str = Field(description="Transport type: stdio, tcp, websocket")
    config: dict[str, Any] = Field(description="Server configuration as JSON")
    enabled: bool = Field(default=True, description="Whether server is enabled")
    last_connected: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last successful connection"
    )
    tool_count: int = Field(default=0, description="Number of tools registered")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "example-server",
                "transport": "stdio",
                "config": {
                    "command": "python",
                    "args": ["server.py"]
                },
                "enabled": True,
                "tool_count": 5
            }
        }


class ToolHistorySchema(BaseModel):
    """Schema for tool execution history."""

    id: Optional[int] = Field(default=None, description="Auto-generated history ID")
    server_name: Optional[str] = Field(default=None, description="Server name")
    tool_name: Optional[str] = Field(default=None, description="Tool name")
    arguments: Optional[dict[str, Any]] = Field(default=None, description="Arguments passed")
    result: Optional[dict[str, Any]] = Field(default=None, description="Result returned")
    success: bool = Field(default=True, description="Whether execution succeeded")
    duration_ms: Optional[int] = Field(default=None, description="Execution duration in ms")
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Execution timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "server_name": "example-server",
                "tool_name": "get_data",
                "arguments": {"id": "123"},
                "result": {"data": "value"},
                "success": True,
                "duration_ms": 150,
            }
        }


class SchemaManager:
    """Manages DuckDB schema creation and initialization."""

    def __init__(self, connection_pool: DatabaseConnectionPool, db_name: str = "mcp_tools"):
        """
        Initialize schema manager.

        Args:
            connection_pool: DatabaseConnectionPool instance
            db_name: Name of the database
        """
        self.connection_pool = connection_pool
        self.db_name = db_name
        self.conn = connection_pool.get_connection(db_name)

    def initialize_schema(self) -> None:
        """Initialize all database tables and sequences."""
        self._create_sequences()
        self._create_tables()
        self._setup_fts_extension()

    def _create_sequences(self) -> None:
        """Create auto-increment sequences for IDs."""
        try:
            self.conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS seq_mcp_tools START 1"
            )
            self.conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS seq_mcp_tool_history START 1"
            )
        except Exception as e:
            raise Exception(f"Failed to create sequences: {e}")

    def _create_tables(self) -> None:
        """Create all required tables."""
        try:
            # mcp_tools table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS mcp_tools (
                    id INTEGER PRIMARY KEY DEFAULT nextval('seq_mcp_tools'),
                    server_name VARCHAR NOT NULL,
                    tool_name VARCHAR NOT NULL,
                    description TEXT,
                    input_schema JSON,
                    required_params VARCHAR[],
                    search_text TEXT GENERATED ALWAYS AS (
                        server_name || ' ' || tool_name || ' ' || COALESCE(description, '')
                    ) VIRTUAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    enabled BOOLEAN DEFAULT TRUE,
                    UNIQUE(server_name, tool_name)
                )
            """)

            # mcp_servers table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    name VARCHAR PRIMARY KEY,
                    transport VARCHAR NOT NULL,
                    config JSON NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    last_connected TIMESTAMP,
                    tool_count INTEGER DEFAULT 0
                )
            """)

            # mcp_tool_history table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS mcp_tool_history (
                    id INTEGER PRIMARY KEY DEFAULT nextval('seq_mcp_tool_history'),
                    server_name VARCHAR,
                    tool_name VARCHAR,
                    arguments JSON,
                    result JSON,
                    success BOOLEAN,
                    duration_ms INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        except Exception as e:
            raise Exception(f"Failed to create tables: {e}")

    def _setup_fts_extension(self) -> None:
        """Load FTS extension and create full-text search index."""
        try:
            # Load FTS extension
            self.conn.execute("INSTALL fts")
            self.conn.execute("LOAD fts")

            # Create FTS index on mcp_tools search_text
            # Note: DuckDB FTS is created automatically on virtual columns
            # but we can create a dedicated index table if needed
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS mcp_tools_fts AS
                SELECT id, server_name, tool_name, description, search_text
                FROM mcp_tools
            """)

        except Exception as e:
            # FTS extension may not be available in all DuckDB versions
            print(f"Warning: Could not load FTS extension: {e}")

    def drop_all_tables(self) -> None:
        """Drop all tables and sequences (for testing/reset)."""
        try:
            self.conn.execute("DROP TABLE IF EXISTS mcp_tools_fts")
            self.conn.execute("DROP TABLE IF EXISTS mcp_tool_history")
            self.conn.execute("DROP TABLE IF EXISTS mcp_servers")
            self.conn.execute("DROP TABLE IF EXISTS mcp_tools")
            self.conn.execute("DROP SEQUENCE IF EXISTS seq_mcp_tool_history")
            self.conn.execute("DROP SEQUENCE IF EXISTS seq_mcp_tools")
        except Exception as e:
            raise Exception(f"Failed to drop tables: {e}")


class ToolRegistry:
    """Registry for managing MCP tools with database persistence."""

    def __init__(self, connection_pool: DatabaseConnectionPool, db_name: str = "mcp_tools"):
        """
        Initialize tool registry.

        Args:
            connection_pool: DatabaseConnectionPool instance
            db_name: Name of the database
        """
        self.connection_pool = connection_pool
        self.db_name = db_name
        self.conn = connection_pool.get_connection(db_name)

        # Initialize schema if not already done
        schema_manager = SchemaManager(connection_pool, db_name)
        schema_manager.initialize_schema()

    def add_tool(
        self,
        server_name: str,
        tool_name: str,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        required_params: Optional[list[str]] = None,
    ) -> ToolSchema:
        """
        Add a new tool to the registry.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool
            description: Optional tool description
            input_schema: Optional JSON schema for parameters
            required_params: Optional list of required parameter names

        Returns:
            ToolSchema object of the added tool

        Raises:
            ValueError: If tool already exists
        """
        try:
            # Check if tool already exists
            existing = self.conn.execute(
                """
                SELECT id FROM mcp_tools
                WHERE server_name = ? AND tool_name = ?
                """,
                [server_name, tool_name]
            ).fetchone()

            if existing:
                raise ValueError(
                    f"Tool '{tool_name}' already exists in server '{server_name}'"
                )

            # Insert new tool
            schema_json = json.dumps(input_schema) if input_schema else None
            params_array = required_params or []

            self.conn.execute(
                """
                INSERT INTO mcp_tools (
                    server_name, tool_name, description,
                    input_schema, required_params, last_updated, enabled
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, TRUE)
                """,
                [
                    server_name,
                    tool_name,
                    description,
                    schema_json,
                    params_array
                ]
            )

            # Update tool count for server
            self._update_server_tool_count(server_name)

            # Return the created tool
            return self.get_tool(server_name, tool_name)

        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to add tool: {e}")

    def get_tool(self, server_name: str, tool_name: str) -> Optional[ToolSchema]:
        """
        Get a specific tool by server and tool name.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool

        Returns:
            ToolSchema object if found, None otherwise
        """
        try:
            result = self.conn.execute(
                """
                SELECT
                    id, server_name, tool_name, description,
                    input_schema, required_params, last_updated, enabled
                FROM mcp_tools
                WHERE server_name = ? AND tool_name = ?
                """,
                [server_name, tool_name]
            ).fetchone()

            if not result:
                return None

            return self._row_to_tool_schema(result)

        except Exception as e:
            raise Exception(f"Failed to get tool: {e}")

    def list_tools(self, server_name: Optional[str] = None) -> list[ToolSchema]:
        """
        List tools, optionally filtered by server.

        Args:
            server_name: Optional server name to filter by

        Returns:
            List of ToolSchema objects
        """
        try:
            if server_name:
                query = """
                    SELECT
                        id, server_name, tool_name, description,
                        input_schema, required_params, last_updated, enabled
                    FROM mcp_tools
                    WHERE server_name = ?
                    ORDER BY tool_name
                """
                results = self.conn.execute(query, [server_name]).fetchall()
            else:
                query = """
                    SELECT
                        id, server_name, tool_name, description,
                        input_schema, required_params, last_updated, enabled
                    FROM mcp_tools
                    ORDER BY server_name, tool_name
                """
                results = self.conn.execute(query).fetchall()

            return [self._row_to_tool_schema(row) for row in results]

        except Exception as e:
            raise Exception(f"Failed to list tools: {e}")

    def search_tools(self, query: str) -> list[ToolSchema]:
        """
        Search tools by description and name using FTS.

        Args:
            query: Search query string

        Returns:
            List of matching ToolSchema objects
        """
        try:
            # Use LIKE for simple text search if FTS is not available
            search_term = f"%{query.lower()}%"

            results = self.conn.execute(
                """
                SELECT
                    id, server_name, tool_name, description,
                    input_schema, required_params, last_updated, enabled
                FROM mcp_tools
                WHERE LOWER(search_text) LIKE ?
                ORDER BY server_name, tool_name
                """,
                [search_term]
            ).fetchall()

            return [self._row_to_tool_schema(row) for row in results]

        except Exception as e:
            raise Exception(f"Failed to search tools: {e}")

    def update_tool(
        self,
        server_name: str,
        tool_name: str,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        required_params: Optional[list[str]] = None,
        enabled: Optional[bool] = None,
    ) -> ToolSchema:
        """
        Update an existing tool.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool
            description: Optional new description
            input_schema: Optional new input schema
            required_params: Optional new required params list
            enabled: Optional enabled status

        Returns:
            Updated ToolSchema object

        Raises:
            ValueError: If tool does not exist
        """
        try:
            # Check if tool exists
            existing = self.get_tool(server_name, tool_name)
            if not existing:
                raise ValueError(
                    f"Tool '{tool_name}' not found in server '{server_name}'"
                )

            # Build update query dynamically
            updates = []
            params = []

            if description is not None:
                updates.append("description = ?")
                params.append(description)

            if input_schema is not None:
                updates.append("input_schema = ?")
                params.append(json.dumps(input_schema))

            if required_params is not None:
                updates.append("required_params = ?")
                params.append(required_params)

            if enabled is not None:
                updates.append("enabled = ?")
                params.append(enabled)

            if not updates:
                return existing

            updates.append("last_updated = CURRENT_TIMESTAMP")
            params.extend([server_name, tool_name])

            query = f"""
                UPDATE mcp_tools
                SET {', '.join(updates)}
                WHERE server_name = ? AND tool_name = ?
            """

            self.conn.execute(query, params)

            return self.get_tool(server_name, tool_name)

        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to update tool: {e}")

    def delete_tool(self, server_name: str, tool_name: str) -> bool:
        """
        Delete a tool from the registry.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool

        Returns:
            True if tool was deleted, False if not found
        """
        try:
            result = self.conn.execute(
                """
                DELETE FROM mcp_tools
                WHERE server_name = ? AND tool_name = ?
                """,
                [server_name, tool_name]
            )

            deleted = result.fetchall()
            if deleted:
                self._update_server_tool_count(server_name)
                return True

            return False

        except Exception as e:
            raise Exception(f"Failed to delete tool: {e}")

    def count_tools(self, server_name: Optional[str] = None) -> int:
        """
        Count tools in registry, optionally filtered by server.

        Args:
            server_name: Optional server name to filter by

        Returns:
            Number of tools
        """
        try:
            if server_name:
                query = "SELECT COUNT(*) FROM mcp_tools WHERE server_name = ?"
                result = self.conn.execute(query, [server_name]).fetchone()
            else:
                query = "SELECT COUNT(*) FROM mcp_tools"
                result = self.conn.execute(query).fetchone()

            return result[0] if result else 0

        except Exception as e:
            raise Exception(f"Failed to count tools: {e}")

    def clear_tools(self, server_name: Optional[str] = None) -> int:
        """
        Clear tools from registry, optionally filtered by server.

        Args:
            server_name: Optional server name to filter by

        Returns:
            Number of tools deleted
        """
        try:
            if server_name:
                result = self.conn.execute(
                    "DELETE FROM mcp_tools WHERE server_name = ?",
                    [server_name]
                )
                self._update_server_tool_count(server_name)
            else:
                result = self.conn.execute("DELETE FROM mcp_tools")
                # Reset all server tool counts
                self.conn.execute("UPDATE mcp_servers SET tool_count = 0")

            return result.rows_affected if hasattr(result, 'rows_affected') else 0

        except Exception as e:
            raise Exception(f"Failed to clear tools: {e}")

    def get_enabled_tools(self, server_name: Optional[str] = None) -> list[ToolSchema]:
        """
        Get only enabled tools, optionally filtered by server.

        Args:
            server_name: Optional server name to filter by

        Returns:
            List of enabled ToolSchema objects
        """
        try:
            if server_name:
                query = """
                    SELECT
                        id, server_name, tool_name, description,
                        input_schema, required_params, last_updated, enabled
                    FROM mcp_tools
                    WHERE server_name = ? AND enabled = TRUE
                    ORDER BY tool_name
                """
                results = self.conn.execute(query, [server_name]).fetchall()
            else:
                query = """
                    SELECT
                        id, server_name, tool_name, description,
                        input_schema, required_params, last_updated, enabled
                    FROM mcp_tools
                    WHERE enabled = TRUE
                    ORDER BY server_name, tool_name
                """
                results = self.conn.execute(query).fetchall()

            return [self._row_to_tool_schema(row) for row in results]

        except Exception as e:
            raise Exception(f"Failed to get enabled tools: {e}")

    def add_server(
        self,
        name: str,
        transport: str,
        config: dict[str, Any],
        enabled: bool = True,
    ) -> ServerSchema:
        """
        Add a server to the registry.

        Args:
            name: Server name identifier
            transport: Transport type (stdio, tcp, websocket)
            config: Server configuration as dict
            enabled: Whether server is enabled

        Returns:
            ServerSchema object

        Raises:
            ValueError: If server already exists
        """
        try:
            # Check if server already exists
            existing = self.conn.execute(
                "SELECT name FROM mcp_servers WHERE name = ?",
                [name]
            ).fetchone()

            if existing:
                raise ValueError(f"Server '{name}' already exists")

            config_json = json.dumps(config)

            self.conn.execute(
                """
                INSERT INTO mcp_servers (name, transport, config, enabled, tool_count)
                VALUES (?, ?, ?, ?, 0)
                """,
                [name, transport, config_json, enabled]
            )

            return self.get_server(name)

        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to add server: {e}")

    def get_server(self, name: str) -> Optional[ServerSchema]:
        """
        Get a server by name.

        Args:
            name: Server name identifier

        Returns:
            ServerSchema object if found, None otherwise
        """
        try:
            result = self.conn.execute(
                """
                SELECT name, transport, config, enabled, last_connected, tool_count
                FROM mcp_servers
                WHERE name = ?
                """,
                [name]
            ).fetchone()

            if not result:
                return None

            return self._row_to_server_schema(result)

        except Exception as e:
            raise Exception(f"Failed to get server: {e}")

    def list_servers(self) -> list[ServerSchema]:
        """
        List all servers in registry.

        Returns:
            List of ServerSchema objects
        """
        try:
            results = self.conn.execute(
                """
                SELECT name, transport, config, enabled, last_connected, tool_count
                FROM mcp_servers
                ORDER BY name
                """
            ).fetchall()

            return [self._row_to_server_schema(row) for row in results]

        except Exception as e:
            raise Exception(f"Failed to list servers: {e}")

    def delete_server(self, name: str) -> bool:
        """
        Delete a server and its tools.

        Args:
            name: Server name identifier

        Returns:
            True if server was deleted, False if not found
        """
        try:
            # Delete tools first
            self.clear_tools(name)

            # Then delete server
            result = self.conn.execute(
                "DELETE FROM mcp_servers WHERE name = ?",
                [name]
            )

            return True

        except Exception as e:
            raise Exception(f"Failed to delete server: {e}")

    def log_tool_execution(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
        success: bool = True,
        duration_ms: Optional[int] = None,
    ) -> ToolHistorySchema:
        """
        Log a tool execution to history.

        Args:
            server_name: Server name
            tool_name: Tool name
            arguments: Arguments passed to tool
            result: Result returned from tool
            success: Whether execution succeeded
            duration_ms: Execution duration in milliseconds

        Returns:
            ToolHistorySchema object
        """
        try:
            arguments_json = json.dumps(arguments) if arguments else None
            result_json = json.dumps(result) if result else None

            self.conn.execute(
                """
                INSERT INTO mcp_tool_history (
                    server_name, tool_name, arguments, result,
                    success, duration_ms, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [server_name, tool_name, arguments_json, result_json, success, duration_ms]
            )

            # Get the history entry
            history = self.conn.execute(
                """
                SELECT id, server_name, tool_name, arguments, result, success, duration_ms, timestamp
                FROM mcp_tool_history
                WHERE server_name = ? AND tool_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                [server_name, tool_name]
            ).fetchone()

            return self._row_to_history_schema(history)

        except Exception as e:
            raise Exception(f"Failed to log tool execution: {e}")

    def get_tool_history(
        self,
        server_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[ToolHistorySchema]:
        """
        Get tool execution history.

        Args:
            server_name: Optional server name filter
            tool_name: Optional tool name filter
            limit: Maximum number of records to return

        Returns:
            List of ToolHistorySchema objects
        """
        try:
            query = "SELECT id, server_name, tool_name, arguments, result, success, duration_ms, timestamp FROM mcp_tool_history"
            params = []

            if server_name and tool_name:
                query += " WHERE server_name = ? AND tool_name = ?"
                params = [server_name, tool_name]
            elif server_name:
                query += " WHERE server_name = ?"
                params = [server_name]
            elif tool_name:
                query += " WHERE tool_name = ?"
                params = [tool_name]

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            results = self.conn.execute(query, params).fetchall()

            return [self._row_to_history_schema(row) for row in results]

        except Exception as e:
            raise Exception(f"Failed to get tool history: {e}")

    def _update_server_tool_count(self, server_name: str) -> None:
        """
        Update the tool count for a server.

        Args:
            server_name: Server name to update
        """
        try:
            count = self.count_tools(server_name)
            self.conn.execute(
                "UPDATE mcp_servers SET tool_count = ? WHERE name = ?",
                [count, server_name]
            )
        except Exception:
            pass  # Silently fail if server doesn't exist

    @staticmethod
    def _row_to_tool_schema(row: tuple) -> ToolSchema:
        """Convert database row to ToolSchema."""
        return ToolSchema(
            id=row[0],
            server_name=row[1],
            tool_name=row[2],
            description=row[3],
            input_schema=json.loads(row[4]) if row[4] else None,
            required_params=row[5] if row[5] else [],
            last_updated=row[6],
            enabled=row[7]
        )

    @staticmethod
    def _row_to_server_schema(row: tuple) -> ServerSchema:
        """Convert database row to ServerSchema."""
        return ServerSchema(
            name=row[0],
            transport=row[1],
            config=json.loads(row[2]) if row[2] else {},
            enabled=row[3],
            last_connected=row[4],
            tool_count=row[5] or 0
        )

    @staticmethod
    def _row_to_history_schema(row: tuple) -> ToolHistorySchema:
        """Convert database row to ToolHistorySchema."""
        return ToolHistorySchema(
            id=row[0],
            server_name=row[1],
            tool_name=row[2],
            arguments=json.loads(row[3]) if row[3] else None,
            result=json.loads(row[4]) if row[4] else None,
            success=row[5],
            duration_ms=row[6],
            timestamp=row[7]
        )
