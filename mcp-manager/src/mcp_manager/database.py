"""Database connection management for DuckDB with MCP extension."""

from pathlib import Path
from typing import Any

import duckdb

from .security import SecurityConfig, apply_security_settings


class DatabaseConnectionPool:
    """Manages DuckDB connections with MCP extension support."""

    def __init__(self, base_path: Path | None = None):
        """
        Initialize connection pool.
        
        Args:
            base_path: Base directory for database files
        """
        self.base_path = base_path or Path.home() / ".aift" / "databases"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._connections: dict[str, duckdb.DuckDBPyConnection] = {}

    def get_connection(
        self,
        name: str,
        path: Path | None = None,
        security: SecurityConfig | None = None
    ) -> duckdb.DuckDBPyConnection:
        """
        Get or create a database connection.
        
        Args:
            name: Database name
            path: Optional custom path (default: base_path/{name}.duckdb)
            security: Optional security configuration
            
        Returns:
            DuckDB connection with MCP extension loaded
        """
        if name in self._connections:
            return self._connections[name]

        if path is None:
            path = self.base_path / f"{name}.duckdb"

        conn = duckdb.connect(str(path))

        # Install and load MCP extension
        install_mcp_extension(conn)

        # Apply security settings if provided
        if security:
            apply_security_settings(conn, security)

        self._connections[name] = conn
        return conn

    def close_connection(self, name: str) -> None:
        """
        Close a database connection.
        
        Args:
            name: Database name
        """
        if name in self._connections:
            self._connections[name].close()
            del self._connections[name]

    def close_all(self) -> None:
        """Close all database connections."""
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def list_connections(self) -> list[str]:
        """List all active connection names."""
        return list(self._connections.keys())


def install_mcp_extension(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Install and load DuckDB MCP extension from community.
    
    Args:
        conn: DuckDB connection
        
    Raises:
        Exception: If extension installation fails
    """
    try:
        # Install from community extensions
        conn.execute("INSTALL duckdb_mcp FROM community")
        conn.execute("LOAD duckdb_mcp")
    except Exception as e:
        raise Exception(f"Failed to install MCP extension: {e}")


def configure_security(
    conn: duckdb.DuckDBPyConnection,
    security: SecurityConfig
) -> None:
    """
    Configure security settings for DuckDB MCP.
    
    Args:
        conn: DuckDB connection
        security: Security configuration
    """
    apply_security_settings(conn, security)


def execute_query(conn: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    """
    Execute SQL query and return results as list of dicts.
    
    Args:
        conn: DuckDB connection
        query: SQL query string
        
    Returns:
        Query results as list of dictionaries
    """
    result = conn.execute(query).fetchall()
    columns = [desc[0] for desc in conn.description] if conn.description else []

    return [dict(zip(columns, row)) for row in result]


def verify_extension(conn: duckdb.DuckDBPyConnection) -> bool:
    """
    Verify that MCP extension is properly loaded.
    
    Args:
        conn: DuckDB connection
        
    Returns:
        True if extension is loaded, False otherwise
    """
    try:
        result = conn.execute(
            "SELECT * FROM duckdb_extensions() WHERE extension_name = 'duckdb_mcp'"
        ).fetchone()
        return result is not None and result[2]  # loaded column
    except Exception:
        return False

