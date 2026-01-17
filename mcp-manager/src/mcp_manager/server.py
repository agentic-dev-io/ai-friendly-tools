"""MCP server functionality for exposing DuckDB resources."""

import json
from typing import Any

import duckdb

from .database import execute_query
from .security import SecurityConfig


def start_server(
    conn: duckdb.DuckDBPyConnection,
    host: str = "127.0.0.1",
    port: int = 9999,
    transport: str = "stdio",
    config: dict[str, Any] | None = None
) -> None:
    """
    Start MCP server to expose DuckDB content.
    
    Args:
        conn: DuckDB connection
        host: Host to bind to
        port: Port to bind to
        transport: Transport type (stdio, tcp, websocket)
        config: Optional server configuration
        
    Raises:
        Exception: If server start fails
    """
    config_json = json.dumps(config or {})
    query = f"SELECT mcp_server_start('{transport}', '{host}', {port}, '{config_json}')"

    try:
        conn.execute(query)
    except Exception as e:
        raise Exception(f"Failed to start MCP server: {e}")


def stop_server(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Stop running MCP server.
    
    Args:
        conn: DuckDB connection
    """
    conn.execute("SELECT mcp_server_stop()")


def get_server_status(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """
    Get MCP server status.
    
    Args:
        conn: DuckDB connection
        
    Returns:
        Server status information
    """
    result = execute_query(conn, "SELECT mcp_server_status()")

    if result and len(result) > 0:
        status_value = list(result[0].values())[0]
        if isinstance(status_value, str):
            try:
                return json.loads(status_value)
            except json.JSONDecodeError:
                return {"raw": status_value}
        return {"status": status_value}

    return {"status": "unknown"}


def publish_table(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    uri: str,
    format: str = "json"
) -> None:
    """
    Publish a table as an MCP resource.
    
    Args:
        conn: DuckDB connection
        table: Table name to publish
        uri: Resource URI (e.g., 'data://tables/users')
        format: Output format (json, csv, parquet)
        
    Raises:
        Exception: If publishing fails
    """
    query = f"SELECT mcp_publish_table('{table}', '{uri}', '{format}')"

    try:
        conn.execute(query)
    except Exception as e:
        raise Exception(f"Failed to publish table '{table}': {e}")


def publish_query(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    uri: str,
    format: str = "json",
    interval: int | None = None
) -> None:
    """
    Publish a query result as an MCP resource.
    
    Args:
        conn: DuckDB connection
        sql: SQL query to execute
        uri: Resource URI
        format: Output format (json, csv, parquet)
        interval: Optional refresh interval in seconds
        
    Raises:
        Exception: If publishing fails
    """
    if interval is not None:
        query = f"SELECT mcp_publish_query('{sql}', '{uri}', '{format}', {interval})"
    else:
        query = f"SELECT mcp_publish_query('{sql}', '{uri}', '{format}')"

    try:
        conn.execute(query)
    except Exception as e:
        raise Exception(f"Failed to publish query: {e}")


def server_health(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """
    Check MCP server health.
    
    Args:
        conn: DuckDB connection
        
    Returns:
        Health check information
    """
    try:
        result = execute_query(conn, "SELECT mcp_server_health()")

        if result and len(result) > 0:
            health_value = list(result[0].values())[0]
            if isinstance(health_value, str):
                try:
                    return json.loads(health_value)
                except json.JSONDecodeError:
                    return {"raw": health_value}
            return {"health": health_value}

        return {"health": "unknown"}
    except Exception as e:
        return {"health": "error", "error": str(e)}


def list_published_resources(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """
    List all published resources on the MCP server.
    
    Args:
        conn: DuckDB connection
        
    Returns:
        List of published resource information
    """
    try:
        # Query system tables or use appropriate function
        # This is a placeholder - actual implementation depends on extension API
        result = execute_query(
            conn,
            "SELECT * FROM duckdb_tables() WHERE schema = 'mcp_resources'"
        )
        return result
    except Exception:
        return []


def configure_server_security(
    conn: duckdb.DuckDBPyConnection,
    security: SecurityConfig
) -> None:
    """
    Configure security settings for MCP server.
    
    Args:
        conn: DuckDB connection
        security: Security configuration to apply
    """
    from .security import apply_security_settings
    apply_security_settings(conn, security)


def register_prompt_template(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    template: str,
    description: str = ""
) -> None:
    """
    Register a prompt template on the MCP server.
    
    Args:
        conn: DuckDB connection
        name: Template name
        template: Template content
        description: Template description
    """
    query = f"SELECT mcp_register_prompt_template('{name}', '{template}', '{description}')"
    conn.execute(query)


def render_prompt_template(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    args: dict[str, Any] | None = None
) -> str:
    """
    Render a prompt template with arguments.
    
    Args:
        conn: DuckDB connection
        name: Template name
        args: Template arguments
        
    Returns:
        Rendered prompt content
    """
    args_json = json.dumps(args or {})
    result = execute_query(
        conn,
        f"SELECT mcp_render_prompt_template('{name}', '{args_json}')"
    )

    if result and len(result) > 0:
        return list(result[0].values())[0]

    return ""

