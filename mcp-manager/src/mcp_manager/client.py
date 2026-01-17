"""MCP client functionality for querying remote MCP servers."""

import json
from typing import Any

import duckdb

from .database import execute_query


def attach_server(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    transport: str,
    args: list[str] | dict
) -> None:
    """
    Attach to a remote MCP server.
    
    Args:
        conn: DuckDB connection
        name: Server name identifier
        transport: Transport type (stdio, tcp, websocket)
        args: Transport-specific arguments
        
    Raises:
        Exception: If ATTACH fails
    """
    args_json = json.dumps(args) if isinstance(args, list) else str(args)

    # For stdio transport, first arg is the command
    command = args[0] if isinstance(args, list) and args else ""

    attach_sql = f"""
        ATTACH '{command}' AS {name} (
            TYPE mcp,
            TRANSPORT '{transport}',
            ARGS '{args_json}'
        )
    """

    conn.execute(attach_sql)


def list_resources(
    conn: duckdb.DuckDBPyConnection,
    server: str
) -> list[dict[str, Any]]:
    """
    List available resources from an MCP server.
    
    Args:
        conn: DuckDB connection
        server: Server name identifier
        
    Returns:
        List of resource information dictionaries
    """
    query = f"SELECT mcp_list_resources('{server}')"
    result = execute_query(conn, query)

    # Parse JSON result if needed
    if result and len(result) > 0:
        first_value = list(result[0].values())[0]
        if isinstance(first_value, str):
            try:
                return json.loads(first_value)
            except json.JSONDecodeError:
                return result

    return result


def get_resource(
    conn: duckdb.DuckDBPyConnection,
    server: str,
    uri: str
) -> Any:
    """
    Get specific resource content from an MCP server.
    
    Args:
        conn: DuckDB connection
        server: Server name identifier
        uri: Resource URI
        
    Returns:
        Resource content
    """
    query = f"SELECT mcp_get_resource('{server}', '{uri}')"
    result = execute_query(conn, query)

    if result and len(result) > 0:
        return list(result[0].values())[0]

    return None


def call_tool(
    conn: duckdb.DuckDBPyConnection,
    server: str,
    tool: str,
    args: dict[str, Any] | None = None
) -> Any:
    """
    Call a tool on a remote MCP server.
    
    Args:
        conn: DuckDB connection
        server: Server name identifier
        tool: Tool name
        args: Tool arguments as dictionary
        
    Returns:
        Tool execution result
    """
    args_json = json.dumps(args or {})
    query = f"SELECT mcp_call_tool('{server}', '{tool}', '{args_json}')"
    result = execute_query(conn, query)

    if result and len(result) > 0:
        return list(result[0].values())[0]

    return None


def list_tools(
    conn: duckdb.DuckDBPyConnection,
    server: str
) -> list[dict[str, Any]]:
    """
    List available tools from an MCP server.
    
    Args:
        conn: DuckDB connection
        server: Server name identifier
        
    Returns:
        List of tool information dictionaries
    """
    query = f"SELECT mcp_list_tools('{server}')"
    result = execute_query(conn, query)

    if result and len(result) > 0:
        first_value = list(result[0].values())[0]
        if isinstance(first_value, str):
            try:
                return json.loads(first_value)
            except json.JSONDecodeError:
                return result

    return result


def query_resource(
    conn: duckdb.DuckDBPyConnection,
    server: str,
    uri: str,
    reader: str = "csv"
) -> list[dict[str, Any]]:
    """
    Query remote data resource using appropriate reader.
    
    Args:
        conn: DuckDB connection
        server: Server name identifier
        uri: Resource URI
        reader: Reader function (csv, json, parquet)
        
    Returns:
        Query results as list of dictionaries
    """
    mcp_uri = f"mcp://{server}/{uri}"
    query = f"SELECT * FROM read_{reader}('{mcp_uri}')"
    return execute_query(conn, query)


def list_prompts(
    conn: duckdb.DuckDBPyConnection,
    server: str
) -> list[dict[str, Any]]:
    """
    List available prompts from an MCP server.
    
    Args:
        conn: DuckDB connection
        server: Server name identifier
        
    Returns:
        List of prompt information dictionaries
    """
    query = f"SELECT mcp_list_prompts('{server}')"
    result = execute_query(conn, query)

    if result and len(result) > 0:
        first_value = list(result[0].values())[0]
        if isinstance(first_value, str):
            try:
                return json.loads(first_value)
            except json.JSONDecodeError:
                return result

    return result


def get_prompt(
    conn: duckdb.DuckDBPyConnection,
    server: str,
    prompt_name: str,
    args: dict[str, Any] | None = None
) -> Any:
    """
    Get a prompt from an MCP server.
    
    Args:
        conn: DuckDB connection
        server: Server name identifier
        prompt_name: Name of the prompt
        args: Prompt arguments as dictionary
        
    Returns:
        Prompt content
    """
    args_json = json.dumps(args or {})
    query = f"SELECT mcp_get_prompt('{server}', '{prompt_name}', '{args_json}')"
    result = execute_query(conn, query)

    if result and len(result) > 0:
        return list(result[0].values())[0]

    return None

