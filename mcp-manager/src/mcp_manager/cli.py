"""
CLI interface for mcp-man - DuckDB MCP Manager
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal, Optional

import typer
from core.config import Config, get_config
from core.logging import setup_logging
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .agent.templates import AgentMarkdownGenerator
from .client import call_tool
from .config import ConnectionConfig, GatewayConfig
from .discovery.scanner import AsyncToolScanner, ToolRegistry
from .gateway import MCPGateway
from .tools.search import ToolSearcher

app = typer.Typer(
    name="mcp-man",
    help="DuckDB MCP Gateway - Centralized gateway for managing multiple MCP server connections",
    add_completion=False,
)

# Sub-apps
gateway_app = typer.Typer(name="gateway", help="Manage MCP gateway")
security_app = typer.Typer(name="security", help="Manage security settings")

app.add_typer(gateway_app)
app.add_typer(security_app)

console = Console()


def load_gateway_config(config: Optional[Config] = None) -> GatewayConfig:
    """Load gateway configuration from file."""
    if config is None:
        config = get_config()
    config_data = config.get_tool_config("mcp-manager")
    if config_data:
        return GatewayConfig(**config_data)
    return GatewayConfig()


def save_gateway_config(config: Config, gateway_config: GatewayConfig) -> None:
    """Save gateway configuration to file."""
    config.save_tool_config("mcp-manager", gateway_config.model_dump())


def save_connection(config: Config, name: str, transport: str, args: list[str]) -> None:
    """Save connection to configuration."""
    gateway_config = load_gateway_config(config)

    # Check if connection already exists
    for conn in gateway_config.connections:
        if conn.name == name:
            conn.transport = transport
            conn.args = args
            save_gateway_config(config, gateway_config)
            return

    # Add new connection
    gateway_config.connections.append(
        ConnectionConfig(name=name, transport=transport, args=args)
    )
    save_gateway_config(config, gateway_config)


@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
) -> None:
    """mcp-man - DuckDB MCP Gateway."""
    if debug:
        setup_logging("DEBUG")
    else:
        setup_logging(log_level)


@app.command()
def init(
    name: str = typer.Argument("default", help="Database name"),
    path: Optional[Path] = typer.Option(None, "--path", help="Path to database file"),
) -> None:
    """Initialize a new DuckDB database with MCP support."""
    config = get_config()

    if not path:
        db_dir = config.config_dir / "databases"
        db_dir.mkdir(parents=True, exist_ok=True)
        path = db_dir / f"{name}.duckdb"

    try:
        # Create gateway config
        gateway_config = GatewayConfig()
        gateway_config.databases[name] = path

        # Initialize gateway and get connection
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        gateway.get_database_connection(name, path)

        console.print(f"[green]✓[/green] Database initialized: {path}")
        console.print("[green]✓[/green] MCP extension loaded")
        console.print(f"[dim]Database '{name}' ready for MCP operations[/dim]")

        gateway.shutdown()

    except Exception as e:
        console.print(f"[red]✗[/red] Initialization failed: {e}")
        raise typer.Exit(1)


@gateway_app.command("status")
def gateway_status() -> None:
    """Show gateway status and all connections."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")

        # Get connection status
        summary = gateway.registry.get_status_summary()

        # Create status table
        table = Table(title="MCP Gateway Status")
        table.add_column("Connection", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Transport", style="blue")

        for conn_info in gateway.registry.list_all():
            table.add_row(
                conn_info.name,
                conn_info.status.value,
                conn_info.transport
            )

        console.print(table)
        console.print(f"\n[bold]Summary:[/bold] {dict(summary)}")

        gateway.shutdown()

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get status: {e}")
        raise typer.Exit(1)


@app.command()
def connect(
    name: str = typer.Argument(..., help="Connection name"),
    transport: Literal["stdio", "tcp", "websocket"] = typer.Option(
        "stdio",
        "--transport",
        help="Transport type"
    ),
    args: list[str] = typer.Option(
        [],
        "--args",
        help="Transport arguments (can be specified multiple times)"
    ),
    database: str = typer.Option("default", "--database", help="Database to use for connection"),
) -> None:
    """Connect to a remote MCP server."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")

        if not args and transport == "stdio":
            console.print("[red]✗[/red] stdio transport requires command arguments")
            console.print("Example: mcp-man connect myserver --transport stdio --args python3 --args server.py")
            raise typer.Exit(1)

        gateway.connect_server(name, transport, args, database=database)

        console.print(f"[green]✓[/green] Connected to MCP server: {name}")
        console.print(f"[dim]Transport: {transport}, Args: {args}[/dim]")

        # Save connection to config
        save_connection(config, name, transport, args)

        gateway.shutdown()

    except Exception as e:
        console.print(f"[red]✗[/red] Connection failed: {e}")
        raise typer.Exit(1)


@app.command()
def disconnect(
    name: str = typer.Argument(..., help="Connection name"),
) -> None:
    """Disconnect from an MCP server."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")

        gateway.disconnect_server(name)

        console.print(f"[green]✓[/green] Disconnected from: {name}")

        gateway.shutdown()

    except Exception as e:
        console.print(f"[red]✗[/red] Disconnect failed: {e}")
        raise typer.Exit(1)


@app.command()
def connections() -> None:
    """List all MCP connections."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")

        table = Table(title="MCP Connections")
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Transport", style="blue")
        table.add_column("Connected At", style="green")

        for conn_info in gateway.registry.list_all():
            connected_at = conn_info.connected_at.strftime("%Y-%m-%d %H:%M:%S") if conn_info.connected_at else "N/A"
            table.add_row(
                conn_info.name,
                conn_info.status.value,
                conn_info.transport,
                connected_at
            )

        console.print(table)

        gateway.shutdown()

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list connections: {e}")
        raise typer.Exit(1)


@app.command()
def serve(
    database: str = typer.Argument("default", help="Database name"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind MCP server to"),
    port: int = typer.Option(9999, "--port", help="Port to bind MCP server to"),
    transport: Literal["stdio", "tcp", "websocket"] = typer.Option(
        "stdio",
        "--transport",
        help="Server transport type"
    ),
) -> None:
    """Start MCP server to expose DuckDB database."""
    config = get_config()

    try:
        from .server import start_server, stop_server

        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")

        conn = gateway.get_database_connection(database)
        start_server(conn, host, port, transport)

        console.print("[green]✓[/green] MCP server started")
        console.print(f"[dim]Database: {database}")
        console.print(f"[dim]Listening on {host}:{port} ({transport})[/dim]")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]")

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_server(conn)
            console.print("\n[green]✓[/green] Server stopped")

        gateway.shutdown()

    except Exception as e:
        console.print(f"[red]✗[/red] Server failed: {e}")
        raise typer.Exit(1)


@app.command()
def query(
    query_string: str = typer.Argument(..., help="SQL query string"),
    database: str = typer.Option("default", "--database", help="Database to query"),
    format: Literal["table", "json"] = typer.Option("table", "--format", help="Output format"),
) -> None:
    """Execute SQL query across all connected MCP servers."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")

        result = gateway.query_across_servers(query_string, database)

        if format == "json":
            console.print(json.dumps(result, indent=2))
        else:
            if result:
                table = Table(title=f"Query Results ({len(result)} rows)")

                # Add columns
                if result:
                    for col in result[0].keys():
                        table.add_column(col, style="cyan")

                    # Add rows
                    for row in result:
                        table.add_row(*[str(v) for v in row.values()])

                console.print(table)
            else:
                console.print("[yellow]Query returned no results[/yellow]")

        gateway.shutdown()

    except Exception as e:
        console.print(f"[red]✗[/red] Query failed: {e}")
        raise typer.Exit(1)


@app.command()
def resources(
    server: Optional[str] = typer.Option(None, "--server", help="Filter resources by server name"),
) -> None:
    """List resources from all connected MCP servers."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")

        all_resources = gateway.list_all_resources()

        for server_name, resources_list in all_resources.items():
            if server and server != server_name:
                continue

            panel = Panel(
                json.dumps(resources_list, indent=2),
                title=f"[bold cyan]{server_name}[/bold cyan]",
                border_style="blue"
            )
            console.print(panel)

        gateway.shutdown()

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list resources: {e}")
        raise typer.Exit(1)


@app.command()
def publish(
    table: str = typer.Argument(..., help="Table name"),
    uri: str = typer.Option(..., "--uri", help="Resource URI (e.g., data://tables/users)"),
    format: Literal["json", "csv", "parquet"] = typer.Option("json", "--format", help="Output format"),
    database: str = typer.Option("default", "--database", help="Database containing the table"),
) -> None:
    """Publish a table as an MCP resource."""
    config = get_config()

    try:
        from .server import publish_table

        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")

        conn = gateway.get_database_connection(database)
        publish_table(conn, table, uri, format)

        console.print(f"[green]✓[/green] Published table: {table}")
        console.print(f"[dim]URI: {uri}, Format: {format}[/dim]")

        gateway.shutdown()

    except Exception as e:
        console.print(f"[red]✗[/red] Publish failed: {e}")
        raise typer.Exit(1)


@security_app.command("add-command")
def security_add_command(
    command_path: str = typer.Argument(..., help="Command path to add to allowlist"),
) -> None:
    """Add command to allowlist."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)

        if command_path not in gateway_config.security.allowed_commands:
            gateway_config.security.allowed_commands.append(command_path)
            save_gateway_config(config, gateway_config)
            console.print(f"[green]✓[/green] Added command to allowlist: {command_path}")
        else:
            console.print(f"[yellow]Command already in allowlist: {command_path}[/yellow]")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to add command: {e}")
        raise typer.Exit(1)


@security_app.command("add-url")
def security_add_url(
    url_prefix: str = typer.Argument(..., help="URL prefix to add to allowlist"),
) -> None:
    """Add URL prefix to allowlist."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)

        if url_prefix not in gateway_config.security.allowed_urls:
            gateway_config.security.allowed_urls.append(url_prefix)
            save_gateway_config(config, gateway_config)
            console.print(f"[green]✓[/green] Added URL prefix to allowlist: {url_prefix}")
        else:
            console.print(f"[yellow]URL prefix already in allowlist: {url_prefix}[/yellow]")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to add URL: {e}")
        raise typer.Exit(1)


@security_app.command("list")
def security_list() -> None:
    """List security settings."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)

        console.print("[bold]Security Configuration:[/bold]")
        console.print(f"Lock servers: {gateway_config.security.lock_servers}")
        console.print(f"Log level: {gateway_config.security.log_level}")

        console.print("\n[bold]Allowed Commands:[/bold]")
        for cmd in gateway_config.security.allowed_commands:
            console.print(f"  • {cmd}")

        console.print("\n[bold]Allowed URLs:[/bold]")
        for url in gateway_config.security.allowed_urls:
            console.print(f"  • {url}")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list security: {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    method: str = typer.Option("bm25", "--method", help="Search method: bm25, regex, exact, semantic"),
    limit: int = typer.Option(5, "--limit", help="Max results"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search for MCP tools using FTS or pattern matching."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        # Get database connection
        conn = gateway.get_database_connection("default")
        
        try:
            # Initialize searcher
            searcher = ToolSearcher(conn)
            
            # Execute search based on method
            if method == "bm25":
                results = searcher.search_bm25(query, limit)
            elif method == "regex":
                results = searcher.search_regex(query, limit)
            elif method == "exact":
                results = searcher.search_exact(query, limit)
            elif method == "semantic":
                results = searcher.search_semantic(query, limit)
            else:
                console.print(f"[red]✗[/red] Unknown search method: {method}")
                raise typer.Exit(1)
            
            if not results:
                console.print(f"[yellow]No tools found matching '{query}'[/yellow]")
                gateway.shutdown()
                return
            
            if json_mode:
                output = [
                    {
                        "server": r.server,
                        "tool": r.tool,
                        "description": r.description,
                        "required_params": r.required_params,
                        "score": r.score
                    }
                    for r in results
                ]
                console.print(json.dumps(output, indent=2))
            else:
                # Create results table
                table = Table(title=f"Search Results ({len(results)} found)")
                table.add_column("Server", style="cyan")
                table.add_column("Tool", style="green")
                table.add_column("Description", style="white")
                table.add_column("Required Params", style="yellow")
                table.add_column("Score", style="magenta")
                
                for result in results:
                    params = ", ".join(result.required_params) if result.required_params else "—"
                    table.add_row(
                        result.server,
                        result.tool,
                        result.description[:50] + "..." if len(result.description) > 50 else result.description,
                        params,
                        f"{result.score:.2f}"
                    )
                
                console.print(table)
            
            logger.info(f"Search for '{query}' returned {len(results)} results")
            
        finally:
            gateway.shutdown()
            
    except ValueError as e:
        console.print(f"[red]✗[/red] Search error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to search tools: {e}")
        logger.exception("Tool search failed")
        raise typer.Exit(1)


@app.command()
def tools(
    server: Optional[str] = typer.Argument(None, help="Server name (optional)"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List MCP tools from servers."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        # Get database connection
        conn = gateway.get_database_connection("default")
        
        try:
            # Query tools from database
            if server:
                query = "SELECT server_name, tool_name, description, required_params FROM mcp_tools WHERE server_name = ? AND enabled = true ORDER BY tool_name"
                results = conn.execute(query, [server]).fetchall()
                
                if not results:
                    console.print(f"[yellow]No tools found for server '{server}'[/yellow]")
                    gateway.shutdown()
                    return
            else:
                query = "SELECT server_name, tool_name, description, required_params FROM mcp_tools WHERE enabled = true ORDER BY server_name, tool_name"
                results = conn.execute(query).fetchall()
                
                if not results:
                    console.print("[yellow]No tools found in registry[/yellow]")
                    gateway.shutdown()
                    return
            
            if json_mode:
                output = [
                    {
                        "server": row[0],
                        "tool": row[1],
                        "description": row[2],
                        "required_params": row[3] if row[3] else []
                    }
                    for row in results
                ]
                console.print(json.dumps(output, indent=2))
            else:
                # Group by server if not filtered
                if not server:
                    servers_dict: dict[str, list] = {}
                    for row in results:
                        server_name = row[0]
                        if server_name not in servers_dict:
                            servers_dict[server_name] = []
                        servers_dict[server_name].append(row)
                    
                    for srv_name in sorted(servers_dict.keys()):
                        table = Table(title=f"[bold cyan]{srv_name}[/bold cyan] ({len(servers_dict[srv_name])} tools)")
                        table.add_column("Tool", style="green")
                        table.add_column("Description", style="white")
                        table.add_column("Required Params", style="yellow")
                        
                        for row in servers_dict[srv_name]:
                            params = ", ".join(row[3]) if row[3] else "—"
                            table.add_row(
                                row[1],
                                row[2][:50] + "..." if len(row[2]) > 50 else row[2],
                                params
                            )
                        
                        console.print(table)
                        console.print()
                else:
                    # Single server view
                    table = Table(title=f"[bold cyan]{server}[/bold cyan] ({len(results)} tools)")
                    table.add_column("Tool", style="green")
                    table.add_column("Description", style="white")
                    table.add_column("Required Params", style="yellow")
                    
                    for row in results:
                        params = ", ".join(row[3]) if row[3] else "—"
                        table.add_row(
                            row[1],
                            row[2][:50] + "..." if len(row[2]) > 50 else row[2],
                            params
                        )
                    
                    console.print(table)
            
            logger.info(f"Listed {len(results)} tools")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list tools: {e}")
        logger.exception("Tool listing failed")
        raise typer.Exit(1)


@app.command()
def inspect(
    server: str = typer.Argument(..., help="Server name"),
    tool: str = typer.Argument(..., help="Tool name"),
    example: bool = typer.Option(False, "--example", help="Show example call"),
) -> None:
    """Show detailed info about a tool."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        # Get database connection
        conn = gateway.get_database_connection("default")
        
        try:
            # Query tool details
            query = """
                SELECT tool_name, description, input_schema, required_params 
                FROM mcp_tools 
                WHERE server_name = ? AND tool_name = ? AND enabled = true
            """
            result = conn.execute(query, [server, tool]).fetchone()
            
            if not result:
                console.print(f"[red]✗[/red] Tool '{tool}' not found on server '{server}'")
                raise typer.Exit(1)
            
            tool_name, description, input_schema, required_params = result
            
            # Display tool information
            console.print(Panel(
                f"[bold cyan]{tool_name}[/bold cyan]\n{description}",
                title=f"[bold]{server}/{tool_name}[/bold]",
                border_style="blue"
            ))
            
            # Display schema
            if input_schema:
                try:
                    schema_dict = json.loads(input_schema) if isinstance(input_schema, str) else input_schema
                    console.print("\n[bold]Input Schema:[/bold]")
                    syntax = Syntax(
                        json.dumps(schema_dict, indent=2),
                        "json",
                        theme="monokai",
                        line_numbers=False
                    )
                    console.print(syntax)
                except Exception as e:
                    console.print(f"[yellow]Could not parse schema: {e}[/yellow]")
            
            # Display required parameters
            if required_params:
                console.print(f"\n[bold]Required Parameters:[/bold]")
                params_list = required_params if isinstance(required_params, list) else [required_params]
                for param in params_list:
                    console.print(f"  • {param}")
            
            # Show example if requested
            if example:
                console.print("\n[bold]Example Call:[/bold]")
                example_args = {}
                if required_params:
                    params_list = required_params if isinstance(required_params, list) else [required_params]
                    for param in params_list:
                        example_args[param] = "<value>"
                
                example_json = json.dumps(example_args, indent=2)
                console.print(f"[dim]mcp-man call {server} {tool_name} '{example_json}'[/dim]")
            
            logger.info(f"Inspected tool {server}/{tool_name}")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to inspect tool: {e}")
        logger.exception("Tool inspection failed")
        raise typer.Exit(1)


@app.command()
def call(
    server: str = typer.Argument(..., help="Server name"),
    tool: str = typer.Argument(..., help="Tool name"),
    arguments: str = typer.Argument("{}", help="JSON arguments"),
    stdin: bool = typer.Option(False, "--stdin", help="Read args from stdin"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Execute a tool on an MCP server."""
    config = get_config()

    try:
        # Parse arguments
        if stdin:
            import sys
            arguments = sys.stdin.read()
        
        try:
            args_dict: dict[str, Any] = json.loads(arguments)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗[/red] Invalid JSON arguments: {e}")
            raise typer.Exit(1)
        
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            # Get database connection and call the tool
            conn = gateway.get_database_connection("default")
            result = call_tool(conn, server, tool, args_dict)
            
            if json_mode:
                console.print(json.dumps(result, indent=2))
            else:
                # Display result
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        pass
                
                if isinstance(result, dict) and "error" in result:
                    console.print(f"[red]✗[/red] Tool execution failed: {result['error']}")
                else:
                    console.print("[green]✓[/green] Tool executed successfully")
                    syntax = Syntax(
                        json.dumps(result, indent=2) if isinstance(result, dict) else str(result),
                        "json",
                        theme="monokai",
                        line_numbers=False
                    )
                    console.print(syntax)
            
            # Log execution
            logger.info(f"Executed tool {server}/{tool} with args {args_dict}")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to call tool: {e}")
        logger.exception("Tool execution failed")
        raise typer.Exit(1)


@app.command()
def refresh() -> None:
    """Refresh tool index from all servers."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            with console.status("[bold green]Scanning all servers..."):
                async def scan_servers() -> int:
                    scanner = AsyncToolScanner()
                    tool_registry = ToolRegistry()
                    total_tools = 0
                    
                    for conn_info in gateway.registry.list_all():
                        try:
                            logger.info(f"Scanning server: {conn_info.name}")
                            tools = await scanner.scan_server(conn_info.name)
                            tool_registry.add_tools(conn_info.name, tools)
                            total_tools += len(tools)
                        except Exception as e:
                            logger.warning(f"Failed to scan {conn_info.name}: {e}")
                    
                    return total_tools
                
                total = asyncio.run(scan_servers())
            
            console.print(f"[green]✓[/green] Tool index refreshed")
            console.print(f"[dim]Discovered {total} tools across all servers[/dim]")
            logger.info(f"Tool index refresh complete: {total} tools discovered")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to refresh tool index: {e}")
        logger.exception("Tool index refresh failed")
        raise typer.Exit(1)


@app.command()
def agent(
    output: Optional[Path] = typer.Option(None, "--output", help="Save to file"),
) -> None:
    """Generate AGENT.md for Claude/Agents."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            with console.status("[bold green]Generating AGENT.md..."):
                # Get all tools from database
                conn = gateway.get_database_connection("default")
                
                query = """
                    SELECT server_name, tool_name, description, required_params
                    FROM mcp_tools
                    WHERE enabled = true
                    ORDER BY server_name, tool_name
                """
                results = conn.execute(query).fetchall()
                
                if not results:
                    console.print("[yellow]No tools found to document[/yellow]")
                    gateway.shutdown()
                    return
                
                # Generate markdown
                generator = AgentMarkdownGenerator()
                
                # Group tools by server
                servers_tools: dict[str, list] = {}
                for row in results:
                    server_name = row[0]
                    if server_name not in servers_tools:
                        servers_tools[server_name] = []
                    servers_tools[server_name].append({
                        "name": row[1],
                        "description": row[2],
                        "params": row[3] if row[3] else []
                    })
                
                # Build markdown content
                markdown_content = "# MCP Tools Documentation\n\n"
                markdown_content += f"Generated: {json.dumps({'timestamp': str(gateway_config.databases)}, indent=2)}\n\n"
                markdown_content += "## Available Servers and Tools\n\n"
                
                for server_name in sorted(servers_tools.keys()):
                    tools = servers_tools[server_name]
                    markdown_content += f"### {server_name}\n\n"
                    markdown_content += f"**Available Tools:** {len(tools)}\n\n"
                    
                    for tool in tools:
                        params = ", ".join(tool["params"]) if tool["params"] else "none"
                        markdown_content += f"- **{tool['name']}**: {tool['description']}\n"
                        markdown_content += f"  - Required parameters: {params}\n\n"
                
                # Save or print
                if output:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(markdown_content)
                    console.print(f"[green]✓[/green] AGENT.md saved to: {output}")
                    logger.info(f"AGENT.md generated and saved to {output}")
                else:
                    console.print(markdown_content)
                    logger.info("AGENT.md generated and displayed")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to generate AGENT.md: {e}")
        logger.exception("AGENT.md generation failed")
        raise typer.Exit(1)


def cli_main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    cli_main()
