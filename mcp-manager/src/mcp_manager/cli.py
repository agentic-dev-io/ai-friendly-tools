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
from .tools.execution import ToolExecutor, ExecutionContext
from .tools.validation import ToolValidator, ToolErrorSuggester

# AI feature imports
from .ai.nlp import NaturalLanguageProcessor, IntentClassifier
from .ai.examples import ExampleGenerator, UsageExample
from .ai.llm_export import LLMExporter, ExportFormat, ExportConfig
from .ai.suggestions import ToolSuggester, SuggestionContext
from .ai.workflows import (
    Workflow,
    WorkflowStep,
    WorkflowRunner,
    WorkflowBuilder,
    WorkflowRegistry,
    WorkflowStatus,
)

app = typer.Typer(
    name="mcp-man",
    help="DuckDB MCP Gateway - Centralized gateway for managing multiple MCP server connections",
    add_completion=False,
)

# Sub-apps
gateway_app = typer.Typer(name="gateway", help="Manage MCP gateway")
security_app = typer.Typer(name="security", help="Manage security settings")
workflow_app = typer.Typer(name="workflow", help="Manage tool pipelines/workflows")

app.add_typer(gateway_app)
app.add_typer(security_app)
app.add_typer(workflow_app)

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
    semantic: bool = typer.Option(False, "--semantic", help="Use semantic/AI-powered search"),
    intent: bool = typer.Option(False, "--intent", help="Classify user intent and optimize search"),
) -> None:
    """Search for MCP tools using FTS or pattern matching.
    
    Use --semantic for AI-powered semantic search.
    Use --intent to classify query intent and optimize search.
    """
    # Override method if semantic flag is set
    if semantic:
        method = "semantic"
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
    example: bool = typer.Option(False, "--example", help="Show basic example call"),
    examples_flag: bool = typer.Option(False, "--examples", help="Show AI-generated examples with realistic values"),
) -> None:
    """Show detailed info about a tool.
    
    Use --example for basic placeholder example.
    Use --examples for AI-generated examples with realistic values.
    """
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
                console.print(f"[red]x[/red] Tool '{tool}' not found on server '{server}'")
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
                    console.print(f"  . {param}")
            
            # Show basic example if requested
            if example:
                console.print("\n[bold]Example Call:[/bold]")
                example_args = {}
                if required_params:
                    params_list = required_params if isinstance(required_params, list) else [required_params]
                    for param in params_list:
                        example_args[param] = "<value>"
                
                example_json = json.dumps(example_args, indent=2)
                console.print(f"[dim]mcp-man call {server} {tool_name} '{example_json}'[/dim]")
            
            # Show AI-generated examples if requested
            if examples_flag:
                from .tools.schema import ToolSchema
                tool_schema = ToolSchema(
                    server_name=server,
                    tool_name=tool_name,
                    description=description,
                    input_schema=json.loads(input_schema) if isinstance(input_schema, str) else input_schema,
                    required_params=required_params if isinstance(required_params, list) else [],
                )
                
                generator = ExampleGenerator()
                generated_examples = generator.generate_multiple(tool_schema, count=3)
                
                console.print("\n[bold]Generated Examples:[/bold]")
                for i, ex in enumerate(generated_examples, 1):
                    console.print(f"\n[cyan]Example {i}:[/cyan] {ex.description}")
                    console.print(f"[dim]{ex.to_cli_command()}[/dim]")
            
            logger.info(f"Inspected tool {server}/{tool_name}")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]x[/red] Failed to inspect tool: {e}")
        logger.exception("Tool inspection failed")
        raise typer.Exit(1)


@app.command()
def call(
    server: str = typer.Argument(..., help="Server name"),
    tool: str = typer.Argument(..., help="Tool name"),
    arguments: str = typer.Argument("{}", help="JSON arguments"),
    stdin: bool = typer.Option(False, "--stdin", help="Read args from stdin"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate arguments"),
    show_timing: bool = typer.Option(False, "--timing", help="Show execution timing"),
    suggest_next: bool = typer.Option(False, "--suggest-next", help="Show suggested next tools after execution"),
) -> None:
    """Execute a tool on an MCP server with validation and error handling.
    
    Use --suggest-next to get recommendations for follow-up tools.
    """
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
            conn = gateway.get_database_connection("default")
            
            # Get tool schema from database
            query = """
                SELECT tool_name, description, input_schema, required_params
                FROM mcp_tools
                WHERE server_name = ? AND tool_name = ? AND enabled = true
            """
            tool_row = conn.execute(query, [server, tool]).fetchone()
            
            if not tool_row:
                console.print(f"[red]✗[/red] Tool '{tool}' not found on server '{server}'")
                raise typer.Exit(1)
            
            tool_name, description, input_schema, required_params = tool_row
            
            # Create tool schema object
            from .tools.schema import ToolSchema
            tool_schema = ToolSchema(
                server_name=server,
                tool_name=tool_name,
                description=description,
                input_schema=(
                    json.loads(input_schema) if isinstance(input_schema, str) else input_schema
                ),
                required_params=required_params if isinstance(required_params, list) else [],
            )
            
            # Create execution context
            exec_context = ExecutionContext(
                server=server,
                tool=tool_schema,
                arguments=args_dict,
            )
            
            # Execute with validation
            executor = ToolExecutor()
            exec_result = executor.execute(
                conn, exec_context, validate=validate, track_history=True
            )
            
            # Display result
            if json_mode:
                console.print(json.dumps(exec_result.to_dict(), indent=2))
            else:
                if exec_result.success:
                    console.print("[green]✓[/green] Tool executed successfully")
                    
                    # Display result
                    if isinstance(exec_result.result, str):
                        try:
                            result_obj = json.loads(exec_result.result)
                        except json.JSONDecodeError:
                            result_obj = exec_result.result
                    else:
                        result_obj = exec_result.result
                    
                    syntax = Syntax(
                        json.dumps(result_obj, indent=2) if isinstance(result_obj, (dict, list)) else str(result_obj),
                        "json",
                        theme="monokai",
                        line_numbers=False
                    )
                    console.print(syntax)
                    
                    if show_timing:
                        console.print(f"[dim]Execution time: {exec_result.execution_time_ms}ms[/dim]")
                    
                    # Show suggested next tools if requested
                    if suggest_next:
                        suggester = ToolSuggester(conn)
                        suggestions = suggester.suggest_next(server, tool, limit=3)
                        if suggestions:
                            console.print("\n[bold]Suggested next tools:[/bold]")
                            for s in suggestions:
                                console.print(f"  . [cyan]{s.tool_name}[/cyan] - {s.description[:40]}... [{s.confidence:.0%}]")
                    
                else:
                    console.print(f"[red]x[/red] Tool execution failed")
                    
                    # Show validation errors if any
                    if exec_result.validation_errors and not exec_result.validation_errors.is_valid:
                        console.print("\n[yellow]Validation Errors:[/yellow]")
                        for error in exec_result.validation_errors.errors:
                            console.print(f"  {error.parameter}: {error.message}")
                    
                    # Show execution error
                    if exec_result.error:
                        console.print(f"[dim]Error: {exec_result.error}[/dim]")
                    
                    # Show suggestions
                    if exec_result.suggestions:
                        console.print("\n[yellow]Similar tools found:[/yellow]")
                        for suggestion_tool, score in exec_result.suggestions[:3]:
                            console.print(f"  • {suggestion_tool.tool_name} (similarity: {score:.1%})")
                            if suggestion_tool.description:
                                console.print(f"    {suggestion_tool.description}")
                    
                    raise typer.Exit(1)
            
            logger.info(f"Executed tool {server}/{tool} with args {args_dict}")
            
        finally:
            gateway.shutdown()
            
    except typer.Exit:
        raise
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


@app.command()
def history(
    server: Optional[str] = typer.Option(None, "--server", help="Filter by server name"),
    tool: Optional[str] = typer.Option(None, "--tool", help="Filter by tool name"),
    limit: int = typer.Option(20, "--limit", help="Maximum number of records to show"),
    success_only: bool = typer.Option(False, "--success", help="Show only successful executions"),
    failures_only: bool = typer.Option(False, "--failures", help="Show only failed executions"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show tool execution history."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            # Build query
            query = "SELECT id, server_name, tool_name, arguments, result, success, duration_ms, timestamp FROM mcp_tool_history WHERE 1=1"
            params: list[Any] = []
            
            if server:
                query += " AND server_name = ?"
                params.append(server)
            
            if tool:
                query += " AND tool_name = ?"
                params.append(tool)
            
            if success_only:
                query += " AND success = true"
            elif failures_only:
                query += " AND success = false"
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            results = conn.execute(query, params).fetchall()
            
            if not results:
                console.print("[yellow]No execution history found[/yellow]")
                return
            
            if json_output:
                history_list = []
                for row in results:
                    history_list.append({
                        "id": row[0],
                        "server": row[1],
                        "tool": row[2],
                        "arguments": json.loads(row[3]) if row[3] else None,
                        "result": json.loads(row[4]) if row[4] else None,
                        "success": row[5],
                        "duration_ms": row[6],
                        "timestamp": row[7],
                    })
                console.print(json.dumps(history_list, indent=2))
            else:
                # Display as table
                table = Table(title=f"Tool Execution History (Latest {len(results)})")
                table.add_column("ID", style="cyan")
                table.add_column("Server", style="green")
                table.add_column("Tool", style="blue")
                table.add_column("Success", style="yellow")
                table.add_column("Duration", style="magenta")
                table.add_column("Timestamp", style="dim")
                
                for row in results:
                    exec_id, server_name, tool_name, arguments, result, success, duration_ms, timestamp = row
                    status = "[green]✓[/green]" if success else "[red]✗[/red]"
                    duration_str = f"{duration_ms}ms" if duration_ms else "N/A"
                    table.add_row(
                        str(exec_id),
                        server_name or "",
                        tool_name or "",
                        status,
                        duration_str,
                        timestamp or ""
                    )
                
                console.print(table)
            
            logger.info(f"Displayed {len(results)} execution history records")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to retrieve history: {e}")
        logger.exception("History retrieval failed")
        raise typer.Exit(1)


# =============================================================================
# NEW AI FEATURE COMMANDS
# =============================================================================


@app.command()
def ask(
    query: str = typer.Argument(..., help="Natural language query"),
    limit: int = typer.Option(5, "--limit", help="Max results"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search tools using natural language query.
    
    Uses NLP to understand your intent and find the most relevant tools.
    
    Examples:
        mcp-man ask "find tools to read files"
        mcp-man ask "how can I query a database"
        mcp-man ask "tools for API requests"
    """
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            # Use NLP processor
            nlp = NaturalLanguageProcessor(conn)
            result = nlp.process(query)
            
            if json_mode:
                output = {
                    "query": query,
                    "intent": result.query.intent.name,
                    "tools": [
                        {
                            "server": t.server,
                            "tool": t.tool,
                            "description": t.description,
                            "score": t.score,
                        }
                        for t in result.tools
                    ],
                    "suggestions": result.suggestions,
                }
                console.print(json.dumps(output, indent=2))
            else:
                # Show intent
                console.print(f"[dim]Intent: {result.query.intent.name}[/dim]")
                
                if result.tools:
                    table = Table(title=f"Results for: \"{query}\"")
                    table.add_column("Server", style="cyan")
                    table.add_column("Tool", style="green")
                    table.add_column("Description", style="white")
                    table.add_column("Relevance", style="magenta")
                    
                    for tool in result.tools[:limit]:
                        table.add_row(
                            tool.server,
                            tool.tool,
                            tool.description[:50] + "..." if len(tool.description) > 50 else tool.description,
                            f"{tool.score:.0%}",
                        )
                    
                    console.print(table)
                else:
                    console.print("[yellow]No tools found matching your query[/yellow]")
                
                # Show suggestions
                if result.suggestions:
                    console.print("\n[bold]Suggestions:[/bold]")
                    for suggestion in result.suggestions:
                        console.print(f"  [dim]{suggestion}[/dim]")
            
            logger.info(f"NLP search for '{query}' completed")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]x[/red] Failed to process query: {e}")
        logger.exception("NLP query failed")
        raise typer.Exit(1)


@app.command()
def examples(
    server: str = typer.Argument(..., help="Server name"),
    tool: str = typer.Argument(..., help="Tool name"),
    count: int = typer.Option(3, "--count", "-n", help="Number of examples to generate"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show auto-generated usage examples for a tool.
    
    Generates realistic example calls based on the tool's schema.
    """
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            # Get tool schema
            query_str = """
                SELECT tool_name, description, input_schema, required_params
                FROM mcp_tools
                WHERE server_name = ? AND tool_name = ? AND enabled = true
            """
            result = conn.execute(query_str, [server, tool]).fetchone()
            
            if not result:
                console.print(f"[red]x[/red] Tool '{tool}' not found on server '{server}'")
                raise typer.Exit(1)
            
            tool_name, description, input_schema, required_params = result
            
            # Create tool schema object
            from .tools.schema import ToolSchema
            tool_schema = ToolSchema(
                server_name=server,
                tool_name=tool_name,
                description=description,
                input_schema=json.loads(input_schema) if isinstance(input_schema, str) else input_schema,
                required_params=required_params if isinstance(required_params, list) else [],
            )
            
            # Generate examples
            generator = ExampleGenerator()
            generated_examples = generator.generate_multiple(tool_schema, count=count)
            
            if json_mode:
                output = [ex.to_dict() for ex in generated_examples]
                console.print(json.dumps(output, indent=2))
            else:
                console.print(Panel(
                    f"[bold cyan]{tool_name}[/bold cyan]\n{description}",
                    title=f"[bold]Examples for {server}/{tool_name}[/bold]",
                    border_style="blue",
                ))
                
                for i, ex in enumerate(generated_examples, 1):
                    console.print(f"\n[bold]Example {i}:[/bold] {ex.description}")
                    console.print(f"[dim]{ex.to_cli_command()}[/dim]")
                    
                    syntax = Syntax(
                        json.dumps(ex.arguments, indent=2),
                        "json",
                        theme="monokai",
                        line_numbers=False,
                    )
                    console.print(syntax)
            
            logger.info(f"Generated {count} examples for {server}/{tool}")
            
        finally:
            gateway.shutdown()
            
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]x[/red] Failed to generate examples: {e}")
        logger.exception("Example generation failed")
        raise typer.Exit(1)


@app.command()
def export(
    format: str = typer.Argument("json", help="Export format: json, markdown, md, xml, openai, anthropic, yaml"),
    tool_filter: Optional[str] = typer.Option(None, "--tool", help="Filter by tool name pattern"),
    server_filter: Optional[str] = typer.Option(None, "--server", help="Filter by server name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    include_examples: bool = typer.Option(False, "--examples", help="Include generated examples"),
    compact: bool = typer.Option(False, "--compact", help="Compact output (no indentation)"),
) -> None:
    """Export tools documentation in various formats.
    
    Formats:
        json      - JSON format for programmatic use
        markdown  - Markdown documentation
        xml       - XML format
        openai    - OpenAI function calling format
        anthropic - Anthropic tool use format
        yaml      - YAML format
    """
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            # Build query
            query_str = "SELECT server_name, tool_name, description, input_schema, required_params FROM mcp_tools WHERE enabled = true"
            params: list[Any] = []
            
            if server_filter:
                query_str += " AND server_name ILIKE ?"
                params.append(f"%{server_filter}%")
            
            if tool_filter:
                query_str += " AND tool_name ILIKE ?"
                params.append(f"%{tool_filter}%")
            
            query_str += " ORDER BY server_name, tool_name"
            
            results = conn.execute(query_str, params).fetchall()
            
            if not results:
                console.print("[yellow]No tools found to export[/yellow]")
                return
            
            # Create tool schemas
            from .tools.schema import ToolSchema
            tools = [
                ToolSchema(
                    server_name=row[0],
                    tool_name=row[1],
                    description=row[2],
                    input_schema=json.loads(row[3]) if isinstance(row[3], str) else row[3],
                    required_params=row[4] if isinstance(row[4], list) else [],
                )
                for row in results
            ]
            
            # Export
            exporter = LLMExporter()
            export_config = ExportConfig(
                include_examples=include_examples,
                compact=compact,
            )
            
            try:
                export_format = ExportFormat(format.lower())
            except ValueError:
                console.print(f"[red]x[/red] Unknown format: {format}")
                console.print("[dim]Available: json, markdown, md, xml, openai, anthropic, yaml[/dim]")
                raise typer.Exit(1)
            
            content = exporter.export(tools, export_format, export_config)
            
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(content, encoding="utf-8")
                console.print(f"[green].[/green] Exported {len(tools)} tools to: {output}")
            else:
                console.print(content)
            
            logger.info(f"Exported {len(tools)} tools in {format} format")
            
        finally:
            gateway.shutdown()
            
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]x[/red] Export failed: {e}")
        logger.exception("Export failed")
        raise typer.Exit(1)


@app.command()
def suggest(
    server: str = typer.Argument(..., help="Server name"),
    tool: str = typer.Argument(..., help="Current tool name"),
    limit: int = typer.Option(5, "--limit", help="Max suggestions"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Suggest next tools based on usage patterns.
    
    Analyzes execution history to suggest commonly-used follow-up tools.
    """
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            # Get suggestions
            suggester = ToolSuggester(conn)
            suggestions = suggester.suggest_next(server, tool, limit=limit)
            
            if json_mode:
                output = [s.to_dict() for s in suggestions]
                console.print(json.dumps(output, indent=2))
            else:
                if suggestions:
                    console.print(f"[bold]Suggested next tools after {server}/{tool}:[/bold]\n")
                    
                    table = Table()
                    table.add_column("Tool", style="green")
                    table.add_column("Server", style="cyan")
                    table.add_column("Description", style="white")
                    table.add_column("Confidence", style="magenta")
                    
                    for s in suggestions:
                        table.add_row(
                            s.tool_name,
                            s.server_name,
                            s.description[:40] + "..." if len(s.description) > 40 else s.description,
                            f"{s.confidence:.0%}",
                        )
                    
                    console.print(table)
                    
                    if suggestions[0].reason:
                        console.print(f"\n[dim]{suggestions[0].reason}[/dim]")
                else:
                    console.print("[yellow]No suggestions available yet[/yellow]")
                    console.print("[dim]Run more tools to build up usage patterns[/dim]")
            
            logger.info(f"Generated {len(suggestions)} suggestions for {server}/{tool}")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]x[/red] Failed to get suggestions: {e}")
        logger.exception("Suggestion failed")
        raise typer.Exit(1)


# =============================================================================
# WORKFLOW COMMANDS
# =============================================================================


@workflow_app.command("create")
def workflow_create(
    name: str = typer.Argument(..., help="Workflow name"),
    description: str = typer.Option("", "--description", "-d", help="Workflow description"),
    steps_json: Optional[str] = typer.Option(None, "--steps", help="JSON array of steps"),
    from_file: Optional[Path] = typer.Option(None, "--from-file", "-f", help="Load workflow from JSON file"),
) -> None:
    """Create a new workflow/pipeline.
    
    Examples:
        mcp-man workflow create "ETL Pipeline" --description "Extract, transform, load"
        mcp-man workflow create "Data Process" --from-file workflow.json
    """
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            # Build workflow
            if from_file:
                if not from_file.exists():
                    console.print(f"[red]x[/red] File not found: {from_file}")
                    raise typer.Exit(1)
                
                workflow_data = json.loads(from_file.read_text())
                builder = WorkflowBuilder().from_template(workflow_data)
                workflow = builder.build()
                workflow.name = name  # Override name
            elif steps_json:
                steps_data = json.loads(steps_json)
                builder = WorkflowBuilder().name(name).description(description)
                
                for step in steps_data:
                    builder.add_step(
                        step["id"],
                        step.get("name", step["id"]),
                        step["server"],
                        step["tool"],
                        step.get("arguments", {}),
                        depends_on=step.get("depends_on"),
                    )
                
                workflow = builder.build()
            else:
                # Create empty workflow for later editing
                workflow = WorkflowBuilder().name(name).description(description).build()
            
            # Register workflow
            registry = WorkflowRegistry(conn)
            registry.register(workflow)
            
            console.print(f"[green].[/green] Workflow created: {workflow.name}")
            console.print(f"[dim]ID: {workflow.id}[/dim]")
            console.print(f"[dim]Steps: {len(workflow.steps)}[/dim]")
            
            logger.info(f"Created workflow: {workflow.id}")
            
        finally:
            gateway.shutdown()
            
    except typer.Exit:
        raise
    except json.JSONDecodeError as e:
        console.print(f"[red]x[/red] Invalid JSON: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]x[/red] Failed to create workflow: {e}")
        logger.exception("Workflow creation failed")
        raise typer.Exit(1)


@workflow_app.command("run")
def workflow_run(
    workflow_id: str = typer.Argument(..., help="Workflow ID"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show execution plan without running"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run a workflow/pipeline."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            # Get workflow
            registry = WorkflowRegistry(conn)
            workflow = registry.get(workflow_id)
            
            if not workflow:
                console.print(f"[red]x[/red] Workflow not found: {workflow_id}")
                raise typer.Exit(1)
            
            # Run workflow
            runner = WorkflowRunner(conn)
            
            if dry_run:
                console.print(f"[bold]Dry run for workflow: {workflow.name}[/bold]\n")
            
            with console.status(f"[bold green]Running {workflow.name}..."):
                result = runner.run(workflow, dry_run=dry_run)
            
            if json_mode:
                console.print(json.dumps(result.to_dict(), indent=2))
            else:
                # Show results
                status_icon = "[green].[/green]" if result.is_success() else "[red]x[/red]"
                console.print(f"{status_icon} Workflow {result.status.value}: {workflow.name}")
                
                if dry_run:
                    console.print("\n[bold]Planned execution order:[/bold]")
                    for i, step_id in enumerate(result.planned_steps, 1):
                        console.print(f"  {i}. {step_id}")
                else:
                    if result.step_results:
                        console.print("\n[bold]Step Results:[/bold]")
                        
                        table = Table()
                        table.add_column("Step", style="cyan")
                        table.add_column("Status", style="yellow")
                        table.add_column("Duration", style="magenta")
                        
                        for sr in result.step_results:
                            status = "[green].[/green]" if sr.status.value == "success" else "[red]x[/red]"
                            duration = f"{sr.duration_ms}ms" if sr.duration_ms else "N/A"
                            table.add_row(sr.step_id, status, duration)
                        
                        console.print(table)
                
                if result.error:
                    console.print(f"\n[red]Error: {result.error}[/red]")
            
            logger.info(f"Workflow {workflow_id} completed with status: {result.status.value}")
            
        finally:
            gateway.shutdown()
            
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]x[/red] Failed to run workflow: {e}")
        logger.exception("Workflow execution failed")
        raise typer.Exit(1)


@workflow_app.command("list")
def workflow_list(
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List all workflows."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            registry = WorkflowRegistry(conn)
            workflows = registry.list_all()
            
            if not workflows:
                console.print("[yellow]No workflows found[/yellow]")
                return
            
            if json_mode:
                output = [
                    {
                        "id": wf.id,
                        "name": wf.name,
                        "description": wf.description,
                        "steps": len(wf.steps),
                    }
                    for wf in workflows
                ]
                console.print(json.dumps(output, indent=2))
            else:
                table = Table(title="Workflows")
                table.add_column("ID", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Description", style="white")
                table.add_column("Steps", style="magenta")
                
                for wf in workflows:
                    table.add_row(
                        wf.id,
                        wf.name,
                        wf.description[:40] + "..." if len(wf.description) > 40 else wf.description,
                        str(len(wf.steps)),
                    )
                
                console.print(table)
            
            logger.info(f"Listed {len(workflows)} workflows")
            
        finally:
            gateway.shutdown()
            
    except Exception as e:
        console.print(f"[red]x[/red] Failed to list workflows: {e}")
        logger.exception("Workflow listing failed")
        raise typer.Exit(1)


@workflow_app.command("delete")
def workflow_delete(
    workflow_id: str = typer.Argument(..., help="Workflow ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a workflow."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            registry = WorkflowRegistry(conn)
            
            # Check if exists
            workflow = registry.get(workflow_id)
            if not workflow:
                console.print(f"[red]x[/red] Workflow not found: {workflow_id}")
                raise typer.Exit(1)
            
            # Confirm deletion
            if not force:
                confirm = typer.confirm(f"Delete workflow '{workflow.name}'?")
                if not confirm:
                    console.print("[yellow]Cancelled[/yellow]")
                    return
            
            registry.delete(workflow_id)
            
            console.print(f"[green].[/green] Deleted workflow: {workflow.name}")
            logger.info(f"Deleted workflow: {workflow_id}")
            
        finally:
            gateway.shutdown()
            
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]x[/red] Failed to delete workflow: {e}")
        logger.exception("Workflow deletion failed")
        raise typer.Exit(1)


@workflow_app.command("show")
def workflow_show(
    workflow_id: str = typer.Argument(..., help="Workflow ID"),
    json_mode: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show workflow details."""
    config = get_config()

    try:
        gateway_config = load_gateway_config(config)
        gateway = MCPGateway(gateway_config, config.config_dir / "databases")
        
        try:
            conn = gateway.get_database_connection("default")
            
            registry = WorkflowRegistry(conn)
            workflow = registry.get(workflow_id)
            
            if not workflow:
                console.print(f"[red]x[/red] Workflow not found: {workflow_id}")
                raise typer.Exit(1)
            
            if json_mode:
                console.print(workflow.to_json())
            else:
                console.print(Panel(
                    f"[bold cyan]{workflow.name}[/bold cyan]\n{workflow.description}",
                    title=f"[bold]Workflow: {workflow.id}[/bold]",
                    border_style="blue",
                ))
                
                console.print(f"\n[bold]Steps ({len(workflow.steps)}):[/bold]")
                
                for i, step in enumerate(workflow.steps, 1):
                    if isinstance(step, WorkflowStep):
                        deps = f" [dim](depends on: {', '.join(step.depends_on)})[/dim]" if step.depends_on else ""
                        console.print(f"  {i}. [cyan]{step.id}[/cyan]: {step.server}/{step.tool}{deps}")
            
            logger.info(f"Showed workflow: {workflow_id}")
            
        finally:
            gateway.shutdown()
            
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]x[/red] Failed to show workflow: {e}")
        logger.exception("Workflow show failed")
        raise typer.Exit(1)


def cli_main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    cli_main()
