"""
CLI interface for mcp-man - DuckDB MCP Manager
"""

import json
import time
from pathlib import Path
from typing import Literal, Optional

import typer
from core.config import Config, get_config
from core.logging import setup_logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import ConnectionConfig, GatewayConfig
from .gateway import MCPGateway

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


def cli_main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    cli_main()
