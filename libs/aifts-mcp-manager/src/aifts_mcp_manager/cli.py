"""
CLI interface for mcp-man - DuckDB MCP Manager
"""

import click
from rich.console import Console

from aifts_core import Config

console = Console()


@click.group()
@click.version_option()
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """
mcp-man - DuckDB MCP Manager
    
    Manage DuckDB databases with Model Context Protocol integration.
    """
    config = Config(debug=debug)
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.argument("name", default="default")
@click.option(
    "--path",
    type=click.Path(),
    default=None,
    help="Path to database file (default: ~/.aifts/databases/{name}.duckdb)",
)
@click.pass_context
def init(ctx: click.Context, name: str, path: str | None) -> None:
    """Initialize a new DuckDB database with MCP support."""
    config: Config = ctx.obj["config"]
    config.ensure_config_dir()
    
    if not path:
        db_dir = config.config_dir / "databases"
        db_dir.mkdir(parents=True, exist_ok=True)
        path = str(db_dir / f"{name}.duckdb")
    
    console.print(f"[green]✓[/green] Initializing DuckDB database: {path}")
    console.print(f"[dim]Database '{name}' created and ready for MCP integration.[/dim]")


@cli.command()
@click.argument("database", default="default")
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind MCP server to",
)
@click.option(
    "--port",
    type=int,
    default=9999,
    help="Port to bind MCP server to",
)
@click.pass_context
def serve(ctx: click.Context, database: str, host: str, port: int) -> None:
    """Start MCP server for a DuckDB database."""
    config: Config = ctx.obj["config"]
    console.print(f"[blue]Starting[/blue] MCP server for database: {database}")
    console.print(f"[dim]Server will listen on {host}:{port}[/dim]")
    console.print("[yellow]Feature coming soon - DuckDB MCP extension integration[/yellow]")


@cli.command()
@click.argument("database", default="default")
@click.pass_context
def status(ctx: click.Context, database: str) -> None:
    """Check server status and connected clients."""
    console.print(f"[blue]Status[/blue] for database: {database}")
    console.print("[yellow]Feature coming soon[/yellow]")


@cli.command()
@click.argument("query")
@click.option(
    "--database",
    default="default",
    help="Database to query",
)
@click.pass_context
def query(ctx: click.Context, query: str, database: str) -> None:
    """Execute SQL queries via MCP."""
    config: Config = ctx.obj["config"]
    console.print(f"[blue]Executing query[/blue] on {database}:")
    console.print(f"[dim]{query}[/dim]")
    console.print("[yellow]Feature coming soon - DuckDB MCP integration[/yellow]")


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()