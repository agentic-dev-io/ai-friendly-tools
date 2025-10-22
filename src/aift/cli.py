"""Main CLI application for AIFT using Typer."""

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aift.config import get_config
from aift.logging import setup_logging

app = typer.Typer(
    name="aift",
    help="AI-Friendly Tools - Command-line tools optimized for AI interaction",
    add_completion=False,
)
console = Console()


@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
):
    """AI-Friendly Tools - Command-line tools optimized for AI interaction."""
    setup_logging(log_level)


@app.command()
def info():
    """Display information about AIFT configuration."""
    config = get_config()

    logger.info("Displaying AIFT configuration information")

    table = Table(title="AIFT Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Log Level", config.log_level)
    table.add_row("Config Directory", str(config.config_dir))
    table.add_row("Log Directory", str(config.config_dir / "logs"))

    console.print(table)

    panel = Panel(
        "[bold green]AIFT is configured and ready to use![/bold green]\n\n"
        "Logs are stored in: [cyan]~/aift/logs[/cyan]\n"
        "Configuration: [cyan]~/aift/config.env[/cyan]",
        title="Status",
        border_style="green"
    )
    console.print(panel)


@app.command()
def config_show():
    """Show current configuration."""
    config = get_config()

    console.print("[bold cyan]Current Configuration:[/bold cyan]")
    console.print(f"Log Level: [green]{config.log_level}[/green]")
    console.print(f"Config Directory: [green]{config.config_dir}[/green]")

    logger.info("Configuration displayed")


@app.command()
def hello(
    name: str = typer.Argument("World", help="Name to greet"),
    count: int = typer.Option(1, "--count", "-c", help="Number of times to greet"),
):
    """
    Example command - Greet someone with rich formatting.

    This is a sample command demonstrating typer + rich integration.
    """
    logger.info(f"Greeting {name} {count} time(s)")

    for i in range(count):
        console.print(f"[bold green]Hello[/bold green], [bold cyan]{name}[/bold cyan]! 👋")
        if count > 1:
            console.print(f"[dim]({i + 1}/{count})[/dim]")

    logger.debug("Completed greeting operation")


@app.command()
def version():
    """Display version information."""
    from aift import __version__

    console.print(f"[bold cyan]AIFT[/bold cyan] version [bold green]{__version__}[/bold green]")
    logger.info(f"AIFT version {__version__}")


if __name__ == "__main__":
    app()
