"""Main CLI application for AIFT using Typer."""

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import get_config
from core.logging import setup_logging

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
    from core import __version__

    console.print(f"[bold cyan]Core[/bold cyan] version [bold green]{__version__}[/bold green]")
    logger.info(f"Core version {__version__}")


@app.command()
def debug():
    """Enable debug mode and show system information."""
    setup_logging("DEBUG")
    console.print("[bold cyan]Debug Mode Enabled[/bold cyan]")
    console.print()

    import platform

    info_table = Table(title="System Information", show_header=True, header_style="bold magenta")
    info_table.add_column("Property", style="cyan", no_wrap=True)
    info_table.add_column("Value", style="green")

    info_table.add_row("Python Version", platform.python_version())
    info_table.add_row("Platform", platform.platform())
    info_table.add_row("Architecture", platform.machine())

    config = get_config()
    info_table.add_row("Config Directory", str(config.config_dir))
    info_table.add_row("Log Level", config.log_level)

    console.print(info_table)
    logger.debug("Debug information displayed")


@app.command()
def validate():
    """Validate configuration and environment."""
    console.print("[bold cyan]Validating Configuration...[/bold cyan]")
    logger.info("Starting configuration validation")

    config = get_config()
    errors = []
    warnings = []

    # Check config directory
    if not config.config_dir.exists():
        errors.append(f"Config directory does not exist: {config.config_dir}")
    else:
        console.print("[green]✓[/green] Config directory exists")

    # Check log directory
    log_dir = config.config_dir / "logs"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]⚠[/yellow] Created log directory: {log_dir}")
    else:
        console.print("[green]✓[/green] Log directory exists")

    # Display results
    if errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for error in errors:
            console.print(f"  [red]✗ {error}[/red]")

    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in warnings:
            console.print(f"  [yellow]⚠ {warning}[/yellow]")

    if not errors:
        console.print("\n[bold green]✓ All validations passed![/bold green]")
        logger.info("Configuration validation successful")
    else:
        logger.error(f"Configuration validation failed with {len(errors)} error(s)")
        raise typer.Exit(1)


@app.command()
def test():
    """Run basic functionality tests."""
    console.print("[bold cyan]Running AIFT Tests...[/bold cyan]\n")
    logger.info("Starting functionality tests")

    test_results = Table(title="Test Results", show_header=True, header_style="bold magenta")
    test_results.add_column("Test", style="cyan")
    test_results.add_column("Status", style="green")
    test_results.add_column("Details", style="yellow")

    # Test 1: Configuration Loading
    try:
        config = get_config()
        test_results.add_row(
            "Configuration Loading",
            "[green]PASS[/green]",
            "Config loaded successfully",
        )
        logger.info("✓ Configuration loading test passed")
    except Exception as e:
        test_results.add_row("Configuration Loading", "[red]FAIL[/red]", str(e))
        logger.error(f"✗ Configuration loading test failed: {e}")

    # Test 2: Logging System
    try:
        logger.info("Test log message")
        test_results.add_row("Logging System", "[green]PASS[/green]", "Logging initialized")
        logger.info("✓ Logging system test passed")
    except Exception as e:
        test_results.add_row("Logging System", "[red]FAIL[/red]", str(e))
        logger.error(f"✗ Logging system test failed: {e}")

    # Test 3: Config Directory Access
    try:
        config = get_config()
        if config.config_dir.exists():
            config_path = str(config.config_dir)
            test_results.add_row(
                "Config Directory Access",
                "[green]PASS[/green]",
                f"Accessible: {config_path}",
            )
            logger.info("✓ Config directory access test passed")
        else:
            raise Exception("Config directory not accessible")
    except Exception as e:
        test_results.add_row("Config Directory Access", "[red]FAIL[/red]", str(e))
        logger.error(f"✗ Config directory access test failed: {e}")

    console.print(test_results)
    console.print("\n[bold green]Testing complete![/bold green]")


@app.command()
def list_tools():
    """List all available AIFT tools and subcommands."""
    console.print("[bold cyan]Available AIFT Tools:[/bold cyan]\n")
    logger.info("Listing available tools")

    tools = Table(title="Tools", show_header=True, header_style="bold magenta")
    tools.add_column("Tool", style="cyan", no_wrap=True)
    tools.add_column("Description", style="green")
    tools.add_column("Module", style="yellow")

    tool_list = [
        ("core", "Core AIFT functionality - configuration, logging, and CLI base", "aift.cli"),
        ("web", "Web intelligence suite - search, scraping, and API calls", "web.cli"),
        ("mcp-manager", "MCP Manager - DuckDB gateway and security", "mcp_manager.cli"),
        ("memo", "Memory and AI tools - memvid, transformers, torch", "memo"),
    ]

    for name, description, module in tool_list:
        tools.add_row(name, description, module)

    console.print(tools)

    console.print("\n[bold cyan]Core Commands:[/bold cyan]")
    commands = Table(show_header=True, header_style="bold magenta")
    commands.add_column("Command", style="cyan", no_wrap=True)
    commands.add_column("Description", style="green")

    command_list = [
        ("aift info", "Display AIFT configuration and status"),
        ("aift version", "Show version information"),
        ("aift config-show", "Display current configuration"),
        ("aift debug", "Enable debug mode and show system info"),
        ("aift validate", "Validate configuration and environment"),
        ("aift test", "Run functionality tests"),
        ("aift list-tools", "List available tools (this command)"),
        ("aift status", "Show system status and resource usage"),
        ("aift hello [NAME]", "Example greeting command"),
    ]

    for cmd, desc in command_list:
        commands.add_row(cmd, desc)

    console.print(commands)


@app.command()
def status():
    """Display system status and resource information."""
    console.print("[bold cyan]AIFT System Status[/bold cyan]\n")
    logger.info("Displaying system status")

    import psutil
    import platform
    from datetime import datetime

    # Create status table
    status_table = Table(show_header=True, header_style="bold magenta")
    status_table.add_column("Property", style="cyan", no_wrap=True)
    status_table.add_column("Value", style="green")

    # System info
    status_table.add_row("Status", "[green]✓ Running[/green]")
    status_table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    status_table.add_row("Platform", platform.platform())
    status_table.add_row("Python Version", platform.python_version())
    status_table.add_row("Architecture", platform.machine())

    # Process info
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        status_table.add_row("Memory Usage", f"{mem_info.rss / 1024 / 1024:.2f} MB")
        status_table.add_row("Process ID", str(process.pid))
    except Exception as e:
        logger.warning(f"Could not get process info: {e}")

    # CPU info
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        status_table.add_row("CPU Usage", f"{cpu_percent}%")
        status_table.add_row("CPU Count", str(psutil.cpu_count()))
    except Exception as e:
        logger.warning(f"Could not get CPU info: {e}")

    # Disk info
    try:
        disk = psutil.disk_usage("/")
        used_percent = (disk.used / disk.total) * 100
        status_table.add_row("Disk Usage", f"{used_percent:.1f}% ({disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB)")
    except Exception as e:
        logger.warning(f"Could not get disk info: {e}")

    # Config status
    try:
        config = get_config()
        status_table.add_row("Config Directory", str(config.config_dir))
        status_table.add_row("Config Valid", "[green]✓ Yes[/green]")
    except Exception as e:
        status_table.add_row("Config Valid", f"[red]✗ {e}[/red]")
        logger.error(f"Config validation failed: {e}")

    console.print(status_table)
    console.print("\n[bold green]System is operational![/bold green]")
    logger.info("System status displayed")


def cli_main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    cli_main()
