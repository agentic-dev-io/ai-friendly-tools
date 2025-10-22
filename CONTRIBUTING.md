# Adding New Tools to AIFT

This guide shows how to add new command-line tools to the AIFT framework.

## Basic Tool Structure

All tools are added as commands to the main CLI application in `src/aift/cli.py`.

### Example: Simple Tool

```python
@app.command()
def mytool(
    input_text: str = typer.Argument(..., help="Text to process"),
    uppercase: bool = typer.Option(False, "--upper", "-u", help="Convert to uppercase"),
):
    """Process text with various transformations."""
    logger.info(f"Processing text: {input_text}")
    
    result = input_text.upper() if uppercase else input_text
    console.print(f"[green]Result:[/green] {result}")
    
    logger.debug("Processing completed")
```

## Tool with Configuration

Tools can save and load their own configuration:

```python
from aift.config import get_config

@app.command()
def configured_tool(
    value: str = typer.Argument(..., help="Value to process"),
    save_config: bool = typer.Option(False, "--save", help="Save configuration"),
):
    """Example tool with configuration support."""
    config = get_config()
    tool_config = config.get_tool_config("configured_tool")
    
    # Use saved configuration
    default_prefix = tool_config.get("prefix", ">>>")
    
    console.print(f"{default_prefix} {value}")
    
    # Save new configuration if requested
    if save_config:
        new_config = {"prefix": ">>>", "last_value": value}
        config.save_tool_config("configured_tool", new_config)
        logger.info("Configuration saved")
```

## Tool with Rich Output

Use Rich for beautiful terminal output:

```python
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

@app.command()
def rich_tool(count: int = typer.Option(5, help="Number of items")):
    """Display data with rich formatting."""
    # Create a table
    table = Table(title="Data Table", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    
    for i in range(count):
        table.add_row(str(i+1), f"Item {i+1}", "✓ Active")
    
    console.print(table)
    
    # Create a panel
    panel = Panel(
        "[bold green]Processing complete![/bold green]\n"
        f"Processed {count} items successfully",
        title="Summary",
        border_style="green"
    )
    console.print(panel)
```

## Tool with Progress Bar

```python
from rich.progress import track
import time

@app.command()
def progress_tool(items: int = typer.Option(10, help="Number of items to process")):
    """Process items with a progress bar."""
    logger.info(f"Processing {items} items")
    
    for i in track(range(items), description="Processing..."):
        # Simulate work
        time.sleep(0.1)
        logger.debug(f"Processed item {i+1}")
    
    console.print("[bold green]✓[/bold green] All items processed!")
```

## Tool with Interactive Input

```python
@app.command()
def interactive_tool():
    """Interactive tool example."""
    name = typer.prompt("What's your name?")
    age = typer.prompt("What's your age?", type=int)
    
    if typer.confirm("Are you ready to proceed?"):
        console.print(f"[green]Hello {name}, age {age}![/green]")
        logger.info(f"User confirmed: {name}, {age}")
    else:
        console.print("[yellow]Cancelled[/yellow]")
        logger.info("User cancelled operation")
```

## Tool with File Operations

```python
from pathlib import Path

@app.command()
def file_tool(
    input_file: Path = typer.Argument(..., help="Input file path", exists=True),
    output_file: Path = typer.Argument(..., help="Output file path"),
):
    """Process a file and write results."""
    logger.info(f"Reading from {input_file}")
    
    try:
        content = input_file.read_text()
        processed = content.upper()  # Example processing
        
        output_file.write_text(processed)
        
        console.print(f"[green]✓[/green] Processed {input_file} → {output_file}")
        logger.info(f"Successfully wrote to {output_file}")
        
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        logger.error(f"Processing failed: {e}")
        raise typer.Exit(code=1)
```

## Tool with Subcommands

For complex tools with multiple actions:

```python
# Create a sub-application
data_app = typer.Typer(help="Data management commands")

@data_app.command("import")
def data_import(source: str):
    """Import data from source."""
    console.print(f"Importing from {source}")
    logger.info(f"Data import from {source}")

@data_app.command("export")
def data_export(destination: str):
    """Export data to destination."""
    console.print(f"Exporting to {destination}")
    logger.info(f"Data export to {destination}")

# Add to main app
app.add_typer(data_app, name="data")
```

Usage: `aift data import source.json` or `aift data export output.json`

## Best Practices

1. **Always use logging**: Log important operations for debugging
2. **Use rich output**: Make your tools visually appealing and easy to understand
3. **Validate input**: Use Typer's built-in validation features
4. **Handle errors gracefully**: Provide helpful error messages
5. **Document well**: Add clear help text for all commands and options
6. **Use configuration**: Allow users to customize tool behavior
7. **Follow AI-first principles**: Make commands self-explanatory and predictable

## Testing New Tools

Add tests for your new tools in `tests/test_cli.py`:

```python
def test_mytool():
    """Test mytool command."""
    result = runner.invoke(app, ["mytool", "hello"])
    assert result.exit_code == 0
    assert "hello" in result.stdout

def test_mytool_uppercase():
    """Test mytool with uppercase option."""
    result = runner.invoke(app, ["mytool", "hello", "--upper"])
    assert result.exit_code == 0
    assert "HELLO" in result.stdout
```

Run tests with:
```bash
pytest tests/test_cli.py -v
```
