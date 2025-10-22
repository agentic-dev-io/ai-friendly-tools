# AI-Friendly Tools (AIFT)

A collection of command-line tools optimized for AI interaction, built with modern Python best practices.

## Features

- 🚀 **AI-First Design**: Tools designed to be easily understood and used by AI assistants
- 📝 **Rich CLI**: Beautiful command-line interfaces using `typer` and `rich`
- 📊 **Smart Logging**: Automatic logging with `loguru` to `~/aift/logs`
- ⚙️ **Configurable**: Easy configuration management for all tools
- 🐳 **Docker Ready**: Includes Dockerfile for containerized usage
- 📦 **UV Compatible**: Install with `uv tool install`

## Installation

### Using UV (Recommended)

```bash
uv tool install aift
```

### Using pip

```bash
pip install aift
```

### From source

```bash
git clone https://github.com/bjoernbethge/ai-friendly-tools.git
cd ai-friendly-tools
pip install -e .
```

### Using Docker

```bash
# Build the image
docker build -t ai-os .

# Run a command
docker run --rm ai-os hello "AI World"

# Interactive mode
docker run --rm -it ai-os --help
```

## Usage

### Basic Commands

```bash
# Display version information
aift version

# Show configuration
aift info

# Example greeting command
aift hello "World"
aift hello "AI" --count 3

# View configuration
aift config-show

# Set log level
aift --log-level DEBUG hello "World"
```

### Configuration

AIFT stores configuration in `~/aift/`:
- **Logs**: `~/aift/logs/` - Daily rotated log files
- **Config**: `~/aift/config.env` - Environment variables
- **Tool configs**: `~/aift/{tool_name}.json` - Tool-specific settings

You can set environment variables with the `AIFT_` prefix:

```bash
export AIFT_LOG_LEVEL=DEBUG
aift info
```

### Logging

All AIFT tools automatically log to:
- Console output (with colors and formatting)
- Log files in `~/aift/logs/aift_YYYY-MM-DD.log`

Logs are automatically rotated daily and kept for 30 days.

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/bjoernbethge/ai-friendly-tools.git
cd ai-friendly-tools

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Lint with ruff
ruff check src/

# Format code
ruff format src/
```

## Architecture

The package is structured as follows:

- `src/aift/` - Main package directory
  - `cli.py` - CLI application using Typer
  - `config.py` - Configuration management with Pydantic
  - `logging.py` - Logging setup with Loguru
  - `__init__.py` - Package initialization

## Adding New Tools

To add a new tool to AIFT:

1. Create a new command in `src/aift/cli.py`:

```python
@app.command()
def mytool(
    arg: str = typer.Argument(..., help="Description"),
):
    """Your tool description."""
    logger.info(f"Running mytool with {arg}")
    # Your tool logic here
    console.print(f"[green]Result: {arg}[/green]")
```

2. Tool-specific configuration can be stored:

```python
from aift.config import get_config

config = get_config()
tool_config = config.get_tool_config("mytool")
```

## Technologies Used

- **[Typer](https://typer.tiangolo.com/)**: Modern CLI framework
- **[Rich](https://rich.readthedocs.io/)**: Beautiful terminal formatting
- **[Loguru](https://loguru.readthedocs.io/)**: Simplified logging
- **[Pydantic](https://docs.pydantic.dev/)**: Data validation and settings management

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.