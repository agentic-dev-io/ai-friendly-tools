# Core

Core shared library for AI Friendly Tools (AIFT).

Provides common configuration, logging, and CLI utilities for all AIFT tools.

## Features

- **Configuration Management**: Pydantic-based settings with environment variable support
- **Logging Setup**: Automatic loguru configuration with file rotation
- **CLI Framework**: Typer-based CLI utilities with Rich formatting
- **Type Safety**: Full type hints with `py.typed` marker

## Installation

Core is automatically available in the AIFT workspace. For standalone installation:

```bash
cd core
uv pip install -e .
```

## Usage

### Configuration

```python
from core.config import get_config

config = get_config()
print(config.log_level)  # "INFO"
print(config.config_dir)  # Path to ~/aift
```

### Logging

```python
from core.logging import setup_logging

# Initialize logging (usually done automatically)
setup_logging("DEBUG")

from loguru import logger
logger.info("This will be logged to console and file")
```

### CLI

The core package provides a CLI tool `aift`:

```bash
# Show version
aift version

# Display configuration
aift info

# Show current config
aift config-show

# Example command
aift hello "World" --count 3
```

## API Reference

### `core.config`

- `Config`: Main configuration class
- `get_config()`: Get global AIFT configuration instance

### `core.logging`

- `setup_logging(log_level: str)`: Configure loguru logging

### `core.cli`

- `app`: Main Typer application instance
- `main()`: CLI entry point

## Configuration Files

Configuration is stored in `~/aift/`:
- `config.env`: Environment variables
- `{tool-name}.json`: Tool-specific configuration

## Logging

Logs are written to:
- Console: Color-coded output with timestamps
- File: `~/aift/logs/aift_YYYY-MM-DD.log` (rotated daily, kept for 30 days)