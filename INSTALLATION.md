# Installation Guide

## Installing with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

### Prerequisites

First, install uv:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Install AIFT as a tool

```bash
# Install directly from the repository
uv tool install git+https://github.com/bjoernbethge/ai-friendly-tools.git

# Or if you've cloned the repository
cd ai-friendly-tools
uv tool install .
```

This will install `aift` as a standalone tool that you can run from anywhere.

### Verify installation

```bash
aift --help
aift version
aift hello "World"
```

## Alternative Installation Methods

### Using pip

```bash
# From the repository
pip install git+https://github.com/bjoernbethge/ai-friendly-tools.git

# From a local clone
cd ai-friendly-tools
pip install .
```

### Development Installation

For development, install in editable mode with dev dependencies:

```bash
cd ai-friendly-tools
pip install -e ".[dev]"

# Run tests
pytest

# Lint code
ruff check src/
```

## Docker Installation

Build and run using Docker:

```bash
# Build the image
docker build -t ai-os .

# Run commands
docker run --rm ai-os hello "AI World"
docker run --rm ai-os info

# Interactive mode
docker run --rm -it ai-os --help
```

## Configuration

After installation, AIFT creates a configuration directory at `~/aift/`:

- **Logs**: `~/aift/logs/` - Daily rotated log files
- **Config**: `~/aift/config.env` - Environment variables
- **Tool configs**: `~/aift/{tool_name}.json` - Tool-specific settings

You can customize the log level:

```bash
export AIFT_LOG_LEVEL=DEBUG
aift info
```

Or pass it as a command-line option:

```bash
aift --log-level DEBUG hello "World"
```
