# Installation

## Using Docker (Recommended)

```bash
docker-compose up -d
docker-compose exec aift-os bash
```

See [Docker.md](Docker.md) for details.

## Local Development

### Requirements
- Python 3.10+ (3.11 recommended)
- UV package manager
- Git

### Setup

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/bjoernbethge/ai-friendly-tools.git
cd ai-friendly-tools

# Install workspace
uv sync

# Verify installation
aift --help
```

### Install Individual Tools

```bash
# Install just core
uv pip install -e core

# Or specific tool
uv pip install -e web
uv pip install -e mcp-manager

# Or all
uv pip install -e core web mcp-manager memo
```

## Configuration

Create `~/.config/aift/config.env`:

```bash
AIFT_LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

Or use environment variables:

```bash
export AIFT_LOG_LEVEL=DEBUG
aift test
```

## Verify Installation

```bash
# Test core
aift test

# Test web tool
web search "test query"

# Test MCP manager
mcp-man status
```

## Troubleshooting

### Python version issues
```bash
python --version  # Should be 3.10+
uv python list    # Check available versions
```

### UV issues
```bash
uv --version
uv self update
```

### Dependency conflicts
```bash
uv sync --refresh
uv pip list | grep aift
```

## Next Steps

- See [Docker.md](Docker.md) for container setup
- See tool READMEs for specific usage
- Run `aift --help` for CLI reference
