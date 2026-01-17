# AIFT - AI-Friendly Tools

Command-line tools optimized for AI interaction. Simple, clear, context-aware.

## Quick Start

### With Docker (Recommended)
```bash
docker-compose up -d
docker-compose exec aift-os bash
aift test
```

### Local Development
```bash
# Install dependencies
py -m pip install -e core web mcp-manager memo

# Test installation
aift --help
aift test
```

## Available Tools

### AIFT Core
```bash
aift info          # System information
aift debug         # Debug mode with system details
aift validate      # Validate configuration
aift test          # Run functionality tests
aift version       # Show version
```

### Web Intelligence Suite
```bash
web search "query"
web scrape "https://url"
web api "https://api.url"
```

### MCP Manager
```bash
mcp-man status
mcp-man connections
mcp-man query "SELECT * FROM table"
```

## Project Structure

```
aift/
├── core/              # Core library (config, logging, CLI)
├── web/               # Web intelligence suite
├── mcp-manager/       # DuckDB MCP manager
├── memo/              # Memory & AI tools
├── docs/              # 📖 Documentation (see links below)
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Complete stack
└── README.md          # This file
```

## Documentation

Detailed guides in `/docs`:

- **[docs/Docker.md](docs/Docker.md)** - Docker setup & commands
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** - Local installation
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Project structure & design
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development guide & contributing

Tool-specific docs:
- [core/README.md](core/README.md) - Core library
- [web/README.md](web/README.md) - Web tool
- [mcp-manager/README.md](mcp-manager/README.md) - MCP manager
- [tests/README_TESTS.md](tests/README_TESTS.md) - Testing guide

## Docker Setup

### What's Included
- Python 3.11 + all AIFT tools
- DuckDB 1.4.3, Ruff 0.14.13, MyPy 1.19.1
- CLI tools: rg, fd, bat, eza, yq, duckdb
- ML/AI libs: torch, transformers, sklearn, scipy
- SurrealDB + MCP services

### Image Details
- **Version**: 2026-01
- **Size**: 1.04 GB
- **Base**: Python 3.11-slim-bookworm

### Ports
- `8000` - AIFT services
- `8080` - Web interface
- `8888` - SurrealDB
- `9999` - MCP Manager

### Volumes
```yaml
aift-config  → /home/aift/.config      # Config
aift-logs    → /home/aift/aift/logs    # Logs
workspace    → /workspace/user         # Your data
```

## Configuration

Environment variables (prefix: `AIFT_`):
```bash
AIFT_LOG_LEVEL=DEBUG      # Logging level
PYTHONUNBUFFERED=1        # Python output
CLAUDE_CODE_SANDBOX=true  # Sandbox mode
```

Config file: `~/.config/aift/config.env`

## Dependencies

### Updated January 2026

| Package | Version | Notes |
|---------|---------|-------|
| pydantic | 2.12.5 | Data validation |
| typer | 0.21.1 | CLI framework |
| rich | 14.2.0 | Terminal UI |
| duckdb | 1.4.3 | Embedded SQL |
| ruff | 0.14.13 | Linting |
| mypy | 1.19.1 | Type checking |
| pytest | 9.0.2 | Testing |

## Development

### Run Tests
```bash
aift test
pytest
```

### Code Quality
```bash
ruff check src/
mypy src/
```

### Build Docker Image
```bash
docker buildx build -t aift-os:latest . --load
docker-compose up -d
```

## Commands Reference

### CLI Aliases (enabled in container)
```bash
ls    → eza --icons                  # Modern listing
cat   → bat                          # Syntax highlighting
grep  → rg                           # Fast search
find  → fd                           # Fast find
```

### Docker Commands
```bash
docker-compose up -d                 # Start services
docker-compose down                  # Stop services
docker-compose logs -f               # View logs
docker-compose exec aift-os bash     # Access shell
```

## Troubleshooting

### Container issues
```bash
docker-compose down -v
docker-compose up -d --build
```

### Check health
```bash
docker-compose ps
docker-compose logs aift-os
```

### Debug mode
```bash
AIFT_LOG_LEVEL=DEBUG aift test
```

More help in [docs/Docker.md](docs/Docker.md) or [docs/INSTALLATION.md](docs/INSTALLATION.md).

## License

See LICENSE file.

## Getting Help

- 📖 **[Installation](docs/INSTALLATION.md)** - Get started locally
- 🐳 **[Docker](docs/Docker.md)** - Container setup  
- 🏗️ **[Architecture](docs/ARCHITECTURE.md)** - Project structure
- 👨‍💻 **[Development](docs/DEVELOPMENT.md)** - Contributing guide
- 🧪 **[Testing](tests/README_TESTS.md)** - Run & write tests
