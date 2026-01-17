# Development Guide

## Setting Up Dev Environment

```bash
# Install workspace with dev dependencies
uv sync

# Verify dev tools installed
ruff --version    # Linting
mypy --version    # Type checking
pytest --version  # Testing
```

## Code Quality

### Linting
```bash
# Check all packages
ruff check .

# Fix issues
ruff check . --fix
```

### Type Checking
```bash
# Check types
mypy core/
mypy web/
mypy mcp-manager/
mypy memo/
```

### Testing
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_cli.py

# With coverage
pytest --cov
```

## Project Structure

Each tool has:
```
tool/
├── pyproject.toml     # Dependencies & metadata
├── README.md          # Tool documentation
└── src/tool/
    ├── __init__.py
    ├── cli.py         # Typer commands
    ├── core.py        # Main logic
    └── config.py      # Configuration
```

## Adding New Commands

### In Tool's CLI

```python
from core import app
from rich.console import Console

console = Console()

@app.command()
def mycommand(name: str = typer.Argument(..., help="Name")):
    """Do something with name."""
    console.print(f"[green]✓[/green] Processing {name}")
```

### Register in pyproject.toml

```toml
[project.scripts]
my-tool = "tool_name.cli:app"
```

## Adding New Dependencies

```bash
# Add to tool's pyproject.toml
[project]
dependencies = [
    "existing-dep",
    "new-package>=1.0",  # Add here
]

# Or use uv
uv add new-package
```

## Testing Guidelines

- Write tests for new features
- Use pytest fixtures for setup
- Mock external services
- Test edge cases

Example:
```python
def test_my_feature():
    result = my_function("test")
    assert result == expected
```

## Code Style

- Use type hints
- Follow PEP 8
- Keep functions small
- Add docstrings

Ruff will auto-format on `--fix`.

## Debugging

Enable debug mode:
```bash
AIFT_LOG_LEVEL=DEBUG aift test
```

Or use Python debugger:
```python
import pdb; pdb.set_trace()
```

## Building Docker Image

```bash
# Build locally
docker build -t aift-os:dev .

# Or with buildx
docker buildx build -t aift-os:dev . --load

# Test it
docker run -it aift-os:dev
```

## Contributing

1. Create feature branch
2. Make changes
3. Run `ruff check . --fix`
4. Run `mypy` for type checking
5. Run `pytest` for tests
6. Commit with clear message
7. Create pull request

## Common Tasks

### Update dependencies
```bash
# Update one package
uv add package@latest

# Update all
uv sync --upgrade
```

### Run specific tool
```bash
cd web
uv sync
uv run web --help
```

### Clean up
```bash
# Remove cache
rm -rf .pytest_cache __pycache__ .mypy_cache

# Or via docker
docker system prune -a
```

## Resources

- [Typer docs](https://typer.tiangolo.com/)
- [Rich docs](https://rich.readthedocs.io/)
- [Pydantic docs](https://docs.pydantic.dev/)
- [Pytest docs](https://docs.pytest.org/)
