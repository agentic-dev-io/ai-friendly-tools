# GitHub Copilot Instructions for AI-Friendly Tools (AIFT)

## Tech Stack

- **Python**: 3.8+ (target Python 3.8 for compatibility)
- **CLI Framework**: [Typer](https://typer.tiangolo.com/) - Modern CLI framework with type hints
- **Terminal Output**: [Rich](https://rich.readthedocs.io/) - Beautiful terminal formatting
- **Logging**: [Loguru](https://loguru.readthedocs.io/) - Simplified, powerful logging
- **Configuration**: [Pydantic](https://docs.pydantic.dev/) - Data validation and settings management
- **Testing**: pytest with typer.testing.CliRunner
- **Linting/Formatting**: Ruff

## Project Structure

```
ai-friendly-tools/
├── src/aift/           # Main package
│   ├── cli.py         # CLI commands using Typer
│   ├── config.py      # Configuration management with Pydantic
│   ├── logging.py     # Logging setup with Loguru
│   └── __init__.py    # Package initialization
├── tests/             # Test suite
├── libs/              # Additional libraries (aifts-mcp-manager, aifts-core)
└── pyproject.toml     # Project configuration
```

## Coding Conventions

### Python Style

- **Line length**: 100 characters (configured in pyproject.toml)
- **Target version**: Python 3.8
- **Type hints**: Always use type hints for function parameters and return values
- **Docstrings**: Use clear, concise docstrings for all public functions and commands
- **Imports**: Follow Ruff's import sorting (use `ruff format` to auto-format)

### CLI Commands

- Use `@app.command()` decorator for new commands
- Always include help text for commands, arguments, and options
- Use descriptive argument names with type hints
- Use Rich's console for formatted output
- Use Loguru's logger for all logging operations

Example command structure:
```python
@app.command()
def command_name(
    arg: str = typer.Argument(..., help="Description of argument"),
    option: bool = typer.Option(False, "--flag", "-f", help="Description of option"),
):
    """Clear, concise command description."""
    logger.info(f"Action description with {arg}")
    console.print(f"[green]Success message[/green]")
    logger.debug("Detailed debug information")
```

### Rich Output Style

- Use semantic colors: `[green]` for success, `[red]` for errors, `[yellow]` for warnings, `[cyan]` for info
- Use Rich's Table for tabular data
- Use Rich's Panel for important messages or summaries
- Use emojis sparingly but effectively (e.g., ✓ for success, ✗ for errors)
- Keep output clean and AI-readable

### Logging

- Use appropriate log levels:
  - `logger.debug()` - Detailed diagnostic information
  - `logger.info()` - General informational messages
  - `logger.warning()` - Warning messages for potentially harmful situations
  - `logger.error()` - Error messages for serious problems
  - `logger.critical()` - Critical messages for very severe problems
- Logs automatically save to `~/aift/logs/` with daily rotation
- Always log important operations for debugging

### Configuration

- Store configuration in `~/aift/` directory
- Use `get_config()` to access global configuration
- Use `config.get_tool_config(tool_name)` for tool-specific settings
- Environment variables use `AIFT_` prefix
- Configuration is managed through Pydantic Settings

## Testing Practices

- **Test location**: All tests in `tests/` directory
- **Naming**: Test files must start with `test_` (e.g., `test_cli.py`)
- **Test functions**: Must start with `test_` (e.g., `test_version()`)
- **CLI testing**: Use `CliRunner` from `typer.testing`
- **Assertions**: Test exit codes, stdout content, and expected behavior
- **Docstrings**: Every test should have a clear docstring explaining what it tests

Example test structure:
```python
def test_command_name():
    """Test description of what this test validates."""
    result = runner.invoke(app, ["command", "args"])
    assert result.exit_code == 0
    assert "expected output" in result.stdout
```

## Development Commands

### Install for development
```bash
pip install -e ".[dev]"
```

### Run tests
```bash
pytest
pytest -v  # verbose output
pytest tests/test_cli.py  # specific test file
```

### Lint and format
```bash
ruff check src/  # check for linting issues
ruff format src/  # auto-format code
```

### Run the CLI
```bash
aift --help
aift version
aift hello "World"
```

## AI-First Design Principles

This project is specifically designed to be AI-friendly:

1. **Clear naming**: Use descriptive, unambiguous names for functions, variables, and commands
2. **Type hints everywhere**: Make code self-documenting through comprehensive type hints
3. **Rich help text**: Every command and option should have clear, helpful descriptions
4. **Consistent structure**: Follow established patterns when adding new commands
5. **Predictable behavior**: Commands should work as their names and descriptions suggest
6. **Comprehensive logging**: Log important operations to aid debugging and understanding
7. **Error handling**: Provide clear, actionable error messages

## Adding New Tools

When adding a new command-line tool:

1. Add command to `src/aift/cli.py` using the established pattern
2. Use Typer for argument/option parsing
3. Use Rich for formatted output
4. Use Loguru for logging
5. Add configuration support if needed via `config.get_tool_config()`
6. Write tests in `tests/test_cli.py`
7. Update documentation if the tool adds significant new functionality

## Do's and Don'ts

### DO:
- ✓ Use type hints for all function parameters and return values
- ✓ Follow the established Rich output color scheme
- ✓ Log important operations with appropriate log levels
- ✓ Write tests for new commands
- ✓ Keep commands simple and focused
- ✓ Use existing patterns from the codebase
- ✓ Format code with `ruff format` before committing

### DON'T:
- ✗ Add commands without help text
- ✗ Use `print()` instead of `console.print()`
- ✗ Ignore type hints or use `Any` unnecessarily
- ✗ Add external dependencies without good reason
- ✗ Break compatibility with Python 3.8
- ✗ Skip writing tests for new functionality
- ✗ Use inconsistent naming conventions

## Dependencies

When adding new dependencies:
- Prefer well-maintained, popular libraries
- Add to `dependencies` in `pyproject.toml`
- Use version constraints (e.g., `>=1.0.0` not `==1.0.0`)
- Consider if the dependency is necessary or if it can be avoided
- Add development dependencies to `[project.optional-dependencies.dev]`

## File Operations

- Configuration directory: `~/aift/`
- Log directory: `~/aift/logs/`
- Configuration file: `~/aift/config.env`
- Tool-specific configs: `~/aift/{tool_name}.json`
