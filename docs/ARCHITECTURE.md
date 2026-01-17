# Architecture

## Overview

AIFT is a modular Python workspace using UV. Four independent tools with shared core library.

## Structure

```
aift/
├── core/              Shared library
│   ├── cli           CLI framework (typer)
│   ├── config        Configuration management (pydantic)
│   └── logging       Logging setup (loguru)
│
├── web/               Web intelligence suite
│   ├── core          DuckDuckGo + DuckDB + scraping
│   ├── cli           Web commands
│   ├── workflows     Workflow engine
│   └── autolearn     Learning system
│
├── mcp-manager/       DuckDB MCP manager
│   ├── gateway       Connection management
│   ├── database      DuckDB interface
│   ├── security      Security & validation
│   ├── registry      Connection registry
│   └── cli           CLI commands
│
└── memo/              Memory & AI tools
    ├── video processing
    ├── memory management
    └── AI integrations
```

## Key Technologies

- **Python 3.11** - Main language
- **UV** - Package manager & workspace
- **Typer** - CLI framework
- **Rich** - Terminal UI
- **Loguru** - Logging
- **Pydantic** - Data validation
- **DuckDB** - Embedded SQL database
- **PyTorch** - ML/AI (memo tool)

## Dependencies

### Core (shared)
- pydantic 2.12.5 - Data validation
- typer 0.21.1 - CLI framework
- rich 14.2.0 - Terminal formatting
- loguru 0.7.3 - Logging

### Web
- ddgs 9.10.0 - DuckDuckGo search
- duckdb 1.4.3 - SQL database
- httpx 0.28.1 - HTTP client
- lxml 6.0.2 - HTML parsing

### MCP-Manager
- duckdb 1.4.3 - Database
- rich 14.2.0 - UI

### Development
- ruff 0.14.13 - Linting & formatting
- mypy 1.19.1 - Type checking
- pytest 9.0.2 - Testing

## Design Principles

1. **AI-First** - Tools designed for AI interaction
2. **Modular** - Independent tools, shared core
3. **Clear** - Simple, understandable code
4. **Configurable** - Easy to customize
5. **Tested** - Quality assured

## Configuration

Each tool can be configured via:
- Environment variables (AIFT_*)
- Config files (~/.config/aift/)
- Command-line arguments

## Logging

All tools use Loguru for consistent logging to:
- Console (live output)
- Files (~/.aift/logs/)

## Extending

To add a new tool:
1. Create new package in root
2. Add pyproject.toml with dependencies
3. Import core library for CLI/config/logging
4. Add CLI entry point to tool's pyproject.toml
