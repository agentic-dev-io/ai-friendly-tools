# mcp-man - DuckDB MCP Manager

The first AI Friendly Tool (AIFT): A command-line interface for managing DuckDB databases with Model Context Protocol (MCP) integration.

## Installation

```bash
uv sync
```

## Usage

### Initialize a database

```bash
mcp-man init mydb
```

### Start MCP server

```bash
mcp-man serve mydb --port 9999
```

### Execute queries

```bash
mcp-man query "SELECT * FROM my_table" --database mydb
```

### Check server status

```bash
mcp-man status mydb
```

## Features

- 🦆 DuckDB integration with MCP protocol
- 🤖 AI-ready database queries
- 🔌 Model Context Protocol support
- 💻 Simple CLI interface

## References

- [DuckDB MCP Extension](https://duckdb.org/community_extensions/extensions/duckdb_mcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)