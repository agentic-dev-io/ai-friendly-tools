# mcp-man - DuckDB MCP Gateway

A comprehensive MCP Gateway that manages multiple Model Context Protocol server connections and exposes DuckDB databases as MCP resources.

## Overview

`mcp-man` transforms DuckDB into a powerful MCP gateway with dual capabilities:
- **Client Mode**: Connect to and query multiple remote MCP servers
- **Server Mode**: Expose DuckDB tables and queries as MCP resources

## Installation

```bash
uv sync
```

## Quick Start

### 1. Initialize a Database

```bash
mcp-man init mydb
```

### 2. Connect to Remote MCP Servers

```bash
# Add filesystem server
mcp-man connect filesystem --transport stdio --args python3 --args /path/to/fs-server.py

# Add API server
mcp-man connect api --transport tcp --args localhost:8080

# List connections
mcp-man connections
```

### 3. Query Across MCP Servers

```bash
# Query remote resources
mcp-man query "SELECT * FROM read_csv('mcp://filesystem/data.csv')"

# List available resources
mcp-man resources

# Filter by server
mcp-man resources --server filesystem
```

### 4. Expose Database as MCP Server

```bash
# Start MCP server
mcp-man serve mydb --port 9999

# Publish a table
mcp-man publish users --uri "data://tables/users" --format json
```

## Gateway Features

### Connection Management

```bash
# Show gateway status
mcp-man gateway status

# Connect to MCP server
mcp-man connect <name> --transport <stdio|tcp|websocket> --args <args...>

# Disconnect
mcp-man disconnect <name>

# List all connections
mcp-man connections
```

### Security Management

```bash
# Add allowed command
mcp-man security add-command /usr/bin/python3

# Add allowed URL prefix
mcp-man security add-url https://api.example.com

# List security settings
mcp-man security list
```

### Query Operations

```bash
# Execute SQL across all connections
mcp-man query "SELECT * FROM table" --database mydb

# JSON output format
mcp-man query "SELECT * FROM table" --format json
```

## Architecture

**mcp-man** operates as a centralized gateway with:

- **Connection Registry**: Track multiple MCP server connections
- **Security Layer**: Allowlist-based command and URL validation
- **Database Pool**: Manage multiple DuckDB instances
- **Unified API**: Single interface for all MCP operations

## Security

Production-ready security with:
- Command allowlist for stdio transport
- URL prefix allowlist for remote resources
- Server configuration locking
- Comprehensive logging

```bash
# Configure security before connecting
mcp-man security add-command /usr/bin/python3
mcp-man security add-url https://trusted-api.com
```

## Transport Support

- **stdio**: Local process communication (default)
- **tcp**: Network-based connections
- **websocket**: WebSocket connections

## Use Cases

### 1. Database Federation
Query data across multiple MCP-enabled data sources from a single SQL interface.

### 2. AI Agent Integration
Expose database insights as MCP resources for AI agents to access.

### 3. Data Pipeline Hub
Central gateway for aggregating and transforming data from multiple sources.

### 4. Secure Data Access
Controlled access to external data sources with allowlist security.

## Features

- 🌐 **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- 🔒 **Production Security**: Allowlist-based validation for commands and URLs
- 🦆 **DuckDB Native**: Full DuckDB MCP extension integration
- 📊 **Unified Queries**: SQL across all connected resources
- 🔌 **Flexible Transports**: stdio, TCP, and WebSocket support
- 💾 **Resource Publishing**: Expose tables and queries as MCP resources
- 📈 **Connection Management**: Track, monitor, and control all connections
- 🎯 **Rich CLI**: Beautiful terminal UI with tables and formatting

## Configuration

Configuration is stored in `~/.aift/mcp-manager.json`:

```json
{
  "databases": {
    "default": "/path/to/default.duckdb"
  },
  "connections": [
    {
      "name": "filesystem",
      "transport": "stdio",
      "args": ["python3", "/path/to/server.py"],
      "enabled": true
    }
  ],
  "security": {
    "allowed_commands": ["/usr/bin/python3"],
    "allowed_urls": ["https://api.example.com"],
    "lock_servers": true,
    "log_level": "info"
  }
}
```

## References

- [DuckDB MCP Extension](https://duckdb.org/community_extensions/extensions/duckdb_mcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [DuckDB Documentation](https://duckdb.org/docs/)