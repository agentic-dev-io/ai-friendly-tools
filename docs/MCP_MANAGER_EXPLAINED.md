# MCP Manager - How It Works

`mcp-man` is a DuckDB-based gateway that connects to multiple MCP (Model Context Protocol) servers and exposes them through SQL queries.

## Core Concept

```
Multiple MCP Servers
    ↓
  MCP Gateway (mcp-man)
    ↓
DuckDB Database
    ↓
SQL Query Interface
```

Think of it as: **Many data sources → One SQL interface**

## Architecture

### 4 Main Components

#### 1. **Gateway** (`gateway.py`)
Central coordinator that:
- Manages MCP server connections
- Pools database connections
- Routes queries to appropriate servers
- Handles connection lifecycle

```python
gateway = MCPGateway(config)
gateway.connect_server("filesystem", "stdio", ["python3", "server.py"])
result = gateway.execute_query("SELECT * FROM resource")
```

#### 2. **Registry** (`registry.py`)
Tracks all active connections:
- Connection status (connected/disconnected/error)
- Server metadata
- Auto-reconnect logic
- Singleton pattern (one registry per app)

```python
registry = ConnectionRegistry()
registry.register("my_server", "stdio", ["python3", "server.py"])
connections = registry.list_all()
```

#### 3. **Database Pool** (`database.py`)
Manages DuckDB connections:
- Caches connections
- Loads MCP extension
- Configures security
- Executes queries

```python
pool = DatabaseConnectionPool()
conn = pool.get_connection("db1", Path("db1.duckdb"))
result = execute_query(conn, "SELECT * FROM table")
```

#### 4. **Security** (`security.py`)
Protects against unauthorized access:
- Command allowlist (for stdio transport)
- URL allowlist (for remote resources)
- Validation before execution

```python
security = SecurityConfig(
    allowed_commands=["/usr/bin/python3"],
    allowed_urls=["https://api.example.com"]
)
```

## How It Works - Step by Step

### Example: Connect to a File Server

```bash
mcp-man connect filesystem --transport stdio --args python3 --args /path/to/server.py
```

**Behind the scenes:**

1. **CLI parses command** → `cli.py`
2. **Create gateway** → `MCPGateway()`
3. **Validate security** → `security.py` checks if allowed
4. **Register connection** → `registry.py` tracks it
5. **Get DuckDB connection** → `database.py` loads MCP extension
6. **Store config** → Save for future sessions
7. **Return success** → CLI shows confirmation

### Example: Query Remote Data

```bash
mcp-man query "SELECT * FROM read_csv('mcp://filesystem/data.csv')"
```

**Behind the scenes:**

1. **Parse SQL query** → `cli.py`
2. **Get database connection** → `database.py` with MCP extension
3. **Execute query** → DuckDB processes it
4. **MCP extension handles `mcp://` URIs** → Routes to filesystem server
5. **Receive results** → Format and display

## Data Flow

### Connection Establishment

```
User Command (connect)
    ↓
CLI Handler (cli.py)
    ↓
Validate Security (security.py)
    ↓
Create Gateway Instance (gateway.py)
    ↓
Register Connection (registry.py)
    ↓
Get DB Connection (database.py)
    ↓
Load MCP Extension
    ↓
Store Configuration
    ↓
Success Response
```

### Query Execution

```
User Command (query)
    ↓
Parse SQL
    ↓
Get Database Connection
    ↓
DuckDB executes query
    ↓
MCP Extension resolves mcp:// URIs
    ↓
Routes to connected MCP servers
    ↓
Aggregate results
    ↓
Format & display
```

## Key Features

### 1. Multi-Server Support
Connect to many servers at once:
```bash
mcp-man connect filesystem --transport stdio --args python3 fs.py
mcp-man connect api --transport tcp --args localhost:8080
mcp-man connect web --transport websocket --args wss://server.com
```

Then query all of them:
```bash
mcp-man query "SELECT * FROM filesystem_resource UNION SELECT * FROM api_resource"
```

### 2. Transport Types

| Transport | Use Case | Command |
|-----------|----------|---------|
| **stdio** | Local process | `--transport stdio --args python3 server.py` |
| **tcp** | Network | `--transport tcp --args localhost:8080` |
| **websocket** | Real-time | `--transport websocket --args wss://server.com` |

### 3. Security Model

Before connecting:
```bash
# Allow Python to run
mcp-man security add-command /usr/bin/python3

# Allow API calls to specific domain
mcp-man security add-url https://api.trusted.com

# List what's allowed
mcp-man security list
```

### 4. Resource Publishing

Expose your data as MCP resources:
```bash
# Start server
mcp-man serve mydb --port 9999

# Publish table
mcp-man publish users --uri "data://tables/users" --format json

# Publish query result
mcp-man publish reports --query "SELECT * FROM summary" --uri "data://queries/reports"
```

## Configuration

Stored in `~/.config/aift/mcp-manager.json`:

```json
{
  "default_database": "default",
  "databases": {
    "default": "/home/user/.aift/databases/default.duckdb",
    "analytics": "/home/user/.aift/databases/analytics.duckdb"
  },
  "connections": [
    {
      "name": "filesystem",
      "transport": "stdio",
      "args": ["python3", "/path/to/fs-server.py"],
      "enabled": true,
      "auto_reconnect": true
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

## Common Workflows

### Workflow 1: Setup & Connect

```bash
# 1. Initialize database
mcp-man init mydb

# 2. Configure security
mcp-man security add-command /usr/bin/python3

# 3. Connect to server
mcp-man connect filesystem --transport stdio --args python3 /path/to/server.py

# 4. List connections
mcp-man connections

# 5. Verify connection
mcp-man resources --server filesystem
```

### Workflow 2: Query Data

```bash
# 1. List available resources
mcp-man resources

# 2. Execute query
mcp-man query "SELECT * FROM my_resource WHERE id = 123"

# 3. Filter by server
mcp-man resources --server filesystem

# 4. Complex query
mcp-man query "SELECT f.*, a.data FROM filesystem_data f JOIN api_data a ON f.id = a.id"
```

### Workflow 3: Publish Data

```bash
# 1. Start server
mcp-man serve mydb --port 9999 &

# 2. Publish table
mcp-man publish users --uri "data://tables/users"

# 3. Publish query
mcp-man publish active_users --query "SELECT * FROM users WHERE active = true"

# Now other clients can connect and access these resources
```

## CLI Commands

| Command | Purpose |
|---------|---------|
| `mcp-man init [name]` | Initialize database |
| `mcp-man connect` | Add MCP server connection |
| `mcp-man disconnect` | Remove connection |
| `mcp-man connections` | List all connections |
| `mcp-man resources` | List available resources |
| `mcp-man query` | Execute SQL query |
| `mcp-man serve` | Start as MCP server |
| `mcp-man publish` | Expose table/query as resource |
| `mcp-man security` | Manage security settings |
| `mcp-man gateway` | Gateway operations |

## Use Cases

### 1. **Data Federation**
Query data from multiple sources with SQL:
```bash
mcp-man connect db1 --transport tcp --args localhost:5432
mcp-man connect db2 --transport tcp --args localhost:5433
mcp-man query "SELECT * FROM db1 UNION SELECT * FROM db2"
```

### 2. **AI Agent Integration**
Expose database as MCP resource for AI:
```bash
mcp-man serve mydb --port 9999
# AI agents connect to port 9999 and query your database
```

### 3. **Data Pipeline Hub**
Central gateway for ETL:
```bash
mcp-man connect source1 --transport stdio --args python3 source1.py
mcp-man connect source2 --transport tcp --args api.server.com:8080
mcp-man query "INSERT INTO warehouse SELECT * FROM source1 JOIN source2"
```

### 4. **Secure Data Access**
Controlled access with allowlist:
```bash
# Only allow specific commands and URLs
mcp-man security add-command /usr/bin/python3
mcp-man security add-url https://api.company.com
# Now only trusted sources can connect
```

## Error Handling

Common issues:

**Connection fails:**
```bash
mcp-man connections  # Check status
# If ERROR, check logs
AIFT_LOG_LEVEL=DEBUG mcp-man connections
```

**Query fails:**
```bash
# Check if resource exists
mcp-man resources

# Check syntax
mcp-man query "SELECT 1"  # Simple test
```

**Security blocked:**
```bash
# Check allowed commands
mcp-man security list

# Add what you need
mcp-man security add-command /usr/bin/node
```

## Key Design Patterns

1. **Singleton Registry** - Only one connection registry per app
2. **Connection Pool** - Reuse database connections
3. **Config Persistence** - Remember connections across sessions
4. **Security-First** - Allowlist validation before any operation
5. **Unified SQL Interface** - All data accessible via SQL

## Performance

- **Connection pooling** - Reuses connections, avoids overhead
- **Local caching** - DuckDB in-memory where possible
- **Lazy loading** - Extensions loaded only when needed
- **Smart routing** - Queries optimized to relevant servers

---

**See Also:**
- [MCP Manager README](../mcp-manager/README.md) - Command reference
- [Architecture](ARCHITECTURE.md) - Project overview
- [DuckDB Docs](https://duckdb.org/) - SQL engine
- [MCP Protocol](https://modelcontextprotocol.io/) - Protocol spec
