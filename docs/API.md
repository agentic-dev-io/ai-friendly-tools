# AIFT API Documentation

Comprehensive API reference for all AIFT tools and libraries.

## Table of Contents

1. [Core Library](#core-library)
2. [Web Intelligence Suite](#web-intelligence-suite)
3. [MCP Manager](#mcp-manager)
4. [Memory & AI Tools](#memory--ai-tools)

---

## Core Library

The core library provides shared functionality for all AIFT tools.

### Configuration (`core.config`)

```python
from core.config import get_config, Config

config = get_config()
print(config.log_level)  # INFO
print(config.config_dir)  # ~/.aift/
```

**Methods:**
- `get_config()` - Get the configuration instance
- `config.config_dir` - Path to configuration directory

### Logging (`core.logging`)

```python
from core.logging import setup_logging

setup_logging("DEBUG")  # Set debug level
```

**Methods:**
- `setup_logging(level: str)` - Configure logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### CLI (`core.cli`)

The core CLI provides system commands:

```bash
# Display version
aift version

# Show configuration and status
aift info

# Display current configuration
aift config-show

# Enable debug mode
aift debug

# Validate configuration
aift validate

# Run functionality tests
aift test

# List available tools
aift list-tools

# Show system status and resources
aift status

# Greeting example
aift hello [NAME] --count [COUNT]
```

---

## Web Intelligence Suite

The web tool provides web research, scraping, and API integration capabilities.

### Configuration

```python
from web.core import WebConfig, Web

config = WebConfig(
    db_path="./data/web.db",
    max_results=10,
    max_content_size=5 * 1024 * 1024  # 5MB
)

web = Web(config)
```

### Web Core (`web.core`)

Main Web Intelligence class with async operations.

```python
from web.core import Web
import asyncio

async def search():
    web = Web()
    result = await web.execute("search", query="python tutorial")
    print(result)

asyncio.run(search())
```

**Methods:**
- `execute(command: str, **kwargs) -> Any` - Execute a web operation
  - `search` - Search using DuckDuckGo
  - `scrape` - Scrape content from URL
  - `api` - Make API calls

### Cache System (`web.cache`)

File-based caching with TTL support.

```python
from web.cache import Cache, CacheConfig

config = CacheConfig(
    enabled=True,
    ttl_seconds=3600,
    cache_dir="./data/cache"
)

cache = Cache(config)

# Store a value
cache.set("my_key", {"data": "value"}, ttl=3600)

# Retrieve a value
value = cache.get("my_key")

# Delete a value
cache.delete("my_key")

# Clear all cache
cache.clear()

# Get cache size in MB
size_mb = cache.get_size()

# Remove expired entries
removed = cache.cleanup_expired()
```

**Methods:**
- `set(key: str, value: Any, ttl: int = None)` - Store value
- `get(key: str) -> Optional[Any]` - Retrieve value
- `delete(key: str) -> bool` - Delete entry
- `clear()` - Clear all entries
- `get_size() -> int` - Get cache size in MB
- `cleanup_expired() -> int` - Remove expired entries

### Search Sources (`web.sources`)

Multiple search provider support.

```python
from web.sources import SourceManager, DuckDuckGoSource, GoogleSource

manager = SourceManager()

# Search with all sources
results = await manager.search_all("python", max_results=10)

# Search with primary source (DuckDuckGo)
results = await manager.search_primary("python", max_results=10)

# Add custom source
google = GoogleSource(api_key="...", search_engine_id="...")
manager.register(google)

# Health check all sources
health = await manager.health_check_all()

# List available sources
sources = manager.list_sources()
```

**Methods:**
- `search_all(query: str, max_results: int) -> Dict[str, List[SearchResult]]`
- `search_primary(query: str, max_results: int) -> List[SearchResult]`
- `register(source: SearchSource)`
- `health_check_all() -> Dict[str, bool]`
- `list_sources() -> List[str]`
- `get_source(name: str) -> Optional[SearchSource]`

### CLI Commands

```bash
# Search the web
web search "machine learning"

# Scrape a website
web scrape "https://example.com"

# Make API calls
web api "https://api.example.com/data" --method GET

# Workflow operations
web workflow create "name" "description"
web workflow run "name" --param key=value
web workflow list
```

---

## MCP Manager

Manages Model Context Protocol connections and DuckDB integration.

### Configuration (`mcp_manager.config`)

```python
from mcp_manager.config import GatewayConfig, SecurityConfig

security = SecurityConfig(
    allowed_commands=["/usr/bin/python3"],
    allowed_urls=["http://localhost:"]
)

config = GatewayConfig(
    default_database="main",
    databases={"main": "./data/mcp.db"},
    security=security
)
```

### Gateway (`mcp_manager.gateway`)

Central coordinator for MCP connections.

```python
from mcp_manager.gateway import MCPGateway

gateway = MCPGateway()

# Connect to an MCP server
gateway.connect_server(
    name="my_server",
    transport="stdio",
    args=["/usr/bin/python3", "server.py"]
)

# Execute query
result = gateway.execute_query(
    "SELECT * FROM my_table",
    database="main"
)

# Get connection
conn = gateway.get_default_connection()
```

**Methods:**
- `connect_server(name: str, transport: str, args: list, database: str = None)`
- `execute_query(query: str, database: str = None) -> List[tuple]`
- `get_default_connection() -> duckdb.DuckDBPyConnection`
- `get_database_connection(name: str, path: Path = None) -> duckdb.DuckDBPyConnection`

### Registry (`mcp_manager.registry`)

Track and manage MCP server connections.

```python
from mcp_manager.registry import ConnectionRegistry, ConnectionStatus

registry = ConnectionRegistry()

# Register a connection
registry.register("my_server", "stdio", ["/usr/bin/python3", "server.py"])

# Get connection info
info = registry.get("my_server")

# List all connections
connections = registry.list()

# Check if registered
exists = registry.exists("my_server")

# Remove connection
registry.remove("my_server")
```

**Methods:**
- `register(name: str, transport: str, args: list)`
- `get(name: str) -> Optional[ConnectionInfo]`
- `list() -> List[ConnectionInfo]`
- `exists(name: str) -> bool`
- `remove(name: str) -> bool`

### Health Monitoring (`mcp_manager.health`)

Monitor MCP connection health with metrics.

```python
from mcp_manager.health import HealthMonitor
import asyncio

monitor = HealthMonitor(check_interval=60)

# Register connections
monitor.register_connection("server1", "stdio")

# Check single connection
async def check():
    async def health_check():
        return True  # Connection is healthy
    
    is_healthy = await monitor.check_connection("server1", health_check)
    print(is_healthy)

asyncio.run(check())

# Start continuous monitoring
async def monitor_all():
    check_functions = {
        "server1": lambda: asyncio.sleep(0.1) or True
    }
    await monitor.start_monitoring(check_functions)
    # Do work...
    await monitor.stop_monitoring()

# Get health status
health = monitor.get_connection_health("server1")
all_health = monitor.get_all_health()

# Record custom metrics
monitor.record_metric("cpu_usage", "healthy", 45.5)

# Get status summary
summary = monitor.get_status_summary()
overall = monitor.get_overall_status()  # 'healthy', 'degraded', 'unhealthy'
```

**Methods:**
- `register_connection(name: str, transport: str)`
- `check_connection(name: str, check_func) -> bool`
- `get_connection_health(name: str) -> Optional[ConnectionHealth]`
- `get_all_health() -> Dict[str, ConnectionHealth]`
- `record_metric(name: str, status: str, value: Any, details: Dict = None)`
- `get_metric(name: str) -> Optional[HealthMetric]`
- `get_all_metrics() -> Dict[str, HealthMetric]`
- `get_overall_status() -> str`
- `get_status_summary() -> Dict[str, Any]`
- `start_monitoring(check_functions: Dict)`
- `stop_monitoring()`
- `reset()`

### Database Connection Pool (`mcp_manager.database`)

Manage DuckDB connection pooling.

```python
from mcp_manager.database import DatabaseConnectionPool
from pathlib import Path

pool = DatabaseConnectionPool(Path("./data"))

# Get a connection
conn = pool.get_connection("my_db", Path("./data/my.db"))

# Execute query
result = conn.execute("SELECT 1").fetchall()

# Close all connections
pool.close_all()
```

**Methods:**
- `get_connection(name: str, path: Path = None) -> duckdb.DuckDBPyConnection`
- `close_all()`

### Security (`mcp_manager.security`)

Security validation for commands and URLs.

```python
from mcp_manager.security import SecurityConfig, validate_command, validate_url

security = SecurityConfig(
    allowed_commands=["/usr/bin/python3"],
    allowed_urls=["https://api.example.com"]
)

# Validate command
is_allowed = validate_command("/usr/bin/python3", security)

# Validate URL
is_allowed = validate_url("https://api.example.com", security)
```

**Methods:**
- `validate_command(command: str, config: SecurityConfig) -> bool`
- `validate_url(url: str, config: SecurityConfig) -> bool`

### CLI Commands

```bash
# Initialize database
mcp-man init mydb

# Connect to MCP server
mcp-man connect myserver --transport stdio --args "python3 server.py"

# Execute queries
mcp-man query "SELECT * FROM table" --database mydb

# List resources
mcp-man resources

# Security management
mcp-man security add-command /usr/bin/python3
mcp-man security add-url https://api.example.com
mcp-man security list

# Serve as MCP server
mcp-man serve --port 3000

# Health check
mcp-man health
```

---

## Memory & AI Tools

Advanced memory and AI capabilities for AIFT.

### Memory Video (`memo.memvid`)

Compact memory format for videos and sequences.

```python
from memo.memvid import MemoryVideo

# Create memory video
mv = MemoryVideo()

# Add events
mv.add_event(timestamp=0.0, label="start", value={"action": "begin"})
mv.add_event(timestamp=1.5, label="action", value={"action": "run"})

# Compress
compressed = mv.compress()

# Extract features
features = mv.extract_features()
```

---

## Examples

### Complete Web Search Workflow

```python
from web.core import Web
from web.cache import Cache
import asyncio

async def search_with_cache():
    cache = Cache()
    web = Web()
    
    query = "python async programming"
    
    # Check cache first
    cached = cache.get(query)
    if cached:
        return cached
    
    # Search if not cached
    result = await web.execute("search", query=query)
    
    # Cache result
    cache.set(query, result, ttl=86400)  # 24 hours
    
    return result

asyncio.run(search_with_cache())
```

### MCP Connection Health Monitoring

```python
from mcp_manager.health import HealthMonitor
from mcp_manager.gateway import MCPGateway
import asyncio

async def monitor_gateway():
    monitor = HealthMonitor(check_interval=30)
    gateway = MCPGateway()
    
    # Register connections
    monitor.register_connection("main", "stdio")
    
    # Start monitoring
    async def check_main():
        try:
            conn = gateway.get_default_connection()
            result = conn.execute("SELECT 1").fetchone()
            return result is not None
        except:
            return False
    
    check_functions = {"main": check_main}
    await monitor.start_monitoring(check_functions)
    
    # Get status
    status = monitor.get_status_summary()
    print(f"Overall: {monitor.get_overall_status()}")
    print(f"Healthy: {status['connections']['healthy']}")

asyncio.run(monitor_gateway())
```

---

## Error Handling

All AIFT components provide comprehensive error handling:

```python
from web.core import Web
import asyncio

async def safe_search():
    try:
        web = Web()
        result = await web.execute("search", query="")  # Empty query
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

asyncio.run(safe_search())
```

---

## Performance Considerations

- **Caching**: Use cache TTL appropriately (3600-86400 seconds)
- **Connection Pooling**: Reuse gateway instances for better performance
- **Health Checks**: Set monitoring interval based on your needs (30-300 seconds)
- **Async Operations**: Always use async operations for web calls
- **Database**: Use connection pooling for concurrent access

---

## Logging

Enable debug logging for troubleshooting:

```python
from core.logging import setup_logging

setup_logging("DEBUG")  # Show all debug messages
```

Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

---

## Version

Current version: **0.1.0**

---

## Support

For issues and questions:
- GitHub: https://github.com/anomalyco/aift
- Documentation: See `/docs` directory
