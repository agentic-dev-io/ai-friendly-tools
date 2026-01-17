# Docker Setup

## Quick Start

```bash
# Start everything
docker-compose up -d

# Access container
docker-compose exec aift-os bash

# Test it works
aift test
```

## Common Commands

```bash
# Stop
docker-compose down

# Logs
docker-compose logs -f aift-os

# Restart
docker-compose restart

# Rebuild
docker-compose build --no-cache aift-os
```

## Tools Inside Container

```bash
# AIFT commands
aift debug
aift validate
aift test

# Web tool
web search "your query"

# MCP Manager
mcp-man status

# Data query
duckdb
```

## Troubleshooting

### Container won't start
```bash
docker-compose logs aift-os
docker-compose down -v
docker-compose up -d --build
```

### Disk space issues
```bash
docker system prune -a
docker volume prune
```

### Check what's running
```bash
docker ps
docker-compose ps
```

## Ports

- `8000` - AIFT
- `8080` - Web
- `8888` - SurrealDB
- `9999` - MCP

## Volumes

```
aift-config → ~/.config/aift
aift-logs   → ~/.aift/logs
workspace   → /workspace/user (your data)
```

## See Also

- `README.md` - Project overview
- `docker-compose.yml` - Full configuration
- `Dockerfile` - Image definition
