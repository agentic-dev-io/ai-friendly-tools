# Web

Standalone web intelligence suite for AI agents.

## Features

- **DuckDuckGo Search**: Fast web search integration
- **HTML Scraping**: XPath-based scraping with DuckDB webbed extension
- **REST API**: HTTP calls with pattern learning
- **Full-Text Search**: Porter stemmer + BM25 ranking
- **API Discovery**: Automatic endpoint detection

## Usage

```python
from web import Web

web = Web()

# Search
result = await web.execute("search", query="Python async")

# Scrape
result = await web.execute("scrape", url="https://example.com")

# API call
result = await web.execute("api", url="https://api.example.com", method="GET")
```

## CLI

```bash
web search "Python async patterns"
web scrape https://example.com
```
