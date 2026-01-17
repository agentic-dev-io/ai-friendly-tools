#!/usr/bin/env python3
"""Web Intelligence Suite - Standalone DuckDB-powered web research tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import duckdb
from loguru import logger
from pydantic import BaseModel

# ============================================================================
# Configuration
# ============================================================================


class WebConfig(BaseModel):
    """Configuration for web tool operations."""

    db_path: str = "./data/web.db"
    max_results: int = 10
    max_content_size: int = 5 * 1024 * 1024  # 5MB


# ============================================================================
# Web Intelligence Tool
# ============================================================================


class Web:
    """Web Intelligence Suite - Standalone web research tool.

    Features:
    - DuckDuckGo search integration
    - HTML scraping with DuckDB webbed (XPath support)
    - REST API calls with pattern learning
    - API endpoint discovery
    - Full-text search on scraped content
    """

    def __init__(self, config: Optional[WebConfig] = None):
        self.config = config or WebConfig()
        self.db = None
        self._init_database()

    def _init_database(self):
        """Initialize DuckDB with community extensions and tables."""
        # Ensure directory exists
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.db = duckdb.connect(str(db_path))
            logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

        # Install and load community extensions
        try:
            self.db.execute("INSTALL httpfs FROM community")
        except Exception as e:
            logger.debug(f"httpfs extension already installed: {e}")
        try:
            self.db.execute("LOAD httpfs")
        except Exception as e:
            logger.warning(f"Failed to load httpfs extension: {e}")

        try:
            self.db.execute("INSTALL webbed FROM community")
        except Exception:
            pass  # Already installed
        self.db.execute("LOAD webbed")

        try:
            self.db.execute("INSTALL json FROM community")
        except Exception:
            pass  # Already installed
        self.db.execute("LOAD json")

        try:
            self.db.execute("INSTALL fts")
        except Exception:
            pass  # Already installed
        self.db.execute("LOAD fts")

        try:
            self.db.execute("INSTALL http_client FROM community")
        except Exception:
            pass  # Already installed
        self.db.execute("LOAD http_client")

        # Install radio extension (optional - for WebSocket/streaming support)
        # Note: radio may not be available for all platforms (e.g., Windows)
        try:
            self.db.execute("INSTALL radio FROM community")
            self.db.execute("LOAD radio")
            logger.info("✅ Radio extension loaded (WebSocket/streaming support)")
        except Exception:
            # Radio extension not available for this platform - optional, skip silently
            pass

        try:
            self.db.execute("INSTALL minijinja FROM community")
        except Exception:
            pass  # Already installed
        try:
            self.db.execute("LOAD minijinja")
        except Exception:
            logger.warning("Failed to load minijinja extension")

        try:
            self.db.execute("INSTALL jsonata FROM community")
        except Exception:
            pass  # Already installed
        try:
            self.db.execute("LOAD jsonata")
        except Exception:
            logger.warning("Failed to load jsonata extension")

        # Install specialized extensions for knowledge management
        # Only install extensions that are actually available
        available_extensions = []
        for ext in ["duckpgq", "lindel", "bitfilters"]:
            try:
                self.db.execute(f"INSTALL {ext} FROM community")
                self.db.execute(f"LOAD {ext}")
                available_extensions.append(ext)
                logger.info(f"✅ Loaded {ext} extension")
            except Exception as e:
                logger.debug(f"Extension {ext} not available: {e}")
        
        # Note: lsh extension is not available in community extensions
        # Using hash-based similarity instead

        # Create sequences first
        self.db.execute("CREATE SEQUENCE IF NOT EXISTS web_searches_seq START 1")
        self.db.execute("CREATE SEQUENCE IF NOT EXISTS web_content_seq START 1")
        self.db.execute("CREATE SEQUENCE IF NOT EXISTS api_patterns_seq START 1")

        # Create tables
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS web_searches (
                id INTEGER PRIMARY KEY DEFAULT (nextval('web_searches_seq')),
                query TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                results_count INTEGER
            )
        """
        )

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS web_content (
                id INTEGER PRIMARY KEY DEFAULT (nextval('web_content_seq')),
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,           -- Clean text content
                html TEXT,              -- Raw HTML
                links JSON,             -- Structured links
                images JSON,            -- NEW: Structured images
                tables JSON,            -- NEW: Extracted tables
                meta_description TEXT,  -- NEW: Meta description
                meta_keywords TEXT,     -- NEW: Meta keywords
                word_count INTEGER,     -- NEW: Content word count
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                search_id INTEGER REFERENCES web_searches(id)
            )
        """
        )

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS api_patterns (
                id INTEGER PRIMARY KEY DEFAULT (nextval('api_patterns_seq')),
                base_url TEXT,
                endpoint TEXT,
                method TEXT,
                auth_type TEXT,
                required_headers JSON,
                optional_headers JSON,
                params_schema JSON,
                body_schema JSON,
                success_count INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_format TEXT,
                notes TEXT
            )
        """
        )

        # Create FTS index on content
        try:
            self.db.execute(
                """
                PRAGMA create_fts_index(
                    'web_content',
                    'id',
                    'content',
                    stemmer='porter',
                    stopwords='english',
                    lower=1,
                    overwrite=1
                )
            """
            )
        except Exception as e:
            logger.warning(f"FTS index creation failed: {e}")

    async def execute(
        self,
        operation: str,
        query: Optional[str] = None,
        url: Optional[str] = None,
        method: str = "GET",
        headers: Optional[dict] = None,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
        auth: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Execute web operations."""
        logger.info(f"Executing web operation: {operation}")

        try:
            if operation == "search":
                if not query:
                    return "❌ Error: query required for search operation"
                return await self._search(query)

            elif operation == "scrape":
                if not url:
                    return "❌ Error: url required for scrape operation"
                search_id = kwargs.get("search_id")
                return await self._scrape(url, search_id)

            elif operation == "api":
                if not url:
                    return "❌ Error: url required for api operation"
                return await self._api(url, method, headers, json_data, params, auth)

            elif operation == "discover":
                if not url:
                    return "❌ Error: url required for discover operation"
                return await self._discover_api(url)

            elif operation == "workflow":
                workflow_id = kwargs.get("workflow_id")
                if not workflow_id:
                    return "❌ Error: workflow_id required for workflow operation"
                return await self._execute_workflow(workflow_id, kwargs.get("variables"))

            else:
                return f"❌ Error: Unknown operation '{operation}'. Use: search, scrape, api, discover, workflow"

        except Exception as e:
            logger.error(f"Web operation failed: {e}")
            return f"❌ Operation failed: {e}"

    async def _search(self, query: str) -> str:
        """Search DuckDuckGo for URLs."""
        from ddgs import DDGS

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.config.max_results))

            # Store search in DB
            self.db.execute(
                "INSERT INTO web_searches (query, results_count) VALUES (?, ?)",
                (query, len(results)),
            )

            # Format results
            formatted = f"✅ Found {len(results)} results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                formatted += f"{i}. {result['title']}\n"
                formatted += f"   URL: {result['href']}\n"
                formatted += f"   {result['body'][:100]}...\n\n"

            formatted += "\n💡 Use scrape operation with these URLs to fetch content."
            return formatted
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"❌ Search failed: {e}"

    async def _scrape(self, url: str, search_id: Optional[int] = None) -> str:
        """Fetch URL with DuckDB httpfs and parse with webbed using advanced XPath."""
        try:
            # Use DuckDB httpfs to fetch URL and webbed to parse with XPath
            result = self.db.execute(
                """
                WITH fetched AS (
                    SELECT * FROM read_text(?)
                ),
                parsed AS (
                    SELECT 
                        -- Title extraction with fallbacks
                        COALESCE(
                            html_extract_text(content::HTML, '//title'),
                            html_extract_text(content::HTML, '//h1'),
                            html_extract_text(content::HTML, '//meta[@property="og:title"]/@content'),
                            ''
                        ) as title,
                        
                        -- Clean content extraction (main text only, no nav/footer/scripts)
                        COALESCE(
                            html_extract_text(content::HTML, '//main'),
                            html_extract_text(content::HTML, '//article'),
                            html_extract_text(content::HTML, '//body')
                        ) as content,
                        
                        -- Structured links
                        html_extract_links(content::HTML) as links,
                        
                        -- Structured images with metadata
                        html_extract_images(content::HTML) as images,
                        
                        -- Extract tables as structured data (using table function)
                        '[]'::JSON as tables,
                        
                        -- Meta description
                        COALESCE(
                            html_extract_text(content::HTML, '//meta[@name="description"]/@content'),
                            html_extract_text(content::HTML, '//meta[@property="og:description"]/@content'),
                            ''
                        ) as meta_description,
                        
                        -- Meta keywords
                        COALESCE(
                            html_extract_text(content::HTML, '//meta[@name="keywords"]/@content'),
                            ''
                        ) as meta_keywords,
                        
                        -- Raw HTML
                        content as html
                    FROM fetched
                )
                SELECT * FROM parsed
            """,
                (url,),
            ).fetchone()

            if not result:
                return "❌ Scrape failed: No content fetched"

            (
                title,
                content,
                links,
                images,
                tables,
                meta_description,
                meta_keywords,
                html,
            ) = result

            # Calculate word count for AI context estimation
            word_count = len(content.split()) if content else 0

            # Store in DB using UPSERT (DuckDB supports UPSERT)
            self.db.execute(
                """
                INSERT INTO web_content (url, title, content, html, links, images, tables, 
                                         meta_description, meta_keywords, word_count, search_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (url) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    html = excluded.html,
                    links = excluded.links,
                    images = excluded.images,
                    tables = excluded.tables,
                    meta_description = excluded.meta_description,
                    meta_keywords = excluded.meta_keywords,
                    word_count = excluded.word_count,
                    search_id = excluded.search_id,
                    fetched_at = now()
            """,
                (
                    url,
                    title,
                    content,
                    html,
                    json.dumps(links),
                    json.dumps(images),
                    json.dumps(tables),
                    meta_description,
                    meta_keywords,
                    word_count,
                    search_id,
                ),
            )

            # Trigger LSH hash update if knowledge module is used
            # This will be handled by KnowledgeExtractor when content is searched

            # Update FTS index only if it doesn't exist
            try:
                # Check if FTS index exists
                result = self.db.execute("PRAGMA show_tables").fetchall()
                fts_exists = any(
                    "fts_main_web_content" in str(table) for table in result
                )

                if not fts_exists:
                    self.db.execute(
                        """
                        PRAGMA create_fts_index(
                            'web_content',
                            'id',
                            'content',
                            stemmer='porter',
                            stopwords='english',
                            lower=1,
                            overwrite=1
                        )
                    """
                    )
                    logger.info("FTS index created successfully")
            except Exception as e:
                logger.warning(f"FTS index update failed: {e}")

            content_preview = content[:300] + "..." if len(content) > 300 else content
            stats = f"Words: {word_count}, Links: {len(links) if links else 0}, Images: {len(images) if images else 0}, Tables: {len(tables) if tables else 0}"

            logger.success(f"Scraped {url}")
            return f"""✅ Scraped {url}:

Title: {title}
Meta: {meta_description[:100] + '...' if len(meta_description) > 100 else meta_description}

Content preview:
{content_preview}

Stats: {stats}

Stored in database with FTS index."""

        except Exception as e:
            logger.error(f"Scrape failed: {e}")
            return f"❌ Scrape failed: {e}"

    async def _api(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
        auth: Optional[dict] = None,
    ) -> str:
        """Make API requests with DuckDB http_client extension and learning capabilities."""
        try:
            # Build headers from auth if provided
            if auth:
                headers = headers or {}
                auth_type = auth.get("type", "bearer")

                if auth_type == "bearer":
                    headers["Authorization"] = f"Bearer {auth['token']}"
                elif auth_type == "api_key":
                    key_name = auth.get("header", "X-API-Key")
                    headers[key_name] = auth["key"]
                elif auth_type == "basic":
                    import base64

                    creds = base64.b64encode(
                        f"{auth['user']}:{auth['pass']}".encode()
                    ).decode()
                    headers["Authorization"] = f"Basic {creds}"

            # Check for learned pattern
            suggestion = await self._suggest_api_pattern(url, method)
            if suggestion and suggestion["confidence"] > 0.5:
                logger.info(
                    f"Found learned pattern for {url} (confidence: {suggestion['confidence']:.2f})"
                )

            # Build context for minijinja template
            headers_dict = {}
            if headers:
                for key, value in headers.items():
                    headers_dict[key.lower()] = value

            # Ensure content-type for POST with JSON
            if method == "POST" and json_data:
                headers_dict.setdefault("content-type", "application/json")

            # Make request using DuckDB http_client extension with minijinja for safe SQL generation
            if method == "GET":
                result = self.db.execute(
                    "SELECT http_get(?) AS res", (url,)
                ).fetchone()[0]
            elif method == "POST":
                if json_data:
                    # Use http_post for JSON body - render SQL template with minijinja
                    template = """
                    SELECT http_post(
                        ?,
                        headers => MAP {
                            {% for key, value in headers.items() %}
                            '{{ key }}': '{{ value }}'{% if not loop.last %},{% endif %}
                            {% endfor %}
                        },
                        params => MAP {
                            {% for key, value in params.items() %}
                            '{{ key }}': {{ value|tojson }}{% if not loop.last %},{% endif %}
                            {% endfor %}
                        }
                    ) AS res
                    """
                    context = {
                        "headers": headers_dict,
                        "params": json_data,
                    }
                    sql = self.db.execute(
                        "SELECT minijinja_render(?, ?) AS sql",
                        (template, json.dumps(context)),
                    ).fetchone()[0]
                    result = self.db.execute(sql, (url,)).fetchone()[0]
                else:
                    # Use http_post_form for form data
                    params_dict = params or {}
                    template = """
                    SELECT http_post_form(
                        ?,
                        headers => MAP {
                            {% for key, value in headers.items() %}
                            '{{ key }}': '{{ value }}'{% if not loop.last %},{% endif %}
                            {% endfor %}
                        },
                        params => MAP {
                            {% for key, value in params.items() %}
                            '{{ key }}': '{{ value }}'{% if not loop.last %},{% endif %}
                            {% endfor %}
                        }
                    ) AS res
                    """
                    context = {
                        "headers": headers_dict,
                        "params": params_dict,
                    }
                    sql = self.db.execute(
                        "SELECT minijinja_render(?, ?) AS sql",
                        (template, json.dumps(context)),
                    ).fetchone()[0]
                    result = self.db.execute(sql, (url,)).fetchone()[0]
            else:
                return f"❌ Error: Method {method} not supported. Use GET or POST."

            # Parse response - http_client returns structured data
            # Try to extract status, reason, body from response
            result_json = json.dumps(result) if not isinstance(result, str) else result
            
            # Use jsonata if available, otherwise parse JSON directly
            try:
                status = self.db.execute(
                    "SELECT jsonata('$.status', ?::JSON)::INT AS status",
                    (result_json,),
                ).fetchone()[0] or 0
                
                reason = self.db.execute(
                    "SELECT jsonata('$.reason', ?::JSON)::VARCHAR AS reason",
                    (result_json,),
                ).fetchone()[0] or "Unknown"
                
                body = self.db.execute(
                    "SELECT jsonata('$.body', ?::JSON) AS body",
                    (result_json,),
                ).fetchone()[0]
            except Exception:
                # Fallback: parse JSON directly if jsonata fails
                try:
                    parsed = json.loads(result_json) if isinstance(result_json, str) else result_json
                    if isinstance(parsed, dict):
                        status = parsed.get("status", 0)
                        reason = parsed.get("reason", "Unknown")
                        body = parsed.get("body", "")
                    else:
                        status = 200
                        reason = "OK"
                        body = result_json
                except Exception:
                    status = 200
                    reason = "OK"
                    body = result_json

            # Format body content - try to pretty-print if it's JSON
            try:
                if isinstance(body, str):
                    # Try parsing as JSON for pretty formatting
                    body_json = json.loads(body)
                    content = json.dumps(body_json, indent=2)[:500]
                else:
                    content = json.dumps(body, indent=2)[:500]
            except Exception:
                content = str(body)[:500] if body else ""

            # Learn from successful call
            await self._learn_api_pattern(
                url, method, headers, json_data, params, status
            )

            return f"""✅ API {method} {url}:

Status: {status} {reason}

Response:
{content}...
"""
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return f"❌ API request failed: {e}"

    async def _learn_api_pattern(
        self,
        url: str,
        method: str,
        headers: Optional[dict],
        json_data: Optional[dict],
        params: Optional[dict],
        response_status: int,
    ) -> None:
        """Learn from successful API call."""
        if response_status < 200 or response_status >= 300:
            return  # Only learn from successful calls

        from urllib.parse import urlparse

        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        endpoint = parsed.path

        # Detect auth type
        auth_type = None
        required_headers = {}
        if headers:
            if "Authorization" in headers:
                if headers["Authorization"].startswith("Bearer"):
                    auth_type = "bearer"
                elif headers["Authorization"].startswith("Basic"):
                    auth_type = "basic"
            for key in ["X-API-Key", "API-Key", "apikey"]:
                if key in headers:
                    auth_type = "api_key"
                    required_headers[key] = "REQUIRED"

        # Check if pattern exists
        existing = self.db.execute(
            "SELECT id, success_count FROM api_patterns WHERE base_url = ? AND endpoint = ? AND method = ?",
            (base_url, endpoint, method),
        ).fetchone()

        if existing:
            # Update existing pattern
            self.db.execute(
                "UPDATE api_patterns SET success_count = success_count + 1, last_used = now() WHERE id = ?",
                (existing[0],),
            )
        else:
            # Store new pattern
            self.db.execute(
                """
                INSERT INTO api_patterns (base_url, endpoint, method, auth_type, required_headers, params_schema, body_schema)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    base_url,
                    endpoint,
                    method,
                    auth_type,
                    json.dumps(required_headers) if required_headers else None,
                    json.dumps(params) if params else None,
                    json.dumps(json_data) if json_data else None,
                ),
            )

    async def _suggest_api_pattern(
        self, url: str, method: str = "GET"
    ) -> Optional[dict]:
        """Suggest API pattern based on learned patterns."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        endpoint = parsed.path

        # Find matching pattern
        pattern = self.db.execute(
            """
            SELECT auth_type, required_headers, params_schema, body_schema, success_count
            FROM api_patterns 
            WHERE base_url = ? AND endpoint = ? AND method = ?
            ORDER BY success_count DESC, last_used DESC
            LIMIT 1
        """,
            (base_url, endpoint, method),
        ).fetchone()

        if pattern:
            return {
                "auth_type": pattern[0],
                "required_headers": json.loads(pattern[1]) if pattern[1] else {},
                "params_example": json.loads(pattern[2]) if pattern[2] else {},
                "body_example": json.loads(pattern[3]) if pattern[3] else {},
                "confidence": min(pattern[4] / 10.0, 1.0),  # Normalize to 0-1
            }

        return None

    async def _discover_api(self, base_url: str) -> str:
        """Discover API endpoints and patterns using DuckDB http_client extension."""
        discovered = []

        # Try common OpenAPI/Swagger endpoints
        for path in ["/openapi.json", "/swagger.json", "/api-docs"]:
            try:
                url = f"{base_url}{path}"
                result = self.db.execute("SELECT http_get(?) AS res", (url,)).fetchone()[
                    0
                ]
                result_json = json.dumps(result) if not isinstance(result, str) else result
                status = self.db.execute(
                    "SELECT jsonata('$.status', ?::JSON)::INT AS status",
                    (result_json,),
                ).fetchone()[0] or 0
                if status == 200:
                    return f"✅ Found OpenAPI spec at {url}"
            except Exception:
                pass

        # Try common endpoints
        for path in ["/api", "/api/v1", "/health", "/status"]:
            try:
                url = f"{base_url}{path}"
                result = self.db.execute("SELECT http_get(?) AS res", (url,)).fetchone()[
                    0
                ]
                result_json = json.dumps(result) if not isinstance(result, str) else result
                status = self.db.execute(
                    "SELECT jsonata('$.status', ?::JSON)::INT AS status",
                    (result_json,),
                ).fetchone()[0] or 0
                if status == 200:
                    discovered.append(f"{path} (GET)")
            except Exception:
                pass

        if discovered:
            return "✅ Discovered endpoints:\n" + "\n".join(discovered)
        return "❌ No endpoints discovered"

    async def _execute_workflow(
        self, workflow_id: str, variables: Optional[dict] = None
    ) -> str:
        """Execute a workflow."""
        from .workflows import WorkflowEngine

        engine = WorkflowEngine(self)
        try:
            result = await engine.execute_workflow(workflow_id, variables)
            return f"""✅ Workflow {workflow_id} executed successfully:

Execution ID: {result['execution_id']}
Results: {json.dumps(result['results'], indent=2)[:500]}...
"""
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return f"❌ Workflow execution failed: {e}"

    async def close(self):
        """Close resources properly."""
        if hasattr(self, "db"):
            self.db.close()

    def __del__(self):
        """Cleanup resources - fallback only."""
        # Just pass - proper cleanup should use close() method
        pass


# ============================================================================
# Public API
# ============================================================================


def create_web_tool(config: Optional[WebConfig] = None) -> Web:
    """Create a Web tool instance.

    Args:
        config: Optional configuration. Defaults to WebConfig().

    Returns:
        Configured Web instance.
    """
    return Web(config)


# Default instance for convenience (lazy initialization)
# web_tool = Web()  # Commented out to avoid initialization on import
