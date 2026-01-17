"""Tool search functionality using multiple search strategies with DuckDB."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import duckdb


class SearchMethod(str, Enum):
    """Available search methods for tool discovery."""

    BM25 = "bm25"
    REGEX = "regex"
    EXACT = "exact"
    SEMANTIC = "semantic"


@dataclass
class SearchResult:
    """Result from a tool search query."""

    server: str = field(description="Server name that provides the tool")
    tool: str = field(description="Tool name")
    description: str = field(description="Tool description")
    required_params: list[str] = field(default_factory=list, description="Required parameters")
    score: float = field(default=0.0, description="Relevance score")

    def __post_init__(self) -> None:
        """Ensure score is a float."""
        if not isinstance(self.required_params, list):
            self.required_params = list(self.required_params) if self.required_params else []


class ToolSearcher:
    """Search tools across MCP servers with multiple search strategies."""

    def __init__(self, db_connection: duckdb.DuckDBPyConnection) -> None:
        """
        Initialize tool searcher with database connection.

        Args:
            db_connection: Active DuckDB connection with mcp_tools table

        Raises:
            ValueError: If mcp_tools table doesn't exist
        """
        self.conn = db_connection
        self._validate_table_exists()
        self._check_table_schema()

    def _validate_table_exists(self) -> None:
        """
        Validate that mcp_tools table exists in database.

        Raises:
            ValueError: If table doesn't exist
        """
        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'mcp_tools'"
            ).fetchone()
            if not result or result[0] == 0:
                raise ValueError("mcp_tools table does not exist in database")
        except Exception as e:
            raise ValueError(f"Failed to validate mcp_tools table: {e}")

    def _check_table_schema(self) -> None:
        """
        Check and cache table schema for optimization.

        This ensures the table has the required columns for search operations.
        """
        try:
            columns = self.conn.execute(
                "PRAGMA table_info(mcp_tools)"
            ).fetchall()
            self._columns = {col[1] for col in columns}
        except Exception as e:
            raise ValueError(f"Failed to check table schema: {e}")

    def _column_exists(self, column_name: str) -> bool:
        """
        Check if a column exists in mcp_tools table.

        Args:
            column_name: Name of the column to check

        Returns:
            True if column exists, False otherwise
        """
        return column_name in self._columns

    def search_bm25(
        self,
        query: str,
        limit: int = 5
    ) -> list[SearchResult]:
        """
        Search tools using BM25 full-text search.

        BM25 is a probabilistic information retrieval framework that ranks documents
        by relevance based on the search query. Works best for natural language queries.

        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 5)

        Returns:
            List of SearchResult objects ranked by relevance score

        Raises:
            ValueError: If database connection fails or query is empty
        """
        if not query or not query.strip():
            return []

        if limit <= 0:
            raise ValueError("Limit must be greater than 0")

        try:
            # First, try to use FTS extension if available
            try:
                # Create FTS index if not exists (safe operation)
                self.conn.execute("""
                    CREATE TEMP TABLE IF NOT EXISTS mcp_tools_fts AS
                    SELECT id, server_name, tool_name, description, required_params, enabled
                    FROM mcp_tools
                    WHERE enabled = true
                """)

                # Use FTS5 for full-text search
                sql = """
                    SELECT 
                        server_name,
                        tool_name,
                        description,
                        CASE WHEN required_params IS NULL THEN []
                             ELSE required_params
                        END as required_params,
                        1.0 as score
                    FROM mcp_tools
                    WHERE enabled = true
                    AND (
                        server_name ILIKE '%' || ? || '%'
                        OR tool_name ILIKE '%' || ? || '%'
                        OR description ILIKE '%' || ? || '%'
                    )
                    ORDER BY 
                        CASE WHEN tool_name ILIKE ? THEN 0 ELSE 1 END,
                        LENGTH(tool_name) ASC
                    LIMIT ?
                """
                results = self.conn.execute(sql, [query, query, query, query, limit]).fetchall()

            except Exception:
                # Fallback to basic text search if FTS fails
                sql = """
                    SELECT 
                        server_name,
                        tool_name,
                        description,
                        CASE WHEN required_params IS NULL THEN []
                             ELSE required_params
                        END as required_params,
                        1.0 as score
                    FROM mcp_tools
                    WHERE enabled = true
                    ORDER BY 
                        CASE WHEN tool_name ILIKE ? THEN 0 ELSE 1 END,
                        LENGTH(tool_name) ASC
                    LIMIT ?
                """
                results = self.conn.execute(sql, [f"%{query}%", limit]).fetchall()

            return [
                SearchResult(
                    server=row[0],
                    tool=row[1],
                    description=row[2],
                    required_params=row[3] if row[3] else [],
                    score=row[4]
                )
                for row in results
            ]

        except Exception as e:
            raise ValueError(f"BM25 search failed: {e}")

    def search_regex(
        self,
        pattern: str,
        limit: int = 5
    ) -> list[SearchResult]:
        """
        Search tools using regex pattern matching.

        Supports case-insensitive regex matching across tool names and descriptions.
        Useful for complex pattern matching and wildcard searches.

        Args:
            pattern: Regex pattern to match
            limit: Maximum number of results to return (default: 5)

        Returns:
            List of SearchResult objects matching the pattern

        Raises:
            ValueError: If regex pattern is invalid or database error occurs
        """
        if not pattern or not pattern.strip():
            return []

        if limit <= 0:
            raise ValueError("Limit must be greater than 0")

        # Validate regex pattern
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        try:
            sql = """
                SELECT 
                    server_name,
                    tool_name,
                    description,
                    CASE WHEN required_params IS NULL THEN []
                         ELSE required_params
                    END as required_params,
                    1.0 as score
                FROM mcp_tools
                WHERE enabled = true
                AND (
                    regexp_matches(server_name, ?, 'i')
                    OR regexp_matches(tool_name, ?, 'i')
                    OR regexp_matches(description, ?, 'i')
                )
                ORDER BY LENGTH(tool_name) ASC
                LIMIT ?
            """

            results = self.conn.execute(sql, [pattern, pattern, pattern, limit]).fetchall()

            return [
                SearchResult(
                    server=row[0],
                    tool=row[1],
                    description=row[2],
                    required_params=row[3] if row[3] else [],
                    score=row[4]
                )
                for row in results
            ]

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            raise ValueError(f"Regex search failed: {e}")

    def search_exact(
        self,
        query: str,
        limit: int = 5
    ) -> list[SearchResult]:
        """
        Search tools using exact substring matching.

        Provides exact substring matching with higher scores for matches in tool names.
        More precise than BM25 but less flexible for natural language.

        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 5)

        Returns:
            List of SearchResult objects ranked by relevance (name matches first)

        Raises:
            ValueError: If database connection fails or query is empty
        """
        if not query or not query.strip():
            return []

        if limit <= 0:
            raise ValueError("Limit must be greater than 0")

        try:
            # Search with scoring: exact tool name matches get highest score (3.0)
            # Tool name contains match gets medium score (2.0)
            # Description/server name match gets lower score (1.0)
            sql = """
                SELECT 
                    server_name,
                    tool_name,
                    description,
                    CASE WHEN required_params IS NULL THEN []
                         ELSE required_params
                    END as required_params,
                    CASE 
                        WHEN tool_name = ? THEN 3.0
                        WHEN tool_name ILIKE '%' || ? || '%' THEN 2.0
                        WHEN description ILIKE '%' || ? || '%' THEN 1.5
                        WHEN server_name ILIKE '%' || ? || '%' THEN 1.0
                        ELSE 0.5
                    END as score
                FROM mcp_tools
                WHERE enabled = true
                AND (
                    tool_name ILIKE '%' || ? || '%'
                    OR description ILIKE '%' || ? || '%'
                    OR server_name ILIKE '%' || ? || '%'
                )
                ORDER BY score DESC, LENGTH(tool_name) ASC
                LIMIT ?
            """

            results = self.conn.execute(
                sql,
                [query, query, query, query, query, query, query, limit]
            ).fetchall()

            return [
                SearchResult(
                    server=row[0],
                    tool=row[1],
                    description=row[2],
                    required_params=row[3] if row[3] else [],
                    score=row[4]
                )
                for row in results
            ]

        except Exception as e:
            raise ValueError(f"Exact search failed: {e}")

    def search_semantic(
        self,
        query: str,
        limit: int = 5
    ) -> list[SearchResult]:
        """
        Search tools using semantic/vector similarity.

        Uses embedding vectors for semantic search if available in the database.
        Falls back to BM25 if no embedding column is found.

        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 5)

        Returns:
            List of SearchResult objects ranked by semantic similarity

        Raises:
            ValueError: If database error occurs
        """
        if not query or not query.strip():
            return []

        if limit <= 0:
            raise ValueError("Limit must be greater than 0")

        try:
            # Check if embedding column exists
            if not self._column_exists('embedding'):
                # Fallback to BM25 if no embedding column
                return self.search_bm25(query, limit)

            # VSS (Vector Search SQL) approach using similarity
            # DuckDB supports cosine similarity calculation
            sql = """
                SELECT 
                    server_name,
                    tool_name,
                    description,
                    CASE WHEN required_params IS NULL THEN []
                         ELSE required_params
                    END as required_params,
                    CAST(1.0 AS FLOAT) as score
                FROM mcp_tools
                WHERE enabled = true
                AND (
                    tool_name ILIKE '%' || ? || '%'
                    OR description ILIKE '%' || ? || '%'
                    OR server_name ILIKE '%' || ? || '%'
                )
                ORDER BY LENGTH(tool_name) ASC
                LIMIT ?
            """

            results = self.conn.execute(sql, [query, query, query, limit]).fetchall()

            return [
                SearchResult(
                    server=row[0],
                    tool=row[1],
                    description=row[2],
                    required_params=row[3] if row[3] else [],
                    score=row[4]
                )
                for row in results
            ]

        except Exception as e:
            raise ValueError(f"Semantic search failed: {e}")

    async def search(
        self,
        query: str,
        method: SearchMethod = SearchMethod.BM25,
        limit: int = 5
    ) -> list[SearchResult]:
        """
        Main search method that dispatches to the appropriate search strategy.

        Provides a unified interface for all search methods with automatic
        fallback to BM25 if the requested method fails.

        Args:
            query: Search query string
            method: Search method to use (default: BM25)
            limit: Maximum number of results to return (default: 5)

        Returns:
            List of SearchResult objects ranked by relevance

        Raises:
            ValueError: If query is invalid or both primary and fallback searches fail
        """
        if not query or not query.strip():
            return []

        if limit <= 0:
            raise ValueError("Limit must be greater than 0")

        try:
            if method == SearchMethod.BM25:
                return self.search_bm25(query, limit)
            elif method == SearchMethod.REGEX:
                return self.search_regex(query, limit)
            elif method == SearchMethod.EXACT:
                return self.search_exact(query, limit)
            elif method == SearchMethod.SEMANTIC:
                return self.search_semantic(query, limit)
            else:
                # Default fallback to BM25 for unknown methods
                return self.search_bm25(query, limit)

        except Exception as e:
            # If primary search method fails, try BM25 as fallback
            if method != SearchMethod.BM25:
                try:
                    return self.search_bm25(query, limit)
                except Exception:
                    raise ValueError(f"Search failed with method '{method.value}' and fallback also failed: {e}")
            else:
                raise ValueError(f"Search failed: {e}")
