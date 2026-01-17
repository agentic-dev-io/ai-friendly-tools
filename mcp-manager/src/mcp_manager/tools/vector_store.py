"""Vector store for semantic search using DuckDB native storage.

This module provides the VectorStore class that manages embedding storage
and retrieval in DuckDB, using BLOB columns for efficient vector storage
and cosine similarity for search ranking.

No external vector database (like FAISS) is required - all operations
use native DuckDB capabilities.
"""

from __future__ import annotations

import struct
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import duckdb
from loguru import logger
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from ..database import DatabaseConnectionPool

from .embeddings import (
    EmbeddingGenerator,
    EmbeddingGeneratorError,
    blob_to_embedding,
    embedding_to_blob,
    get_default_generator,
)


class SimilarityMetric(str, Enum):
    """Similarity metrics for vector search."""
    
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class VectorStoreConfig(BaseModel):
    """Configuration for vector store.
    
    Attributes:
        embedding_dim: Dimension of embedding vectors
        table_name: Name of the embeddings table
        similarity_metric: Metric to use for similarity calculations
        default_limit: Default number of results for search
        auto_embed: Whether to automatically generate embeddings on add
    """
    
    embedding_dim: int = Field(
        default=384,
        ge=1,
        description="Dimension of embedding vectors (384 for all-MiniLM-L6-v2)"
    )
    table_name: str = Field(
        default="tool_embeddings",
        description="Name of the embeddings table in DuckDB"
    )
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE,
        description="Similarity metric for vector search"
    )
    default_limit: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Default number of results for search queries"
    )
    auto_embed: bool = Field(
        default=True,
        description="Automatically generate embeddings when adding tools"
    )
    
    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        """Validate table name is a valid SQL identifier."""
        if not v or not v.strip():
            raise ValueError("Table name cannot be empty")
        # Basic SQL identifier validation
        if not v.replace("_", "").isalnum():
            raise ValueError("Table name must be alphanumeric with underscores only")
        return v.strip()


class VectorSearchResult(BaseModel):
    """Result from a vector similarity search.
    
    Attributes:
        tool_id: ID of the matched tool
        server_name: Name of the MCP server
        tool_name: Name of the tool
        description: Tool description
        similarity_score: Similarity score (higher is more similar)
        embedding: The stored embedding vector (optional)
    """
    
    tool_id: int = Field(description="Tool ID from mcp_tools table")
    server_name: str = Field(description="MCP server name")
    tool_name: str = Field(description="Tool name")
    description: Optional[str] = Field(default=None, description="Tool description")
    similarity_score: float = Field(description="Similarity score (0-1 for cosine)")
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Stored embedding vector"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "tool_id": 1,
                "server_name": "filesystem",
                "tool_name": "read_file",
                "description": "Read contents of a file",
                "similarity_score": 0.95,
            }
        }


class EmbeddingRecord(BaseModel):
    """Record of an embedding in the vector store.
    
    Attributes:
        id: Auto-generated record ID
        tool_id: Reference to mcp_tools table
        embedding: The embedding vector
        text_content: Original text used to generate embedding
        created_at: Timestamp when embedding was created
        updated_at: Timestamp when embedding was last updated
    """
    
    id: Optional[int] = Field(default=None, description="Record ID")
    tool_id: int = Field(description="Tool ID reference")
    embedding: list[float] = Field(description="Embedding vector")
    text_content: Optional[str] = Field(
        default=None,
        description="Original text used for embedding"
    )
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    pass


class TableNotFoundError(VectorStoreError):
    """Raised when the embeddings table doesn't exist."""
    pass


class EmbeddingNotFoundError(VectorStoreError):
    """Raised when an embedding is not found."""
    pass


class DimensionMismatchError(VectorStoreError):
    """Raised when embedding dimensions don't match."""
    pass


class VectorStore:
    """DuckDB-native vector store for semantic search.
    
    This class manages storage and retrieval of embedding vectors in DuckDB,
    using BLOB columns for efficient storage and SQL-based cosine similarity
    calculations for search.
    
    No external vector database is required - all operations use native DuckDB.
    
    Example:
        ```python
        from mcp_manager.database import DatabaseConnectionPool
        from mcp_manager.tools.vector_store import VectorStore
        
        pool = DatabaseConnectionPool()
        conn = pool.get_connection("mcp_tools")
        
        store = VectorStore(conn)
        store.initialize()
        
        # Add embedding for a tool
        embedding = [0.1, 0.2, ...]  # 384 dimensions
        store.add_embedding(tool_id=1, embedding=embedding)
        
        # Search for similar tools
        query_embedding = [0.15, 0.25, ...]
        results = store.search_similar(query_embedding, limit=5)
        ```
    
    Attributes:
        config: Vector store configuration
        conn: DuckDB connection
        generator: Embedding generator for auto-embed mode
    """
    
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        config: VectorStoreConfig | None = None,
        *,
        generator: EmbeddingGenerator | None = None,
    ) -> None:
        """Initialize vector store.
        
        Args:
            connection: Active DuckDB connection
            config: Vector store configuration (uses defaults if not provided)
            generator: Embedding generator for auto-embed mode
        """
        self.conn = connection
        self.config = config or VectorStoreConfig()
        self._generator = generator
        self._initialized = False
        
        logger.debug(
            "VectorStore initialized (table={}, dim={})",
            self.config.table_name,
            self.config.embedding_dim
        )
    
    @property
    def generator(self) -> EmbeddingGenerator:
        """Get or create embedding generator.
        
        Returns:
            EmbeddingGenerator instance
        """
        if self._generator is None:
            self._generator = get_default_generator()
        return self._generator
    
    @property
    def is_initialized(self) -> bool:
        """Check if the store is initialized."""
        return self._initialized
    
    def initialize(self) -> None:
        """Initialize the vector store schema.
        
        Creates the embeddings table if it doesn't exist and sets up
        necessary indexes for efficient search.
        
        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            logger.info("Initializing vector store schema")
            
            # Create sequence for auto-increment IDs
            self.conn.execute(f"""
                CREATE SEQUENCE IF NOT EXISTS seq_{self.config.table_name} START 1
            """)
            
            # Create embeddings table with BLOB for vector storage
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                    id INTEGER PRIMARY KEY DEFAULT nextval('seq_{self.config.table_name}'),
                    tool_id INTEGER NOT NULL UNIQUE,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL DEFAULT {self.config.embedding_dim},
                    text_content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tool_id) REFERENCES mcp_tools(id) ON DELETE CASCADE
                )
            """)
            
            # Create index on tool_id for fast lookups
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_tool_id 
                ON {self.config.table_name}(tool_id)
            """)
            
            self._initialized = True
            logger.info("Vector store schema initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize vector store: {}", e)
            raise VectorStoreError(f"Failed to initialize vector store: {e}") from e
    
    def add_embedding(
        self,
        tool_id: int,
        embedding: list[float],
        *,
        text_content: str | None = None,
    ) -> int:
        """Add an embedding for a tool.
        
        Args:
            tool_id: ID of the tool in mcp_tools table
            embedding: Embedding vector (must match configured dimension)
            text_content: Optional original text used for the embedding
            
        Returns:
            ID of the created embedding record
            
        Raises:
            DimensionMismatchError: If embedding dimension doesn't match config
            VectorStoreError: If insertion fails
            
        Example:
            ```python
            embedding = generator.generate_embedding("read file contents")
            record_id = store.add_embedding(tool_id=1, embedding=embedding)
            ```
        """
        self._ensure_initialized()
        
        # Validate embedding dimension
        if len(embedding) != self.config.embedding_dim:
            raise DimensionMismatchError(
                f"Expected {self.config.embedding_dim} dimensions, "
                f"got {len(embedding)}"
            )
        
        try:
            # Serialize embedding to BLOB
            embedding_blob = embedding_to_blob(embedding)
            
            # Check if embedding already exists for this tool
            existing = self.conn.execute(
                f"SELECT id FROM {self.config.table_name} WHERE tool_id = ?",
                [tool_id]
            ).fetchone()
            
            if existing:
                # Update existing embedding
                self.conn.execute(
                    f"""
                    UPDATE {self.config.table_name}
                    SET embedding = ?, text_content = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE tool_id = ?
                    """,
                    [embedding_blob, text_content, tool_id]
                )
                logger.debug("Updated embedding for tool_id={}", tool_id)
                return existing[0]
            
            # Insert new embedding
            self.conn.execute(
                f"""
                INSERT INTO {self.config.table_name} 
                (tool_id, embedding, embedding_dim, text_content)
                VALUES (?, ?, ?, ?)
                """,
                [tool_id, embedding_blob, len(embedding), text_content]
            )
            
            # Get the inserted ID
            result = self.conn.execute(
                f"SELECT id FROM {self.config.table_name} WHERE tool_id = ?",
                [tool_id]
            ).fetchone()
            
            record_id = result[0] if result else 0
            logger.debug(
                "Added embedding for tool_id={} (record_id={})",
                tool_id, record_id
            )
            return record_id
            
        except DimensionMismatchError:
            raise
        except Exception as e:
            logger.error("Failed to add embedding for tool_id={}: {}", tool_id, e)
            raise VectorStoreError(f"Failed to add embedding: {e}") from e
    
    def add_embedding_for_text(
        self,
        tool_id: int,
        text: str,
    ) -> int:
        """Generate and add embedding for text.
        
        Convenience method that generates the embedding automatically.
        
        Args:
            tool_id: ID of the tool in mcp_tools table
            text: Text to generate embedding from
            
        Returns:
            ID of the created embedding record
            
        Raises:
            EmbeddingGeneratorError: If embedding generation fails
            VectorStoreError: If insertion fails
        """
        embedding = self.generator.generate_embedding(text)
        return self.add_embedding(tool_id, embedding, text_content=text)
    
    def add_batch(
        self,
        tool_embeddings: list[tuple[int, list[float], str | None]],
    ) -> list[int]:
        """Add multiple embeddings in a batch.
        
        Args:
            tool_embeddings: List of (tool_id, embedding, text_content) tuples
            
        Returns:
            List of created record IDs
            
        Raises:
            DimensionMismatchError: If any embedding dimension doesn't match
            VectorStoreError: If insertion fails
        """
        self._ensure_initialized()
        
        record_ids: list[int] = []
        
        try:
            for tool_id, embedding, text_content in tool_embeddings:
                record_id = self.add_embedding(
                    tool_id, embedding, text_content=text_content
                )
                record_ids.append(record_id)
            
            logger.info("Added {} embeddings in batch", len(record_ids))
            return record_ids
            
        except Exception as e:
            logger.error("Batch embedding insertion failed: {}", e)
            raise VectorStoreError(f"Batch insertion failed: {e}") from e
    
    def get_embedding(self, tool_id: int) -> EmbeddingRecord | None:
        """Get embedding record for a tool.
        
        Args:
            tool_id: ID of the tool
            
        Returns:
            EmbeddingRecord if found, None otherwise
        """
        self._ensure_initialized()
        
        try:
            result = self.conn.execute(
                f"""
                SELECT id, tool_id, embedding, text_content, created_at, updated_at
                FROM {self.config.table_name}
                WHERE tool_id = ?
                """,
                [tool_id]
            ).fetchone()
            
            if not result:
                return None
            
            # Deserialize embedding from BLOB
            embedding = blob_to_embedding(result[2])
            
            return EmbeddingRecord(
                id=result[0],
                tool_id=result[1],
                embedding=embedding,
                text_content=result[3],
                created_at=result[4],
                updated_at=result[5]
            )
            
        except Exception as e:
            logger.error("Failed to get embedding for tool_id={}: {}", tool_id, e)
            raise VectorStoreError(f"Failed to get embedding: {e}") from e
    
    def delete_embedding(self, tool_id: int) -> bool:
        """Delete embedding for a tool.
        
        Args:
            tool_id: ID of the tool
            
        Returns:
            True if deleted, False if not found
        """
        self._ensure_initialized()
        
        try:
            result = self.conn.execute(
                f"DELETE FROM {self.config.table_name} WHERE tool_id = ? RETURNING id",
                [tool_id]
            ).fetchone()
            
            deleted = result is not None
            if deleted:
                logger.debug("Deleted embedding for tool_id={}", tool_id)
            
            return deleted
            
        except Exception as e:
            logger.error("Failed to delete embedding for tool_id={}: {}", tool_id, e)
            raise VectorStoreError(f"Failed to delete embedding: {e}") from e
    
    def search_similar(
        self,
        query_embedding: list[float],
        limit: int | None = None,
        *,
        min_similarity: float = 0.0,
        server_filter: str | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar tools using cosine similarity.
        
        Performs vector similarity search against all stored embeddings,
        joining with mcp_tools to return full tool information.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results (uses config default if not specified)
            min_similarity: Minimum similarity threshold (0-1)
            server_filter: Optional filter by server name
            
        Returns:
            List of VectorSearchResult sorted by similarity (highest first)
            
        Raises:
            DimensionMismatchError: If query dimension doesn't match
            VectorStoreError: If search fails
            
        Example:
            ```python
            query = generator.generate_embedding("read file contents")
            results = store.search_similar(query, limit=5)
            for r in results:
                print(f"{r.tool_name}: {r.similarity_score:.4f}")
            ```
        """
        self._ensure_initialized()
        
        # Validate query dimension
        if len(query_embedding) != self.config.embedding_dim:
            raise DimensionMismatchError(
                f"Expected {self.config.embedding_dim} dimensions, "
                f"got {len(query_embedding)}"
            )
        
        effective_limit = limit or self.config.default_limit
        
        try:
            # Serialize query embedding
            query_blob = embedding_to_blob(query_embedding)
            
            # Build query with optional server filter
            where_clauses = ["t.enabled = true"]
            params: list[Any] = [query_blob, query_blob]
            
            if server_filter:
                where_clauses.append("t.server_name = ?")
                params.append(server_filter)
            
            if min_similarity > 0:
                params.append(min_similarity)
                having_clause = "HAVING similarity >= ?"
            else:
                having_clause = ""
            
            params.append(effective_limit)
            
            # Use SQL-based cosine similarity calculation
            # For normalized vectors, cosine similarity = dot product
            sql = f"""
                WITH ranked AS (
                    SELECT 
                        t.id as tool_id,
                        t.server_name,
                        t.tool_name,
                        t.description,
                        e.embedding,
                        (
                            SELECT SUM(q.value * s.value)
                            FROM (
                                SELECT 
                                    row_number() OVER () as idx,
                                    unnest(list_transform(
                                        generate_series(1, {self.config.embedding_dim}),
                                        i -> CAST(
                                            get_bit(?, (i-1)*32 + 24) * 1 +
                                            get_bit(?, (i-1)*32 + 25) * 2 +
                                            get_bit(?, (i-1)*32 + 26) * 4 +
                                            get_bit(?, (i-1)*32 + 27) * 8
                                            AS FLOAT
                                        )
                                    )) as value
                            ) q
                            JOIN (
                                SELECT 
                                    row_number() OVER () as idx,
                                    unnest(list_transform(
                                        generate_series(1, {self.config.embedding_dim}),
                                        i -> CAST(
                                            get_bit(e.embedding, (i-1)*32 + 24) * 1
                                            AS FLOAT
                                        )
                                    )) as value
                            ) s ON q.idx = s.idx
                        ) as similarity
                    FROM {self.config.table_name} e
                    JOIN mcp_tools t ON e.tool_id = t.id
                    WHERE {' AND '.join(where_clauses)}
                )
                SELECT tool_id, server_name, tool_name, description, similarity
                FROM ranked
                {having_clause}
                ORDER BY similarity DESC
                LIMIT ?
            """
            
            # Alternative simpler approach: load embeddings and compute in Python
            # This is more reliable than complex SQL bit manipulation
            results = self._search_similar_python(
                query_embedding,
                effective_limit,
                min_similarity,
                server_filter
            )
            
            return results
            
        except DimensionMismatchError:
            raise
        except Exception as e:
            logger.error("Vector search failed: {}", e)
            raise VectorStoreError(f"Vector search failed: {e}") from e
    
    def _search_similar_python(
        self,
        query_embedding: list[float],
        limit: int,
        min_similarity: float,
        server_filter: str | None,
    ) -> list[VectorSearchResult]:
        """Perform similarity search with Python-based cosine calculation.
        
        This approach loads embeddings from DB and computes similarity in Python,
        which is more reliable than complex SQL operations on BLOB data.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            server_filter: Optional server name filter
            
        Returns:
            List of VectorSearchResult
        """
        # Build query
        where_clauses = ["t.enabled = true"]
        params: list[Any] = []
        
        if server_filter:
            where_clauses.append("t.server_name = ?")
            params.append(server_filter)
        
        sql = f"""
            SELECT 
                t.id,
                t.server_name,
                t.tool_name,
                t.description,
                e.embedding
            FROM {self.config.table_name} e
            JOIN mcp_tools t ON e.tool_id = t.id
            WHERE {' AND '.join(where_clauses)}
        """
        
        rows = self.conn.execute(sql, params).fetchall()
        
        # Calculate similarities
        results: list[tuple[float, Any]] = []
        
        for row in rows:
            tool_id, server_name, tool_name, description, embedding_blob = row
            
            # Deserialize stored embedding
            stored_embedding = blob_to_embedding(embedding_blob)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            
            if similarity >= min_similarity:
                results.append((similarity, (tool_id, server_name, tool_name, description)))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Convert to VectorSearchResult
        search_results: list[VectorSearchResult] = []
        for similarity, (tool_id, server_name, tool_name, description) in results[:limit]:
            search_results.append(VectorSearchResult(
                tool_id=tool_id,
                server_name=server_name,
                tool_name=tool_name,
                description=description,
                similarity_score=similarity,
            ))
        
        logger.debug(
            "Vector search returned {} results (min_sim={:.2f})",
            len(search_results), min_similarity
        )
        
        return search_results
    
    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def search_by_text(
        self,
        query_text: str,
        limit: int | None = None,
        *,
        min_similarity: float = 0.0,
        server_filter: str | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar tools by text query.
        
        Convenience method that generates embedding from text first.
        
        Args:
            query_text: Text query to search for
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            server_filter: Optional filter by server name
            
        Returns:
            List of VectorSearchResult sorted by similarity
            
        Example:
            ```python
            results = store.search_by_text("read file contents", limit=5)
            ```
        """
        if not query_text or not query_text.strip():
            logger.warning("Empty query text provided")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.generator.generate_embedding(query_text)
            
            # Perform vector search
            return self.search_similar(
                query_embedding,
                limit,
                min_similarity=min_similarity,
                server_filter=server_filter,
            )
            
        except EmbeddingGeneratorError as e:
            logger.error("Failed to generate query embedding: {}", e)
            raise VectorStoreError(f"Failed to generate query embedding: {e}") from e
    
    def count_embeddings(self) -> int:
        """Count total embeddings in store.
        
        Returns:
            Number of embeddings stored
        """
        self._ensure_initialized()
        
        try:
            result = self.conn.execute(
                f"SELECT COUNT(*) FROM {self.config.table_name}"
            ).fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error("Failed to count embeddings: {}", e)
            raise VectorStoreError(f"Failed to count embeddings: {e}") from e
    
    def has_embedding(self, tool_id: int) -> bool:
        """Check if a tool has an embedding.
        
        Args:
            tool_id: ID of the tool
            
        Returns:
            True if embedding exists
        """
        self._ensure_initialized()
        
        try:
            result = self.conn.execute(
                f"SELECT 1 FROM {self.config.table_name} WHERE tool_id = ?",
                [tool_id]
            ).fetchone()
            return result is not None
        except Exception as e:
            logger.error("Failed to check embedding existence: {}", e)
            return False
    
    def get_tools_without_embeddings(self) -> list[tuple[int, str, str, str | None]]:
        """Get tools that don't have embeddings yet.
        
        Returns:
            List of (tool_id, server_name, tool_name, description) tuples
        """
        self._ensure_initialized()
        
        try:
            results = self.conn.execute(
                f"""
                SELECT t.id, t.server_name, t.tool_name, t.description
                FROM mcp_tools t
                LEFT JOIN {self.config.table_name} e ON t.id = e.tool_id
                WHERE e.id IS NULL AND t.enabled = true
                """
            ).fetchall()
            
            return [(r[0], r[1], r[2], r[3]) for r in results]
            
        except Exception as e:
            logger.error("Failed to get tools without embeddings: {}", e)
            raise VectorStoreError(f"Failed to query tools: {e}") from e
    
    def rebuild_embeddings(
        self,
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> int:
        """Rebuild embeddings for all tools.
        
        Generates new embeddings for all tools in mcp_tools table.
        Existing embeddings are updated.
        
        Args:
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress
            
        Returns:
            Number of embeddings created/updated
        """
        self._ensure_initialized()
        
        try:
            # Get all enabled tools
            tools = self.conn.execute(
                """
                SELECT id, server_name, tool_name, description
                FROM mcp_tools
                WHERE enabled = true
                """
            ).fetchall()
            
            if not tools:
                logger.info("No tools found to embed")
                return 0
            
            logger.info("Rebuilding embeddings for {} tools", len(tools))
            
            # Prepare texts for embedding
            texts: list[str] = []
            tool_ids: list[int] = []
            
            for tool_id, server_name, tool_name, description in tools:
                # Create searchable text combining tool metadata
                text = f"{tool_name} {description or ''} {server_name}"
                texts.append(text)
                tool_ids.append(tool_id)
            
            # Generate embeddings in batch
            embeddings = self.generator.generate_batch(
                texts,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            
            # Store embeddings
            count = 0
            for tool_id, embedding, text in zip(tool_ids, embeddings, texts):
                if embedding:  # Skip empty embeddings
                    self.add_embedding(tool_id, embedding, text_content=text)
                    count += 1
            
            logger.info("Successfully rebuilt {} embeddings", count)
            return count
            
        except Exception as e:
            logger.error("Failed to rebuild embeddings: {}", e)
            raise VectorStoreError(f"Failed to rebuild embeddings: {e}") from e
    
    def clear(self) -> int:
        """Clear all embeddings from the store.
        
        Returns:
            Number of embeddings deleted
        """
        self._ensure_initialized()
        
        try:
            count = self.count_embeddings()
            self.conn.execute(f"DELETE FROM {self.config.table_name}")
            logger.info("Cleared {} embeddings from store", count)
            return count
        except Exception as e:
            logger.error("Failed to clear embeddings: {}", e)
            raise VectorStoreError(f"Failed to clear embeddings: {e}") from e
    
    def drop_table(self) -> None:
        """Drop the embeddings table.
        
        Use with caution - this permanently deletes all embeddings.
        """
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.config.table_name}")
            self.conn.execute(f"DROP SEQUENCE IF EXISTS seq_{self.config.table_name}")
            self._initialized = False
            logger.info("Dropped embeddings table: {}", self.config.table_name)
        except Exception as e:
            logger.error("Failed to drop embeddings table: {}", e)
            raise VectorStoreError(f"Failed to drop table: {e}") from e
    
    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized.
        
        Raises:
            VectorStoreError: If store is not initialized
        """
        if not self._initialized:
            # Try to check if table exists
            try:
                result = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = ?
                    """,
                    [self.config.table_name]
                ).fetchone()
                
                if result and result[0] > 0:
                    self._initialized = True
                else:
                    # Auto-initialize if table doesn't exist
                    self.initialize()
            except Exception:
                # Auto-initialize on any error
                self.initialize()
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        self._ensure_initialized()
        
        try:
            total_embeddings = self.count_embeddings()
            
            # Get tools without embeddings
            tools_without = len(self.get_tools_without_embeddings())
            
            # Get total tools
            total_tools = self.conn.execute(
                "SELECT COUNT(*) FROM mcp_tools WHERE enabled = true"
            ).fetchone()
            total_tools_count = total_tools[0] if total_tools else 0
            
            return {
                "total_embeddings": total_embeddings,
                "total_tools": total_tools_count,
                "tools_without_embeddings": tools_without,
                "coverage_percent": (
                    (total_embeddings / total_tools_count * 100)
                    if total_tools_count > 0 else 0.0
                ),
                "embedding_dimension": self.config.embedding_dim,
                "table_name": self.config.table_name,
                "similarity_metric": self.config.similarity_metric.value,
            }
            
        except Exception as e:
            logger.error("Failed to get store stats: {}", e)
            raise VectorStoreError(f"Failed to get stats: {e}") from e


def create_vector_store(
    connection: duckdb.DuckDBPyConnection,
    *,
    auto_initialize: bool = True,
    **config_kwargs: Any,
) -> VectorStore:
    """Factory function to create and optionally initialize a VectorStore.
    
    Args:
        connection: DuckDB connection
        auto_initialize: Whether to initialize the store immediately
        **config_kwargs: Additional configuration options
        
    Returns:
        Configured VectorStore instance
        
    Example:
        ```python
        store = create_vector_store(conn, embedding_dim=384)
        ```
    """
    config = VectorStoreConfig(**config_kwargs)
    store = VectorStore(connection, config)
    
    if auto_initialize:
        store.initialize()
    
    return store
