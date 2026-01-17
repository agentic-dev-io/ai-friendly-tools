"""Embedding functionality for semantic search in MCP tools."""

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from ..tools.schema import ToolSchema


class EmbeddingModel:
    """Simple embedding model using TF-IDF-like approach.
    
    This is a lightweight implementation that doesn't require external ML libraries.
    For production, consider using sentence-transformers or OpenAI embeddings.
    """

    def __init__(self, dimension: int = 128) -> None:
        """Initialize embedding model.
        
        Args:
            dimension: Size of embedding vectors
        """
        self.dimension = dimension
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}

    def embed(self, text: Optional[str]) -> list[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text:
            return [0.0] * self.dimension

        # Tokenize and hash
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        # Generate embedding using hash-based approach
        embedding = [0.0] * self.dimension
        for token in tokens:
            # Hash token to get position
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            pos = token_hash % self.dimension
            # Add weight based on term frequency
            weight = 1.0 / len(tokens)
            embedding[pos] += weight

        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(x * x for x in vec1))
        mag2 = math.sqrt(sum(x * x for x in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return max(0.0, min(1.0, dot_product / (mag1 * mag2)))

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return [t for t in tokens if len(t) > 1]


class EmbeddingCache:
    """LRU cache for embeddings."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize cache.
        
        Args:
            max_size: Maximum number of entries to cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[list[float]]:
        """Get embedding from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached embedding or None
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def store(self, key: str, embedding: list[float]) -> None:
        """Store embedding in cache.
        
        Args:
            key: Cache key
            embedding: Embedding to store
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                self._cache.popitem(last=False)
            self._cache[key] = embedding

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
        }


class ToolEmbedder:
    """Generate embeddings for MCP tools."""

    def __init__(self, model: Optional[EmbeddingModel] = None) -> None:
        """Initialize tool embedder.
        
        Args:
            model: Embedding model to use (creates default if not provided)
        """
        self.model = model or EmbeddingModel()
        self.cache = EmbeddingCache()

    def embed_tool(self, tool: ToolSchema) -> list[float]:
        """Generate embedding for a tool.
        
        Args:
            tool: Tool schema to embed
            
        Returns:
            Embedding vector
        """
        # Create cache key
        cache_key = f"{tool.server_name}:{tool.tool_name}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Build text representation
        text_parts = [
            tool.tool_name or "",
            tool.description or "",
        ]

        # Add parameter names and descriptions
        if tool.input_schema and "properties" in tool.input_schema:
            for param_name, param_info in tool.input_schema["properties"].items():
                text_parts.append(param_name)
                if isinstance(param_info, dict) and "description" in param_info:
                    text_parts.append(param_info["description"])

        text = " ".join(text_parts)
        embedding = self.model.embed(text)
        
        # Cache result
        self.cache.store(cache_key, embedding)
        
        return embedding

    def embed_tools(self, tools: list[ToolSchema]) -> list[list[float]]:
        """Generate embeddings for multiple tools.
        
        Args:
            tools: List of tool schemas
            
        Returns:
            List of embedding vectors
        """
        return [self.embed_tool(tool) for tool in tools]


@dataclass
class SemanticSearchResult:
    """Result from semantic search."""
    
    server: str
    tool: str
    description: str
    required_params: list[str] = field(default_factory=list)
    score: float = 0.0


class SemanticSearcher:
    """Perform semantic search over MCP tools."""

    def __init__(self, db_connection: Any) -> None:
        """Initialize searcher.
        
        Args:
            db_connection: DuckDB connection
        """
        self.conn = db_connection
        self.embedder = ToolEmbedder()

    def search(
        self,
        query: str,
        limit: int = 5,
        server: Optional[str] = None,
    ) -> list[SemanticSearchResult]:
        """Search for tools semantically.
        
        Args:
            query: Search query
            limit: Maximum results
            server: Optional server filter
            
        Returns:
            List of search results
        """
        if not query or not query.strip():
            return []

        # Get query embedding
        query_embedding = self.embedder.model.embed(query)

        # Build SQL query
        sql = """
            SELECT server_name, tool_name, description, required_params, embedding
            FROM mcp_tools
            WHERE enabled = true
        """
        params: list[Any] = []

        if server:
            sql += " AND server_name = ?"
            params.append(server)

        try:
            results = self.conn.execute(sql, params).fetchall()
        except Exception:
            # Fallback to basic search if no embedding column
            return self._fallback_search(query, limit, server)

        # Calculate similarities and rank
        scored_results: list[tuple[SemanticSearchResult, float]] = []
        for row in results:
            server_name, tool_name, description, required_params, embedding_data = row
            
            # Parse embedding
            try:
                if isinstance(embedding_data, str):
                    import json
                    tool_embedding = json.loads(embedding_data)
                elif embedding_data is None:
                    # Generate embedding on the fly
                    tool = ToolSchema(
                        server_name=server_name,
                        tool_name=tool_name,
                        description=description,
                    )
                    tool_embedding = self.embedder.embed_tool(tool)
                else:
                    tool_embedding = list(embedding_data)
            except Exception:
                continue

            # Calculate similarity
            similarity = self.embedder.model.cosine_similarity(query_embedding, tool_embedding)
            
            result = SemanticSearchResult(
                server=server_name,
                tool=tool_name,
                description=description or "",
                required_params=required_params if required_params else [],
                score=similarity,
            )
            scored_results.append((result, similarity))

        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored_results[:limit]]

    def _fallback_search(
        self,
        query: str,
        limit: int,
        server: Optional[str],
    ) -> list[SemanticSearchResult]:
        """Fallback to text-based search.
        
        Args:
            query: Search query
            limit: Maximum results
            server: Optional server filter
            
        Returns:
            List of search results
        """
        sql = """
            SELECT server_name, tool_name, description, required_params
            FROM mcp_tools
            WHERE enabled = true
            AND (
                tool_name ILIKE '%' || ? || '%'
                OR description ILIKE '%' || ? || '%'
            )
        """
        params = [query, query]

        if server:
            sql += " AND server_name = ?"
            params.append(server)

        sql += " LIMIT ?"
        params.append(limit)

        try:
            results = self.conn.execute(sql, params).fetchall()
            return [
                SemanticSearchResult(
                    server=row[0],
                    tool=row[1],
                    description=row[2] or "",
                    required_params=row[3] if row[3] else [],
                    score=1.0,
                )
                for row in results
            ]
        except Exception:
            return []
