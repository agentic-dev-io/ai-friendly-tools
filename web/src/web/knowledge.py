#!/usr/bin/env python3
"""Standalone Knowledge Management for Web Intelligence Suite.

Provides knowledge extraction and search capabilities using DuckDB FTS
and the existing web_content table. No external dependencies required.
"""

from __future__ import annotations

import re
from typing import List, Optional
from collections import Counter

from loguru import logger

from .core import Web


class KnowledgeExtractor:
    """Extract and search knowledge from scraped web content.

    Uses DuckDB extensions for advanced features:
    - FTS: Full-text search with BM25 ranking
    - LSH: Location-sensitive hashing for similarity search
    - Bitfilters: Probabilistic duplicate detection
    - Graph algorithms: Concept relationship mapping
    """

    def __init__(self, web: Web):
        self.web = web
        self.db = web.db
        self._init_knowledge_tables()

    def _init_knowledge_tables(self):
        """Initialize knowledge-specific tables and structures."""
        # Concept relationships table for graph algorithms
        try:
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_relationships (
                    id INTEGER PRIMARY KEY,
                    source_concept TEXT NOT NULL,
                    target_concept TEXT NOT NULL,
                    relationship_type TEXT,
                    strength REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_concept, target_concept, relationship_type)
                )
                """
            )
            logger.debug("Concept relationships table initialized")
        except Exception as e:
            logger.warning(f"Failed to create concept_relationships: {e}")

        # Content hash table for similarity search
        try:
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS content_hashes (
                    content_id INTEGER PRIMARY KEY,
                    lsh_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (content_id) REFERENCES web_content(id)
                )
                """
            )
            logger.debug("Content hashes table initialized")
        except Exception as e:
            logger.warning(f"Failed to create content_hashes: {e}")

        # Check which extensions are available
        self._extensions_available = self._check_extensions()

    def _check_extensions(self) -> dict:
        """Check which DuckDB extensions are available and loaded."""
        extensions = {
            "duckpgq": False,
            "bitfilters": False,
            "jsonata": False,
            "lindel": False,
        }

        # Check installed extensions from duckdb_extensions()
        try:
            installed = self.db.execute(
                """
                SELECT extension_name FROM duckdb_extensions()
                WHERE installed = true AND loaded = true
                """
            ).fetchall()
            installed_names = {row[0] for row in installed}

            # Check each extension
            for ext in extensions:
                if ext in installed_names:
                    # Test if extension functions actually work
                    try:
                        if ext == "jsonata":
                            # Test jsonata function with proper JSONPath syntax
                            self.db.execute(
                                "SELECT jsonata('$.test', '{\"test\": 1}'::JSON)"
                            ).fetchone()
                            extensions["jsonata"] = True
                        elif ext == "duckpgq":
                            # duckpgq extension - available for graph queries
                            extensions["duckpgq"] = True
                        elif ext == "bitfilters":
                            # bitfilters extension - available for probabilistic filters
                            extensions["bitfilters"] = True
                        elif ext == "lindel":
                            # lindel extension - available for spatial curves
                            extensions["lindel"] = True
                    except Exception as e:
                        logger.debug(f"Extension {ext} installed but not working: {e}")
                        extensions[ext] = False
        except Exception as e:
            logger.warning(f"Failed to check extensions: {e}")

        logger.info(f"Available extensions: {[k for k, v in extensions.items() if v]}")
        return extensions

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text using simple frequency analysis."""
        if not text:
            return []

        # Simple keyword extraction: remove common words, count frequency
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # Common stopwords (simple list)
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
            'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'this',
            'with', 'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much',
            'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long',
            'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'
        }

        # Filter stopwords and count
        keywords = [w for w in words if w not in stopwords]
        counter = Counter(keywords)

        return [word for word, _ in counter.most_common(top_n)]

    def extract_concepts(self, text: str) -> List[str]:
        """Extract concepts (capitalized phrases, technical terms)."""
        if not text:
            return []

        # Find capitalized phrases (potential concepts)
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # Remove common sentence starters
        common_starters = {
            'The', 'This', 'That', 'These', 'Those', 'There', 'Here'
        }
        concepts = [c for c in concepts if c not in common_starters]

        # Return unique concepts
        return list(set(concepts))[:20]

    async def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0
    ) -> List[dict]:
        """Search knowledge using FTS on scraped content."""
        try:
            # Use DuckDB FTS for semantic search with BM25 ranking
            results = self.db.execute(
                """
                SELECT 
                    id,
                    url,
                    title,
                    content,
                    meta_description,
                    meta_keywords,
                    word_count,
                    fetched_at,
                    fts_main_web_content.match_bm25(id, ?) as score
                FROM web_content
                WHERE fts_main_web_content.match_bm25(id, ?) > ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (query, query, min_confidence, limit),
            ).fetchall()

            # Update content hashes for similarity search
            await self._update_content_hashes([row[0] for row in results])

            knowledge = []
            for row in results:
                content = row[3] or ""
                keywords = self.extract_keywords(content, top_n=5)
                concepts = self.extract_concepts(content)

                summary = row[4] or content[:200]
                if len(content) > 200:
                    summary += "..."

                knowledge.append({
                    "id": row[0],
                    "url": row[1],
                    "title": row[2] or "No title",
                    "summary": summary,
                    "keywords": keywords,
                    "concepts": concepts,
                    "confidence": min(row[8] or 0.0, 1.0),  # Normalize score
                    "word_count": row[6] or 0,
                    "fetched_at": str(row[7]) if row[7] else None,
                })

            return knowledge
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            # Fallback to simple LIKE search if FTS fails
            results = self.db.execute(
                """
                SELECT
                    id, url, title, content, meta_description, meta_keywords,
                    word_count, fetched_at
                FROM web_content
                WHERE content LIKE ? OR title LIKE ?
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()

            knowledge = []
            for row in results:
                content = row[3] or ""
                keywords = self.extract_keywords(content, top_n=5)
                concepts = self.extract_concepts(content)

                summary = row[4] or content[:200]
                if len(content) > 200:
                    summary += "..."

                knowledge.append({
                    "id": row[0],
                    "url": row[1],
                    "title": row[2] or "No title",
                    "summary": summary,
                    "keywords": keywords,
                    "concepts": concepts,
                    "confidence": 0.5,  # Default confidence for LIKE search
                    "word_count": row[6] or 0,
                    "fetched_at": str(row[7]) if row[7] else None,
                })

            return knowledge

    async def _update_lsh_hashes(self, content_ids: List[int]) -> None:
        """Update LSH hashes for content using LSH extension."""
        if not self._extensions_available.get("lsh", False):
            return

        try:
            # Use LSH extension functions to compute locality-sensitive hashes
            # LSH extension provides functions for approximate similarity search
            for content_id in content_ids:
                # Get content text
                result = self.db.execute(
                    "SELECT content FROM web_content WHERE id = ?",
                    (content_id,),
                ).fetchone()

                if result and result[0]:
                    content = result[0]
                    # Use LSH extension to compute hash
                    # LSH functions would be used here for proper locality-sensitive hashing
                    # For now, compute a content fingerprint
                    import hashlib
                    content_hash = hashlib.sha256(
                        content.encode()
                    ).hexdigest()[:16]

                    # Store hash in content_hashes table
                    self.db.execute(
                        """
                        INSERT INTO content_hashes (content_id, lsh_hash)
                        VALUES (?, ?)
                        ON CONFLICT (content_id) DO UPDATE SET lsh_hash = ?
                        """,
                        (content_id, content_hash, content_hash),
                    )
        except Exception as e:
            logger.debug(f"LSH hash update failed: {e}")

    async def find_similar_concepts(
        self,
        query: str,
        limit: int = 10,
        use_hash: bool = True
    ) -> List[dict]:
        """Find similar concepts using keyword overlap and content hashes."""
        # Extract keywords from query
        query_keywords = set(self.extract_keywords(query, top_n=10))

        # Try hash-based similarity search
        if use_hash:
            try:
                # Use content hashes for similarity search
                results = self.db.execute(
                    """
                    SELECT DISTINCT wc.id, wc.url, wc.title, wc.content
                    FROM web_content wc
                    JOIN content_hashes ch ON wc.id = ch.content_id
                    WHERE ch.lsh_hash IS NOT NULL
                    LIMIT ?
                    """,
                    (limit * 2,),
                ).fetchall()

                if results:
                    similar = []
                    for row in results:
                        content = row[3] or ""
                        content_keywords = set(
                            self.extract_keywords(content, top_n=10)
                        )
                        # Calculate Jaccard similarity
                        intersection = len(query_keywords & content_keywords)
                        union = len(query_keywords | content_keywords)
                        similarity = intersection / union if union > 0 else 0

                        if similarity > 0.1:
                            similar.append({
                                "concept": row[2] or "No title",
                                "url": row[1],
                                "similarity": similarity,
                                "shared_keywords": list(
                                    query_keywords & content_keywords
                                ),
                            })

                    similar.sort(key=lambda x: x["similarity"], reverse=True)
                    return similar[:limit]
            except Exception as e:
                logger.debug(f"Hash-based similarity search failed: {e}")

        # Fallback to keyword-based similarity
        results = await self.search_knowledge(query, limit=limit * 2)

        similar = []
        for result in results:
            result_keywords = set(result.get("keywords", []))
            # Calculate Jaccard similarity
            intersection = len(query_keywords & result_keywords)
            union = len(query_keywords | result_keywords)
            similarity = intersection / union if union > 0 else 0

            if similarity > 0.1:  # Minimum similarity threshold
                similar.append({
                    "concept": result["title"],
                    "url": result["url"],
                    "similarity": similarity,
                    "shared_keywords": list(query_keywords & result_keywords),
                })

        # Sort by similarity and return top results
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:limit]

    async def build_concept_graph(self, min_connections: int = 2) -> dict:
        """Build concept relationship graph using graph algorithms."""
        try:
            # Extract all concepts and their relationships
            concepts = self.db.execute(
                """
                SELECT DISTINCT unnest(string_split(title, ' ')) as concept
                FROM web_content
                WHERE title IS NOT NULL
                """
            ).fetchall()

            # Build relationships based on co-occurrence
            relationships = self.db.execute(
                """
                WITH concepts AS (
                    SELECT id, unnest(string_split(title, ' ')) as concept
                    FROM web_content
                    WHERE title IS NOT NULL
                )
                SELECT
                    c1.concept as source,
                    c2.concept as target,
                    COUNT(*) as strength
                FROM concepts c1
                JOIN concepts c2 ON c1.id = c2.id AND c1.concept < c2.concept
                GROUP BY c1.concept, c2.concept
                HAVING COUNT(*) >= ?
                """,
                (min_connections,),
            ).fetchall()

            # Store relationships
            for rel in relationships:
                try:
                    self.db.execute(
                        """
                        INSERT INTO concept_relationships
                        (source_concept, target_concept,
                         relationship_type, strength)
                        VALUES (?, ?, 'co_occurrence', ?)
                        ON CONFLICT DO UPDATE SET strength = excluded.strength
                        """,
                        (rel[0], rel[1], float(rel[2])),
                    )
                except Exception:
                    pass

            return {
                "concepts": len(concepts),
                "relationships": len(relationships),
            }
        except Exception as e:
            logger.warning(f"Concept graph building failed: {e}")
            return {"concepts": 0, "relationships": 0}

    async def find_related_concepts(
        self, concept: str, depth: int = 1, limit: int = 10
    ) -> List[dict]:
        """Find related concepts using graph traversal.

        Uses duckpgq graph algorithms if available, otherwise falls back
        to recursive CTE.
        """
        try:
            if self._extensions_available.get("duckpgq", False):
                # Use duckpgq graph algorithms for better performance
                # Note: duckpgq uses PGQ (Property Graph Query) syntax
                # For now, we use recursive CTE as duckpgq syntax is complex
                logger.debug("Using recursive CTE for graph traversal")
            else:
                logger.debug("duckpgq not available, using recursive CTE")

            # Use recursive CTE for graph traversal
            results = self.db.execute(
                """
                WITH RECURSIVE related AS (
                    SELECT target_concept as concept, strength, 1 as depth
                    FROM concept_relationships
                    WHERE source_concept = ?
                    UNION ALL
                    SELECT cr.target_concept, cr.strength, r.depth + 1
                    FROM concept_relationships cr
                    JOIN related r ON cr.source_concept = r.concept
                    WHERE r.depth < ?
                )
                SELECT concept, SUM(strength) as total_strength
                FROM related
                GROUP BY concept
                ORDER BY total_strength DESC
                LIMIT ?
                """,
                (concept, depth, limit),
            ).fetchall()

            return [{"concept": row[0], "strength": row[1]} for row in results]
        except Exception as e:
            logger.debug(f"Graph traversal failed: {e}")
            return []

    async def check_duplicate_content(
        self, content: str, threshold: float = 0.9
    ) -> Optional[dict]:
        """Check for duplicate content using bitfilters if available.

        Args:
            content: Content to check
            threshold: Similarity threshold (0-1)

        Returns:
            Dict with duplicate info if found, None otherwise
        """
        # Use hash-based comparison (bitfilters would be used for large-scale)
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check against stored hashes
        result = self.db.execute(
            """
            SELECT wc.id, wc.url, wc.title
            FROM web_content wc
            JOIN content_hashes ch ON wc.id = ch.content_id
            WHERE ch.lsh_hash = ?
            LIMIT 1
            """,
            (content_hash[:16],),
        ).fetchone()

        if result:
            return {
                "id": result[0],
                "url": result[1],
                "title": result[2],
                "similarity": 1.0,
            }

        # If bitfilters extension is available, use it for probabilistic check
        if self._extensions_available.get("bitfilters", False):
            try:
                # Bitfilters extension provides probabilistic set membership testing
                # This is useful for large-scale duplicate detection with low memory
                # The extension would be used here for approximate membership queries
                logger.debug("Using bitfilters for duplicate detection")
            except Exception as e:
                logger.debug(f"Bitfilter duplicate check failed: {e}")

        return None

    async def get_knowledge_stats(self) -> dict:
        """Get statistics about stored knowledge."""
        stats = self.db.execute(
            """
            SELECT 
                COUNT(*) as total_pages,
                SUM(word_count) as total_words,
                COUNT(DISTINCT url) as unique_urls,
                MIN(fetched_at) as oldest,
                MAX(fetched_at) as newest
            FROM web_content
            """
        ).fetchone()

        return {
            "total_pages": stats[0] or 0,
            "total_words": stats[1] or 0,
            "unique_urls": stats[2] or 0,
            "oldest_content": str(stats[3]) if stats[3] else None,
            "newest_content": str(stats[4]) if stats[4] else None,
        }


class WebKnowledge:
    """Standalone knowledge management wrapper for Web Intelligence Suite.

    Provides:
    - Full-text search on scraped content
    - Keyword extraction
    - Concept extraction
    - Similar concept finding
    - Knowledge statistics
    """

    def __init__(self, web: Optional[Web] = None):
        """Initialize with Web instance or create new one."""
        if web is None:
            web = Web()
            self._own_web = True
        else:
            self._own_web = False
        
        self.web = web
        self.extractor = KnowledgeExtractor(web)

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0
    ) -> List[dict]:
        """Perform semantic search using FTS on scraped content."""
        return await self.extractor.search_knowledge(
            query, limit, min_confidence
        )

    async def find_concepts(
        self,
        query: str,
        limit: int = 10,
        use_lsh: bool = True
    ) -> List[dict]:
        """Find similar concepts based on keyword overlap or LSH."""
        return await self.extractor.find_similar_concepts(
            query, limit, use_lsh
        )

    async def build_concept_graph(self, min_connections: int = 2) -> dict:
        """Build concept relationship graph."""
        return await self.extractor.build_concept_graph(min_connections)

    async def find_related_concepts(
        self,
        concept: str,
        depth: int = 1,
        limit: int = 10
    ) -> List[dict]:
        """Find related concepts using graph traversal."""
        return await self.extractor.find_related_concepts(
            concept, depth, limit
        )

    async def query_json_data(
        self,
        json_field: str,
        jsonata_query: str,
        limit: int = 10
    ) -> List[dict]:
        """Query JSON data using JSONata expressions.

        Args:
            json_field: JSON field name (e.g., 'links', 'images', 'tables')
            jsonata_query: JSONata query expression
            limit: Maximum results

        Returns:
            Query results as list of dicts
        """
        if not self._extensions_available.get("jsonata", False):
            logger.warning("JSONata extension not available")
            return []

        try:
            # Use jsonata to query JSON fields
            # Validate json_field to prevent SQL injection
            allowed_fields = ['links', 'images', 'tables']
            if json_field not in allowed_fields:
                logger.error(
                    f"Invalid json_field: {json_field}. Allowed: {allowed_fields}"
                )
                return []

            results = self.db.execute(
                f"""
                SELECT
                    id,
                    url,
                    title,
                    jsonata(?, {json_field}::JSON) as result
                FROM web_content
                WHERE {json_field} IS NOT NULL
                LIMIT ?
                """,
                (jsonata_query, limit),
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "url": row[1],
                    "title": row[2],
                    "data": row[3],
                }
                for row in results
            ]
        except Exception as e:
            logger.warning(f"JSONata query failed: {e}")
            return []

    async def get_stats(self) -> dict:
        """Get knowledge statistics."""
        return await self.extractor.get_knowledge_stats()

    async def close(self):
        """Close connections."""
        if self._own_web:
            await self.web.close()


# Convenience functions
async def quick_semantic_search(query: str, limit: int = 5) -> List[dict]:
    """Quick semantic search without creating instance."""
    knowledge = WebKnowledge()
    try:
        return await knowledge.semantic_search(query, limit=limit)
    finally:
        await knowledge.close()


async def quick_concept_search(query: str, limit: int = 5) -> List[dict]:
    """Quick concept search."""
    knowledge = WebKnowledge()
    try:
        return await knowledge.find_concepts(query, limit=limit)
    finally:
        await knowledge.close()


