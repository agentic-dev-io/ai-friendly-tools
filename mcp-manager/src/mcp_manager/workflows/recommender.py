"""Tool recommendation engine based on usage patterns."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import duckdb
from loguru import logger
from pydantic import BaseModel, Field

from ..tools.schema import ToolSchema


class ToolRecommendation(BaseModel):
    """A recommended tool with context."""

    tool: str = Field(description="Tool name")
    server: str = Field(description="Server name")
    score: float = Field(
        description="Recommendation score (0-1)",
        ge=0.0,
        le=1.0
    )
    reason: str = Field(description="Human-readable reason for recommendation")
    confidence: float = Field(
        default=0.5,
        description="Confidence in the recommendation (0-1)",
        ge=0.0,
        le=1.0
    )
    frequency: int = Field(
        default=0,
        description="How often this tool was used in this context"
    )
    avg_success_rate: float = Field(
        default=0.0,
        description="Average success rate when used"
    )
    related_tools: list[str] = Field(
        default_factory=list,
        description="Other tools commonly used together"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool": self.tool,
            "server": self.server,
            "score": self.score,
            "reason": self.reason,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "avg_success_rate": self.avg_success_rate,
            "related_tools": self.related_tools,
        }


class ToolPair(BaseModel):
    """A pair of tools that are commonly used together."""

    tool_a: str = Field(description="First tool (server/tool)")
    tool_b: str = Field(description="Second tool (server/tool)")
    co_occurrence_count: int = Field(description="Times used together")
    avg_time_between_ms: float = Field(
        default=0.0,
        description="Average time between tool calls"
    )
    success_rate: float = Field(
        default=0.0,
        description="Success rate when used together"
    )
    direction_weight: float = Field(
        default=0.5,
        description="Weight indicating typical order (0=A->B, 1=B->A)"
    )

    @property
    def is_sequential(self) -> bool:
        """Check if tools are typically used in sequence."""
        return self.direction_weight < 0.3 or self.direction_weight > 0.7


class UsagePattern(BaseModel):
    """A discovered usage pattern."""

    pattern_id: str
    description: str
    tools: list[str] = Field(description="List of tools in the pattern")
    frequency: int = Field(description="How often this pattern occurs")
    avg_duration_ms: float = Field(description="Average total duration")
    success_rate: float = Field(description="Pattern success rate")
    context_hints: dict[str, Any] = Field(
        default_factory=dict,
        description="Context hints for when to use this pattern"
    )


class WorkflowRecommender:
    """
    Recommends tools based on usage history and patterns.
    
    Learns from the mcp_tool_history table to provide intelligent
    recommendations for next steps in workflows.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        history_window_hours: int = 168,  # 1 week
        min_occurrences: int = 2,
    ) -> None:
        """
        Initialize the recommender.

        Args:
            conn: DuckDB connection with history table
            history_window_hours: How far back to look for patterns
            min_occurrences: Minimum occurrences to consider a pattern
        """
        self.conn = conn
        self.history_window_hours = history_window_hours
        self.min_occurrences = min_occurrences
        self._pattern_cache: dict[str, list[ToolRecommendation]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes

    def recommend_next_tools(
        self,
        current_tool: Optional[str] = None,
        current_server: Optional[str] = None,
        recent_tools: Optional[list[tuple[str, str]]] = None,
        limit: int = 5,
    ) -> list[ToolRecommendation]:
        """
        Recommend next tools based on current context.

        Args:
            current_tool: Currently used tool name
            current_server: Currently used server name
            recent_tools: List of (server, tool) tuples recently used
            limit: Maximum recommendations to return

        Returns:
            List of ToolRecommendation objects sorted by score
        """
        recommendations: list[ToolRecommendation] = []

        # Build context key for caching
        context_key = self._build_context_key(
            current_tool, current_server, recent_tools
        )

        # Check cache
        if self._is_cache_valid() and context_key in self._pattern_cache:
            logger.debug(f"Using cached recommendations for {context_key}")
            return self._pattern_cache[context_key][:limit]

        logger.info(
            f"Generating recommendations for tool={current_tool}, "
            f"server={current_server}"
        )

        # Get recommendations from different sources
        if current_tool and current_server:
            # Recommend based on co-occurrence
            co_occurrence_recs = self._recommend_by_co_occurrence(
                current_server, current_tool, limit
            )
            recommendations.extend(co_occurrence_recs)

            # Recommend based on sequential patterns
            sequential_recs = self._recommend_by_sequence(
                current_server, current_tool, limit
            )
            recommendations.extend(sequential_recs)

        # If we have recent tool history, use that too
        if recent_tools and len(recent_tools) >= 2:
            pattern_recs = self._recommend_by_pattern(recent_tools, limit)
            recommendations.extend(pattern_recs)

        # Add popular tools as fallback
        if len(recommendations) < limit:
            popular_recs = self._recommend_popular_tools(
                limit - len(recommendations),
                exclude=[(r.server, r.tool) for r in recommendations]
            )
            recommendations.extend(popular_recs)

        # Deduplicate and merge scores
        merged = self._merge_recommendations(recommendations)

        # Sort by score
        sorted_recs = sorted(merged, key=lambda r: r.score, reverse=True)[:limit]

        # Cache results
        self._pattern_cache[context_key] = sorted_recs
        self._cache_timestamp = datetime.now()

        return sorted_recs

    def _recommend_by_co_occurrence(
        self,
        server: str,
        tool: str,
        limit: int
    ) -> list[ToolRecommendation]:
        """Find tools commonly used together with the given tool."""
        try:
            # Find tools used within 5 minutes of the given tool
            query = """
                WITH current_uses AS (
                    SELECT timestamp, server_name, tool_name
                    FROM mcp_tool_history
                    WHERE server_name = ? AND tool_name = ?
                    AND timestamp > CURRENT_TIMESTAMP - INTERVAL ? HOUR
                ),
                nearby_tools AS (
                    SELECT 
                        h.server_name,
                        h.tool_name,
                        COUNT(*) as co_count,
                        AVG(CASE WHEN h.success THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(ABS(EPOCH(h.timestamp - c.timestamp))) as avg_time_diff
                    FROM mcp_tool_history h
                    JOIN current_uses c 
                        ON ABS(EPOCH(h.timestamp - c.timestamp)) < 300
                    WHERE NOT (h.server_name = ? AND h.tool_name = ?)
                    GROUP BY h.server_name, h.tool_name
                    HAVING COUNT(*) >= ?
                )
                SELECT server_name, tool_name, co_count, success_rate, avg_time_diff
                FROM nearby_tools
                ORDER BY co_count DESC
                LIMIT ?
            """

            results = self.conn.execute(
                query,
                [
                    server, tool, self.history_window_hours,
                    server, tool, self.min_occurrences, limit
                ]
            ).fetchall()

            recommendations = []
            max_count = results[0][2] if results else 1

            for row in results:
                srv, tl, count, success_rate, avg_time = row
                score = (count / max_count) * 0.6 + (success_rate or 0) * 0.4

                recommendations.append(ToolRecommendation(
                    tool=tl,
                    server=srv,
                    score=min(score, 1.0),
                    reason=f"Often used together with {tool} ({count} times)",
                    confidence=min(count / 10, 1.0),
                    frequency=count,
                    avg_success_rate=success_rate or 0.0,
                ))

            return recommendations

        except Exception as e:
            logger.warning(f"Failed to get co-occurrence recommendations: {e}")
            return []

    def _recommend_by_sequence(
        self,
        server: str,
        tool: str,
        limit: int
    ) -> list[ToolRecommendation]:
        """Find tools that typically follow the given tool."""
        try:
            # Find tools that commonly come right after the given tool
            query = """
                WITH tool_sequence AS (
                    SELECT 
                        server_name,
                        tool_name,
                        timestamp,
                        LAG(server_name) OVER (ORDER BY timestamp) as prev_server,
                        LAG(tool_name) OVER (ORDER BY timestamp) as prev_tool,
                        EPOCH(timestamp) - EPOCH(LAG(timestamp) OVER (ORDER BY timestamp)) as time_diff
                    FROM mcp_tool_history
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL ? HOUR
                ),
                following_tools AS (
                    SELECT 
                        server_name,
                        tool_name,
                        COUNT(*) as follow_count,
                        AVG(time_diff) as avg_time_after
                    FROM tool_sequence
                    WHERE prev_server = ? AND prev_tool = ?
                    AND time_diff > 0 AND time_diff < 300
                    GROUP BY server_name, tool_name
                    HAVING COUNT(*) >= ?
                )
                SELECT server_name, tool_name, follow_count, avg_time_after
                FROM following_tools
                ORDER BY follow_count DESC
                LIMIT ?
            """

            results = self.conn.execute(
                query,
                [self.history_window_hours, server, tool, self.min_occurrences, limit]
            ).fetchall()

            recommendations = []
            max_count = results[0][2] if results else 1

            for row in results:
                srv, tl, count, avg_time = row
                score = (count / max_count) * 0.8

                recommendations.append(ToolRecommendation(
                    tool=tl,
                    server=srv,
                    score=min(score, 1.0),
                    reason=f"Typically used after {tool} ({count} times)",
                    confidence=min(count / 5, 1.0),
                    frequency=count,
                ))

            return recommendations

        except Exception as e:
            logger.warning(f"Failed to get sequence recommendations: {e}")
            return []

    def _recommend_by_pattern(
        self,
        recent_tools: list[tuple[str, str]],
        limit: int
    ) -> list[ToolRecommendation]:
        """Find tools that match discovered patterns."""
        try:
            # Build pattern signature from recent tools
            if len(recent_tools) < 2:
                return []

            # Look for sequences that start with the same tools
            recent_servers = [t[0] for t in recent_tools[-3:]]
            recent_tool_names = [t[1] for t in recent_tools[-3:]]

            # Find similar historical sequences
            query = """
                WITH numbered_history AS (
                    SELECT 
                        server_name,
                        tool_name,
                        timestamp,
                        ROW_NUMBER() OVER (ORDER BY timestamp) as seq_num
                    FROM mcp_tool_history
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL ? HOUR
                ),
                sequences AS (
                    SELECT 
                        h1.server_name as s1, h1.tool_name as t1,
                        h2.server_name as s2, h2.tool_name as t2,
                        h3.server_name as s3, h3.tool_name as t3
                    FROM numbered_history h1
                    JOIN numbered_history h2 ON h2.seq_num = h1.seq_num + 1
                    JOIN numbered_history h3 ON h3.seq_num = h2.seq_num + 1
                    WHERE h1.server_name = ? AND h1.tool_name = ?
                    AND h2.server_name = ? AND h2.tool_name = ?
                )
                SELECT s3, t3, COUNT(*) as pattern_count
                FROM sequences
                GROUP BY s3, t3
                HAVING COUNT(*) >= ?
                ORDER BY COUNT(*) DESC
                LIMIT ?
            """

            if len(recent_tools) >= 2:
                results = self.conn.execute(
                    query,
                    [
                        self.history_window_hours,
                        recent_servers[-2], recent_tool_names[-2],
                        recent_servers[-1], recent_tool_names[-1],
                        self.min_occurrences, limit
                    ]
                ).fetchall()

                recommendations = []
                max_count = results[0][2] if results else 1

                for row in results:
                    srv, tl, count = row
                    score = (count / max_count) * 0.9

                    recommendations.append(ToolRecommendation(
                        tool=tl,
                        server=srv,
                        score=min(score, 1.0),
                        reason=f"Matches common workflow pattern ({count} occurrences)",
                        confidence=min(count / 3, 1.0),
                        frequency=count,
                    ))

                return recommendations

            return []

        except Exception as e:
            logger.warning(f"Failed to get pattern recommendations: {e}")
            return []

    def _recommend_popular_tools(
        self,
        limit: int,
        exclude: Optional[list[tuple[str, str]]] = None
    ) -> list[ToolRecommendation]:
        """Recommend popular tools as fallback."""
        try:
            exclude = exclude or []
            exclude_set = {f"{s}/{t}" for s, t in exclude}

            query = """
                SELECT 
                    server_name,
                    tool_name,
                    COUNT(*) as usage_count,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate
                FROM mcp_tool_history
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL ? HOUR
                GROUP BY server_name, tool_name
                ORDER BY COUNT(*) DESC
                LIMIT ?
            """

            results = self.conn.execute(
                query, [self.history_window_hours, limit + len(exclude)]
            ).fetchall()

            recommendations = []
            max_count = results[0][2] if results else 1

            for row in results:
                srv, tl, count, success_rate = row
                if f"{srv}/{tl}" in exclude_set:
                    continue

                score = (count / max_count) * 0.3  # Lower score for popularity-only

                recommendations.append(ToolRecommendation(
                    tool=tl,
                    server=srv,
                    score=min(score, 1.0),
                    reason=f"Popular tool ({count} recent uses)",
                    confidence=0.3,  # Lower confidence for popularity
                    frequency=count,
                    avg_success_rate=success_rate or 0.0,
                ))

                if len(recommendations) >= limit:
                    break

            return recommendations

        except Exception as e:
            logger.warning(f"Failed to get popular tools: {e}")
            return []

    def _merge_recommendations(
        self,
        recommendations: list[ToolRecommendation]
    ) -> list[ToolRecommendation]:
        """Merge duplicate recommendations and combine scores."""
        merged: dict[str, ToolRecommendation] = {}

        for rec in recommendations:
            key = f"{rec.server}/{rec.tool}"

            if key in merged:
                existing = merged[key]
                # Combine scores (weighted average favoring higher)
                combined_score = (
                    max(existing.score, rec.score) * 0.7 +
                    min(existing.score, rec.score) * 0.3
                )
                # Combine reasons
                if rec.reason not in existing.reason:
                    combined_reason = f"{existing.reason}; {rec.reason}"
                else:
                    combined_reason = existing.reason

                merged[key] = ToolRecommendation(
                    tool=rec.tool,
                    server=rec.server,
                    score=min(combined_score * 1.1, 1.0),  # Boost for multiple signals
                    reason=combined_reason,
                    confidence=max(existing.confidence, rec.confidence),
                    frequency=existing.frequency + rec.frequency,
                    avg_success_rate=max(existing.avg_success_rate, rec.avg_success_rate),
                    related_tools=list(set(existing.related_tools + rec.related_tools)),
                )
            else:
                merged[key] = rec

        return list(merged.values())

    def analyze_patterns(
        self,
        min_pattern_length: int = 2,
        max_pattern_length: int = 5,
        limit: int = 20
    ) -> list[UsagePattern]:
        """
        Analyze tool history to discover common usage patterns.

        Args:
            min_pattern_length: Minimum tools in a pattern
            max_pattern_length: Maximum tools in a pattern
            limit: Maximum patterns to return

        Returns:
            List of discovered UsagePattern objects
        """
        patterns: list[UsagePattern] = []

        logger.info("Analyzing tool usage patterns...")

        try:
            # Get tool sequences
            query = """
                WITH tool_sessions AS (
                    SELECT 
                        server_name,
                        tool_name,
                        success,
                        duration_ms,
                        timestamp,
                        SUM(CASE WHEN EPOCH(timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) > 300 
                            THEN 1 ELSE 0 END) OVER (ORDER BY timestamp) as session_id
                    FROM mcp_tool_history
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL ? HOUR
                ),
                session_tools AS (
                    SELECT 
                        session_id,
                        LIST(server_name || '/' || tool_name ORDER BY timestamp) as tool_sequence,
                        SUM(duration_ms) as total_duration,
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM tool_sessions
                    GROUP BY session_id
                    HAVING COUNT(*) >= ?
                )
                SELECT tool_sequence, total_duration, success_rate, COUNT(*) OVER () as session_count
                FROM session_tools
                ORDER BY LENGTH(tool_sequence::VARCHAR) DESC
                LIMIT 1000
            """

            results = self.conn.execute(
                query, [self.history_window_hours, min_pattern_length]
            ).fetchall()

            # Count pattern occurrences
            pattern_counts: dict[tuple, list[tuple[float, float]]] = defaultdict(list)

            for row in results:
                tools_list, duration, success_rate, _ = row
                if isinstance(tools_list, str):
                    tools = json.loads(tools_list)
                else:
                    tools = list(tools_list) if tools_list else []

                # Generate sub-patterns
                for length in range(min_pattern_length, min(len(tools) + 1, max_pattern_length + 1)):
                    for i in range(len(tools) - length + 1):
                        sub_pattern = tuple(tools[i:i + length])
                        pattern_counts[sub_pattern].append((duration or 0, success_rate or 0))

            # Convert to UsagePattern objects
            for pattern_tools, occurrences in pattern_counts.items():
                if len(occurrences) < self.min_occurrences:
                    continue

                avg_duration = sum(d for d, _ in occurrences) / len(occurrences)
                avg_success = sum(s for _, s in occurrences) / len(occurrences)

                pattern = UsagePattern(
                    pattern_id=f"pattern_{hash(pattern_tools) % 10000:04d}",
                    description=self._describe_pattern(list(pattern_tools)),
                    tools=list(pattern_tools),
                    frequency=len(occurrences),
                    avg_duration_ms=avg_duration,
                    success_rate=avg_success,
                )
                patterns.append(pattern)

            # Sort by frequency
            patterns.sort(key=lambda p: p.frequency, reverse=True)

            logger.info(f"Discovered {len(patterns)} usage patterns")
            return patterns[:limit]

        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
            return []

    def _describe_pattern(self, tools: list[str]) -> str:
        """Generate a human-readable description of a pattern."""
        if len(tools) == 2:
            return f"Sequence: {tools[0]} followed by {tools[1]}"
        elif len(tools) == 3:
            return f"Workflow: {tools[0]} -> {tools[1]} -> {tools[2]}"
        else:
            return f"{len(tools)}-step workflow starting with {tools[0]}"

    def get_tool_pairs(
        self,
        min_co_occurrence: int = 3,
        limit: int = 50
    ) -> list[ToolPair]:
        """
        Get commonly co-occurring tool pairs.

        Args:
            min_co_occurrence: Minimum times tools must appear together
            limit: Maximum pairs to return

        Returns:
            List of ToolPair objects
        """
        try:
            query = """
                WITH tool_pairs AS (
                    SELECT 
                        h1.server_name || '/' || h1.tool_name as tool_a,
                        h2.server_name || '/' || h2.tool_name as tool_b,
                        COUNT(*) as pair_count,
                        AVG(ABS(EPOCH(h2.timestamp - h1.timestamp))) as avg_time_between,
                        AVG(CASE WHEN h1.success AND h2.success THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(CASE WHEN h1.timestamp < h2.timestamp THEN 0.0 ELSE 1.0 END) as direction
                    FROM mcp_tool_history h1
                    JOIN mcp_tool_history h2 
                        ON ABS(EPOCH(h2.timestamp - h1.timestamp)) < 300
                        AND h2.id > h1.id
                    WHERE h1.timestamp > CURRENT_TIMESTAMP - INTERVAL ? HOUR
                    GROUP BY tool_a, tool_b
                    HAVING COUNT(*) >= ?
                )
                SELECT tool_a, tool_b, pair_count, avg_time_between, success_rate, direction
                FROM tool_pairs
                ORDER BY pair_count DESC
                LIMIT ?
            """

            results = self.conn.execute(
                query, [self.history_window_hours, min_co_occurrence, limit]
            ).fetchall()

            pairs = []
            for row in results:
                tool_a, tool_b, count, avg_time, success_rate, direction = row
                pairs.append(ToolPair(
                    tool_a=tool_a,
                    tool_b=tool_b,
                    co_occurrence_count=count,
                    avg_time_between_ms=avg_time * 1000 if avg_time else 0,
                    success_rate=success_rate or 0,
                    direction_weight=direction or 0.5,
                ))

            logger.info(f"Found {len(pairs)} common tool pairs")
            return pairs

        except Exception as e:
            logger.error(f"Failed to get tool pairs: {e}")
            return []

    def suggest_workflow(
        self,
        goal: str,
        available_servers: Optional[list[str]] = None,
        max_steps: int = 5
    ) -> list[ToolRecommendation]:
        """
        Suggest a workflow to achieve a goal based on patterns.

        Args:
            goal: Description of what user wants to achieve
            available_servers: Limit to specific servers
            max_steps: Maximum workflow steps

        Returns:
            Ordered list of recommended tools
        """
        # This is a simplified implementation
        # A full implementation would use NLP to match goals to patterns
        logger.info(f"Suggesting workflow for goal: {goal}")

        # Get popular tools as starting point
        recommendations = self._recommend_popular_tools(limit=max_steps * 2)

        # Filter by available servers
        if available_servers:
            recommendations = [
                r for r in recommendations
                if r.server in available_servers
            ]

        # Build a simple workflow by finding sequential patterns
        workflow: list[ToolRecommendation] = []

        if recommendations:
            # Start with most popular tool
            workflow.append(recommendations[0])

            # Add follow-up tools
            for _ in range(max_steps - 1):
                if not workflow:
                    break

                last_tool = workflow[-1]
                next_tools = self._recommend_by_sequence(
                    last_tool.server, last_tool.tool, limit=3
                )

                # Filter out already added tools
                existing = {(r.server, r.tool) for r in workflow}
                next_tools = [
                    t for t in next_tools
                    if (t.server, t.tool) not in existing
                ]

                if next_tools:
                    workflow.append(next_tools[0])
                else:
                    break

        return workflow

    def _build_context_key(
        self,
        current_tool: Optional[str],
        current_server: Optional[str],
        recent_tools: Optional[list[tuple[str, str]]]
    ) -> str:
        """Build a cache key from context."""
        parts = []
        if current_server and current_tool:
            parts.append(f"{current_server}/{current_tool}")
        if recent_tools:
            recent_str = ",".join(f"{s}/{t}" for s, t in recent_tools[-3:])
            parts.append(recent_str)
        return "|".join(parts) or "default"

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_timestamp:
            return False
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self._cache_ttl_seconds

    def clear_cache(self) -> None:
        """Clear the recommendation cache."""
        self._pattern_cache.clear()
        self._cache_timestamp = None
        logger.debug("Cleared recommendation cache")

    def get_tool_stats(
        self,
        server: str,
        tool: str
    ) -> dict[str, Any]:
        """
        Get statistics for a specific tool.

        Args:
            server: Server name
            tool: Tool name

        Returns:
            Dictionary with tool statistics
        """
        try:
            query = """
                SELECT 
                    COUNT(*) as total_uses,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_uses,
                    AVG(duration_ms) as avg_duration_ms,
                    MIN(timestamp) as first_used,
                    MAX(timestamp) as last_used
                FROM mcp_tool_history
                WHERE server_name = ? AND tool_name = ?
                AND timestamp > CURRENT_TIMESTAMP - INTERVAL ? HOUR
            """

            result = self.conn.execute(
                query, [server, tool, self.history_window_hours]
            ).fetchone()

            if result:
                total, successful, avg_duration, first_used, last_used = result
                return {
                    "server": server,
                    "tool": tool,
                    "total_uses": total or 0,
                    "successful_uses": successful or 0,
                    "success_rate": (successful / total) if total else 0,
                    "avg_duration_ms": avg_duration or 0,
                    "first_used": first_used.isoformat() if first_used else None,
                    "last_used": last_used.isoformat() if last_used else None,
                }

            return {
                "server": server,
                "tool": tool,
                "total_uses": 0,
                "successful_uses": 0,
                "success_rate": 0,
                "avg_duration_ms": 0,
                "first_used": None,
                "last_used": None,
            }

        except Exception as e:
            logger.error(f"Failed to get tool stats: {e}")
            return {"server": server, "tool": tool, "error": str(e)}
