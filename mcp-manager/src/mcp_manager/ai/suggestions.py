"""Tool and workflow suggestions for MCP Manager."""

from dataclasses import dataclass, field
from typing import Any, Optional

from ..tools.schema import ToolSchema


@dataclass
class Suggestion:
    """A tool suggestion."""
    
    tool_name: str
    server_name: str
    description: str = ""
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool": self.tool_name,
            "server": self.server_name,
            "description": self.description,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class SuggestionContext:
    """Context for generating suggestions."""
    
    current_server: Optional[str] = None
    current_tool: Optional[str] = None
    recent_tools: list[str] = field(default_factory=list)
    user_intent: Optional[str] = None
    session_data: dict[str, Any] = field(default_factory=dict)


class ToolSuggester:
    """Suggest next tools based on usage patterns and context."""

    def __init__(self, db_connection: Any) -> None:
        """Initialize suggester.
        
        Args:
            db_connection: DuckDB connection
        """
        self.conn = db_connection

    def suggest_next(
        self,
        server: str,
        current_tool: str,
        limit: int = 5,
    ) -> list[Suggestion]:
        """Suggest next tools based on current tool.
        
        Args:
            server: Server name
            current_tool: Currently executed tool
            limit: Maximum suggestions
            
        Returns:
            List of suggestions
        """
        # Query tool execution history for common patterns
        sql = """
            SELECT 
                h2.tool_name as next_tool,
                COUNT(*) as frequency
            FROM mcp_tool_history h1
            JOIN mcp_tool_history h2 
                ON h1.server_name = h2.server_name 
                AND h2.timestamp > h1.timestamp
                AND h2.timestamp <= h1.timestamp + INTERVAL '5 minutes'
            WHERE h1.server_name = ?
            AND h1.tool_name = ?
            AND h2.tool_name != h1.tool_name
            GROUP BY h2.tool_name
            ORDER BY frequency DESC
            LIMIT ?
        """
        
        try:
            results = self.conn.execute(sql, [server, current_tool, limit]).fetchall()
            
            if not results:
                # Fallback to suggesting related tools
                return self._suggest_related(server, current_tool, limit)
            
            suggestions = []
            total_freq = sum(r[1] for r in results)
            
            for tool_name, frequency in results:
                confidence = frequency / total_freq if total_freq > 0 else 0.0
                
                # Get tool description
                desc_result = self.conn.execute(
                    "SELECT description FROM mcp_tools WHERE server_name = ? AND tool_name = ?",
                    [server, tool_name]
                ).fetchone()
                
                description = desc_result[0] if desc_result else ""
                
                suggestions.append(Suggestion(
                    tool_name=tool_name,
                    server_name=server,
                    description=description,
                    confidence=confidence,
                    reason=f"Commonly used after {current_tool} ({frequency} times)",
                ))
            
            return suggestions
            
        except Exception:
            return self._suggest_related(server, current_tool, limit)

    def suggest_for_context(
        self,
        context: SuggestionContext,
        limit: int = 5,
    ) -> list[Suggestion]:
        """Suggest tools based on context.
        
        Args:
            context: Suggestion context
            limit: Maximum suggestions
            
        Returns:
            List of suggestions
        """
        suggestions: list[Suggestion] = []
        
        # If we have a current tool, suggest next tools
        if context.current_server and context.current_tool:
            next_suggestions = self.suggest_next(
                context.current_server,
                context.current_tool,
                limit=limit // 2,
            )
            suggestions.extend(next_suggestions)
        
        # If we have user intent, search for matching tools
        if context.user_intent:
            intent_suggestions = self._suggest_by_intent(
                context.user_intent,
                context.current_server,
                limit=limit - len(suggestions),
            )
            suggestions.extend(intent_suggestions)
        
        # Avoid recently used tools
        recent_set = set(context.recent_tools)
        suggestions = [
            s for s in suggestions
            if s.tool_name not in recent_set
        ]
        
        return suggestions[:limit]

    def _suggest_related(
        self,
        server: str,
        tool_name: str,
        limit: int,
    ) -> list[Suggestion]:
        """Suggest related tools based on naming patterns.
        
        Args:
            server: Server name
            tool_name: Tool name
            limit: Maximum suggestions
            
        Returns:
            List of suggestions
        """
        # Extract prefixes/keywords from tool name
        parts = tool_name.lower().split("_")
        
        suggestions = []
        
        for part in parts:
            if len(part) < 3:
                continue
            
            sql = """
                SELECT tool_name, description
                FROM mcp_tools
                WHERE server_name = ?
                AND tool_name ILIKE '%' || ? || '%'
                AND tool_name != ?
                AND enabled = true
                LIMIT ?
            """
            
            try:
                results = self.conn.execute(sql, [server, part, tool_name, limit]).fetchall()
                
                for name, description in results:
                    suggestions.append(Suggestion(
                        tool_name=name,
                        server_name=server,
                        description=description or "",
                        confidence=0.5,
                        reason=f"Related to {tool_name} (shared: {part})",
                    ))
            except Exception:
                continue
        
        return suggestions[:limit]

    def _suggest_by_intent(
        self,
        intent: str,
        server: Optional[str],
        limit: int,
    ) -> list[Suggestion]:
        """Suggest tools matching user intent.
        
        Args:
            intent: User intent description
            server: Optional server filter
            limit: Maximum suggestions
            
        Returns:
            List of suggestions
        """
        # Extract keywords from intent
        keywords = intent.lower().split()
        keywords = [k for k in keywords if len(k) > 2]
        
        if not keywords:
            return []
        
        sql = """
            SELECT server_name, tool_name, description
            FROM mcp_tools
            WHERE enabled = true
        """
        params: list[Any] = []
        
        if server:
            sql += " AND server_name = ?"
            params.append(server)
        
        # Match any keyword
        keyword_conditions = []
        for keyword in keywords[:5]:  # Limit keywords
            keyword_conditions.append(
                "(tool_name ILIKE '%' || ? || '%' OR description ILIKE '%' || ? || '%')"
            )
            params.extend([keyword, keyword])
        
        if keyword_conditions:
            sql += " AND (" + " OR ".join(keyword_conditions) + ")"
        
        sql += f" LIMIT {limit}"
        
        try:
            results = self.conn.execute(sql, params).fetchall()
            
            return [
                Suggestion(
                    tool_name=row[1],
                    server_name=row[0],
                    description=row[2] or "",
                    confidence=0.7,
                    reason=f"Matches intent: {intent[:50]}",
                )
                for row in results
            ]
        except Exception:
            return []


class WorkflowSuggester:
    """Suggest workflows based on patterns and goals."""

    def __init__(self, db_connection: Any) -> None:
        """Initialize suggester.
        
        Args:
            db_connection: DuckDB connection
        """
        self.conn = db_connection

    def suggest_workflow(
        self,
        goal: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Suggest a workflow for a goal.
        
        Args:
            goal: Goal description
            limit: Maximum suggestions
            
        Returns:
            List of workflow suggestions
        """
        # Extract keywords from goal
        keywords = goal.lower().split()
        keywords = [k for k in keywords if len(k) > 2]
        
        if not keywords:
            return []
        
        # Find tools matching the goal
        sql = """
            SELECT server_name, tool_name, description
            FROM mcp_tools
            WHERE enabled = true
            AND (
                tool_name ILIKE '%' || ? || '%'
                OR description ILIKE '%' || ? || '%'
            )
            LIMIT 10
        """
        
        matching_tools = []
        for keyword in keywords[:5]:
            try:
                results = self.conn.execute(sql, [keyword, keyword]).fetchall()
                matching_tools.extend(results)
            except Exception:
                continue
        
        if not matching_tools:
            return []
        
        # Build suggested workflow
        seen = set()
        steps = []
        for server, tool, desc in matching_tools:
            if tool not in seen:
                seen.add(tool)
                steps.append({
                    "server": server,
                    "tool": tool,
                    "description": desc or "",
                })
                if len(steps) >= 5:
                    break
        
        return [{
            "name": f"Workflow for: {goal[:50]}",
            "steps": steps,
            "confidence": 0.6,
        }]

    def suggest_from_pattern(
        self,
        pattern_name: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Suggest workflows from common patterns.
        
        Args:
            pattern_name: Pattern name or description
            limit: Maximum suggestions
            
        Returns:
            List of workflow suggestions
        """
        # Common workflow patterns
        patterns: dict[str, list[str]] = {
            "etl": ["extract", "transform", "load"],
            "crud": ["create", "read", "update", "delete"],
            "pipeline": ["read", "process", "write"],
            "api": ["fetch", "parse", "store"],
            "file": ["read", "process", "write"],
            "data": ["query", "transform", "save"],
        }
        
        # Find matching pattern
        pattern_lower = pattern_name.lower()
        matching_keywords = []
        
        for pattern_key, keywords in patterns.items():
            if pattern_key in pattern_lower:
                matching_keywords = keywords
                break
        
        if not matching_keywords:
            # Try to extract from pattern name
            matching_keywords = [w for w in pattern_lower.split() if len(w) > 3][:3]
        
        if not matching_keywords:
            return []
        
        # Find tools for each keyword
        steps = []
        for keyword in matching_keywords:
            sql = """
                SELECT server_name, tool_name, description
                FROM mcp_tools
                WHERE enabled = true
                AND (tool_name ILIKE '%' || ? || '%')
                LIMIT 1
            """
            
            try:
                result = self.conn.execute(sql, [keyword]).fetchone()
                if result:
                    steps.append({
                        "server": result[0],
                        "tool": result[1],
                        "description": result[2] or "",
                    })
            except Exception:
                continue
        
        if not steps:
            return []
        
        return [{
            "name": f"Pattern: {pattern_name}",
            "steps": steps,
            "confidence": 0.7,
        }]

    def detect_patterns(self, limit: int = 5) -> list[dict[str, Any]]:
        """Detect common patterns from execution history.
        
        Args:
            limit: Maximum patterns
            
        Returns:
            List of detected patterns
        """
        # Query for common tool sequences
        sql = """
            WITH tool_sequences AS (
                SELECT 
                    h1.tool_name as tool1,
                    h2.tool_name as tool2,
                    h3.tool_name as tool3,
                    COUNT(*) as frequency
                FROM mcp_tool_history h1
                JOIN mcp_tool_history h2 
                    ON h1.server_name = h2.server_name 
                    AND h2.timestamp > h1.timestamp
                    AND h2.timestamp <= h1.timestamp + INTERVAL '5 minutes'
                JOIN mcp_tool_history h3
                    ON h2.server_name = h3.server_name
                    AND h3.timestamp > h2.timestamp
                    AND h3.timestamp <= h2.timestamp + INTERVAL '5 minutes'
                WHERE h1.tool_name != h2.tool_name
                AND h2.tool_name != h3.tool_name
                AND h1.tool_name != h3.tool_name
                GROUP BY h1.tool_name, h2.tool_name, h3.tool_name
                HAVING COUNT(*) >= 2
                ORDER BY frequency DESC
                LIMIT ?
            )
            SELECT tool1, tool2, tool3, frequency FROM tool_sequences
        """
        
        try:
            results = self.conn.execute(sql, [limit]).fetchall()
            
            patterns = []
            for tool1, tool2, tool3, frequency in results:
                patterns.append({
                    "name": f"Pattern: {tool1} -> {tool2} -> {tool3}",
                    "tools": [tool1, tool2, tool3],
                    "frequency": frequency,
                    "confidence": min(0.9, frequency / 10.0),
                })
            
            return patterns
            
        except Exception:
            return []
