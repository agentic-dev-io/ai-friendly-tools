"""Natural Language Processing module for MCP tool discovery.

This module provides regex and pattern-based NLP capabilities for understanding
user queries and matching them to available MCP tools.

Features:
- Intent detection (READ, WRITE, TRANSFORM, SEARCH, ANALYZE, EXECUTE, FILTER, CONVERT)
- Entity extraction (file types, actions, targets, paths)
- Tool matching based on intent and entities
- Auto-correction of common typos
- Fast parsing (<100ms target)

Example:
    >>> from mcp_manager.nlp import QueryParser, IntentMatcher, Intent
    >>> 
    >>> # Parse a natural language query
    >>> parser = QueryParser()
    >>> result = parser.parse("convert json file to csv format")
    >>> print(result.intent)  # Intent.CONVERT
    >>> 
    >>> # Match intent to tools
    >>> matcher = IntentMatcher()
    >>> matcher.register_tool(ToolInfo(
    ...     name="json_to_csv",
    ...     server="converter",
    ...     description="Convert JSON files to CSV format"
    ... ))
    >>> matches = matcher.match_intent(result)
"""

from .intent_matcher import (
    IntentMatch,
    IntentMatcher,
    SuggestionResult,
    ToolInfo,
)
from .query_parser import (
    Entity,
    Intent,
    QueryParser,
    QueryResult,
)

__all__ = [
    # Query Parser
    "QueryParser",
    "QueryResult",
    "Intent",
    "Entity",
    # Intent Matcher
    "IntentMatcher",
    "IntentMatch",
    "ToolInfo",
    "SuggestionResult",
]
