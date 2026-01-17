"""Intent matching for MCP tool discovery and recommendation.

This module matches parsed intents to available MCP tools and provides
intelligent suggestions with auto-correction capabilities.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from .query_parser import Entity, Intent, QueryParser, QueryResult


class IntentMatch(BaseModel):
    """A matched tool for a given intent."""

    tool_name: str = Field(description="Name of the matched tool")
    server: str = Field(description="Server providing the tool")
    score: float = Field(ge=0.0, le=1.0, description="Match confidence score")
    reason: str = Field(description="Explanation for why this tool was matched")
    required_params: list[str] = Field(
        default_factory=list,
        description="Required parameters for the tool"
    )
    optional_params: list[str] = Field(
        default_factory=list,
        description="Optional parameters for the tool"
    )
    example_usage: Optional[str] = Field(
        default=None,
        description="Example of how to use this tool"
    )

    class Config:
        frozen = True


class ToolInfo(BaseModel):
    """Information about an available MCP tool."""

    name: str = Field(description="Tool name")
    server: str = Field(description="Server name")
    description: str = Field(default="", description="Tool description")
    input_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="JSON schema for input parameters"
    )
    required_params: list[str] = Field(
        default_factory=list,
        description="Required parameter names"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tool tags/categories"
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Usage examples"
    )

    class Config:
        frozen = False


class SuggestionResult(BaseModel):
    """Result of tool suggestions for a query."""

    query: str = Field(description="Original query")
    parsed_intent: Intent = Field(description="Detected intent")
    matches: list[IntentMatch] = Field(
        default_factory=list,
        description="Matched tools"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Additional suggestions or tips"
    )
    corrected_query: Optional[str] = Field(
        default=None,
        description="Auto-corrected query if applicable"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Time taken to process"
    )

    class Config:
        frozen = True


# Intent to tool keyword mappings
INTENT_TOOL_KEYWORDS: dict[Intent, list[str]] = {
    Intent.READ: [
        "read", "get", "fetch", "retrieve", "load", "list", "show", "view",
        "cat", "display", "open", "access", "download", "pull", "query",
    ],
    Intent.WRITE: [
        "write", "create", "save", "store", "insert", "add", "put", "upload",
        "push", "post", "append", "generate", "make", "new", "set",
    ],
    Intent.TRANSFORM: [
        "transform", "modify", "change", "update", "edit", "alter", "replace",
        "process", "manipulate", "reformat", "mutate", "patch",
    ],
    Intent.SEARCH: [
        "search", "find", "query", "lookup", "locate", "discover", "grep",
        "match", "filter", "seek", "scan",
    ],
    Intent.ANALYZE: [
        "analyze", "inspect", "examine", "check", "review", "audit",
        "validate", "verify", "diagnose", "debug", "profile", "measure",
        "count", "stats", "statistics", "summarize", "aggregate",
    ],
    Intent.EXECUTE: [
        "run", "execute", "call", "invoke", "trigger", "start", "launch",
        "perform", "apply", "use", "action", "command",
    ],
    Intent.FILTER: [
        "filter", "select", "where", "subset", "extract", "pick", "choose",
        "exclude", "include", "limit", "slice", "range",
    ],
    Intent.CONVERT: [
        "convert", "transform", "export", "import", "format", "encode",
        "decode", "parse", "serialize", "deserialize", "translate",
    ],
}

# Common tool name patterns for each intent
INTENT_TOOL_PATTERNS: dict[Intent, list[str]] = {
    Intent.READ: [
        r"get_\w+", r"read_\w+", r"fetch_\w+", r"list_\w+", r"show_\w+",
        r"\w+_reader", r"\w+_loader",
    ],
    Intent.WRITE: [
        r"write_\w+", r"create_\w+", r"save_\w+", r"insert_\w+", r"add_\w+",
        r"put_\w+", r"\w+_writer", r"\w+_creator",
    ],
    Intent.TRANSFORM: [
        r"transform_\w+", r"modify_\w+", r"update_\w+", r"edit_\w+",
        r"\w+_transformer", r"\w+_modifier",
    ],
    Intent.SEARCH: [
        r"search_\w+", r"find_\w+", r"query_\w+", r"lookup_\w+",
        r"\w+_searcher", r"\w+_finder",
    ],
    Intent.ANALYZE: [
        r"analyze_\w+", r"check_\w+", r"validate_\w+", r"inspect_\w+",
        r"\w+_analyzer", r"\w+_validator",
    ],
    Intent.EXECUTE: [
        r"run_\w+", r"execute_\w+", r"call_\w+", r"invoke_\w+",
        r"\w+_executor", r"\w+_runner",
    ],
    Intent.FILTER: [
        r"filter_\w+", r"select_\w+", r"extract_\w+",
        r"\w+_filter", r"\w+_selector",
    ],
    Intent.CONVERT: [
        r"convert_\w+", r"export_\w+", r"import_\w+", r"format_\w+",
        r"\w+_converter", r"\w+_exporter", r"\w+_importer",
        r"\w+_to_\w+",
    ],
}

# File type to tool keyword mappings
FILE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "json": ["json", "object", "dict", "dictionary"],
    "csv": ["csv", "comma", "tabular", "spreadsheet"],
    "xml": ["xml", "markup"],
    "yaml": ["yaml", "yml", "config", "configuration"],
    "sql": ["sql", "query", "database", "db", "table"],
    "markdown": ["markdown", "md", "document", "docs"],
    "text": ["text", "txt", "plain", "string"],
    "html": ["html", "webpage", "web"],
    "python": ["python", "py", "script"],
    "javascript": ["javascript", "js", "node"],
    "typescript": ["typescript", "ts"],
    "parquet": ["parquet", "columnar", "arrow"],
    "excel": ["excel", "xlsx", "xls", "spreadsheet"],
}

# Common typos and corrections
COMMON_TYPOS: dict[str, str] = {
    "jsno": "json",
    "jons": "json",
    "csvv": "csv",
    "ymal": "yaml",
    "yamal": "yaml",
    "markdwon": "markdown",
    "markdonw": "markdown",
    "pytohn": "python",
    "pyhton": "python",
    "javascirpt": "javascript",
    "typescrpit": "typescript",
    "exel": "excel",
    "excle": "excel",
    "qurey": "query",
    "qeury": "query",
    "serach": "search",
    "searhc": "search",
    "retrive": "retrieve",
    "retreive": "retrieve",
    "cretae": "create",
    "craete": "create",
    "wirte": "write",
    "wrtie": "write",
    "delte": "delete",
    "deleet": "delete",
    "upadte": "update",
    "udpate": "update",
    "anaylze": "analyze",
    "anaylse": "analyse",
    "covnert": "convert",
    "convet": "convert",
    "exectue": "execute",
    "excute": "execute",
    "fidn": "find",
    "fnid": "find",
    "fiel": "file",
    "flie": "file",
}


class IntentMatcher:
    """Match parsed intents to available MCP tools.
    
    This matcher uses a combination of keyword matching, pattern matching,
    and fuzzy string similarity to find the best tools for a given query.
    
    Example:
        >>> matcher = IntentMatcher()
        >>> matcher.register_tool(ToolInfo(
        ...     name="read_json",
        ...     server="file_server",
        ...     description="Read JSON files"
        ... ))
        >>> result = matcher.match_intent(query_result)
    """

    def __init__(
        self,
        *,
        query_parser: Optional[QueryParser] = None,
        min_score_threshold: float = 0.3,
        max_suggestions: int = 5,
    ) -> None:
        """Initialize the intent matcher.
        
        Args:
            query_parser: Optional QueryParser instance (created if not provided)
            min_score_threshold: Minimum score for a match to be included
            max_suggestions: Maximum number of tool suggestions to return
        """
        self._parser = query_parser or QueryParser()
        self._min_score = min_score_threshold
        self._max_suggestions = max_suggestions
        
        # Tool registry - maps server -> tool_name -> ToolInfo
        self._tools: dict[str, dict[str, ToolInfo]] = {}
        
        # Compiled intent patterns
        self._compiled_patterns: dict[Intent, list[re.Pattern[str]]] = {}
        for intent, patterns in INTENT_TOOL_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        # Pre-computed keyword sets for fast lookup
        self._intent_keywords: dict[Intent, set[str]] = {
            intent: set(keywords)
            for intent, keywords in INTENT_TOOL_KEYWORDS.items()
        }
        
        self._file_type_keywords: dict[str, set[str]] = {
            file_type: set(keywords)
            for file_type, keywords in FILE_TYPE_KEYWORDS.items()
        }
        
        logger.debug("IntentMatcher initialized")

    def register_tool(self, tool: ToolInfo) -> None:
        """Register a tool for matching.
        
        Args:
            tool: Tool information to register
        """
        if tool.server not in self._tools:
            self._tools[tool.server] = {}
        
        self._tools[tool.server][tool.name] = tool
        logger.debug("Registered tool: {}.{}", tool.server, tool.name)

    def register_tools(self, tools: list[ToolInfo]) -> None:
        """Register multiple tools.
        
        Args:
            tools: List of tools to register
        """
        for tool in tools:
            self.register_tool(tool)
        logger.info("Registered {} tools", len(tools))

    def unregister_tool(self, server: str, tool_name: str) -> bool:
        """Unregister a tool.
        
        Args:
            server: Server name
            tool_name: Tool name
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if server in self._tools and tool_name in self._tools[server]:
            del self._tools[server][tool_name]
            if not self._tools[server]:
                del self._tools[server]
            logger.debug("Unregistered tool: {}.{}", server, tool_name)
            return True
        return False

    def clear_tools(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        logger.debug("Cleared all tools")

    def get_all_tools(self) -> list[ToolInfo]:
        """Get all registered tools.
        
        Returns:
            List of all registered tools
        """
        tools = []
        for server_tools in self._tools.values():
            tools.extend(server_tools.values())
        return tools

    def match_intent(
        self,
        query_result: QueryResult,
        *,
        server_filter: Optional[str] = None,
    ) -> list[IntentMatch]:
        """Match a parsed query to available tools.
        
        Args:
            query_result: Parsed query result from QueryParser
            server_filter: Optional server name to filter tools
            
        Returns:
            List of IntentMatch objects sorted by score
        """
        start_time = time.perf_counter()
        
        matches: list[IntentMatch] = []
        
        # Get relevant tools
        tools = self._get_filtered_tools(server_filter)
        
        if not tools:
            logger.warning("No tools registered for matching")
            return []
        
        # Score each tool
        for tool in tools:
            score, reason = self._score_tool(tool, query_result)
            
            if score >= self._min_score:
                matches.append(IntentMatch(
                    tool_name=tool.name,
                    server=tool.server,
                    score=round(score, 3),
                    reason=reason,
                    required_params=tool.required_params,
                    optional_params=self._get_optional_params(tool),
                    example_usage=tool.examples[0] if tool.examples else None,
                ))
        
        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        
        # Limit results
        matches = matches[:self._max_suggestions]
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(
            "Matched {} tools in {:.2f}ms for intent: {}",
            len(matches),
            elapsed,
            query_result.intent.value,
        )
        
        return matches

    def suggest_tools(
        self,
        query: str,
        *,
        server_filter: Optional[str] = None,
        auto_correct: bool = True,
    ) -> SuggestionResult:
        """Suggest tools for a natural language query.
        
        This is a convenience method that combines parsing and matching.
        
        Args:
            query: Natural language query
            server_filter: Optional server name filter
            auto_correct: Whether to auto-correct typos
            
        Returns:
            SuggestionResult with matches and suggestions
        """
        start_time = time.perf_counter()
        
        # Auto-correct query if enabled
        corrected_query = None
        if auto_correct:
            corrected = self.auto_correct_query(query)
            if corrected != query:
                corrected_query = corrected
                query = corrected
        
        # Parse the query
        parse_result = self._parser.parse(query)
        
        # Match intent to tools
        matches = self.match_intent(parse_result, server_filter=server_filter)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(parse_result, matches)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return SuggestionResult(
            query=parse_result.original_query,
            parsed_intent=parse_result.intent,
            matches=matches,
            suggestions=suggestions,
            corrected_query=corrected_query,
            processing_time_ms=round(elapsed, 2),
        )

    def auto_correct_query(self, query: str) -> str:
        """Auto-correct common typos in a query.
        
        Args:
            query: Original query string
            
        Returns:
            Corrected query string
        """
        corrected = query
        query_lower = query.lower()
        
        for typo, correction in COMMON_TYPOS.items():
            # Case-insensitive replacement preserving original case
            pattern = re.compile(re.escape(typo), re.IGNORECASE)
            if pattern.search(corrected):
                corrected = pattern.sub(correction, corrected)
                logger.debug("Auto-corrected '{}' to '{}'", typo, correction)
        
        return corrected

    def find_similar_tools(
        self,
        tool_name: str,
        threshold: float = 0.6,
    ) -> list[tuple[ToolInfo, float]]:
        """Find tools with similar names.
        
        Args:
            tool_name: Tool name to find similar matches for
            threshold: Minimum similarity threshold
            
        Returns:
            List of (ToolInfo, similarity_score) tuples
        """
        similar: list[tuple[ToolInfo, float]] = []
        
        for tool in self.get_all_tools():
            similarity = self._string_similarity(tool_name.lower(), tool.name.lower())
            if similarity >= threshold:
                similar.append((tool, similarity))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:self._max_suggestions]

    def _get_filtered_tools(self, server_filter: Optional[str]) -> list[ToolInfo]:
        """Get tools filtered by server.
        
        Args:
            server_filter: Optional server name filter
            
        Returns:
            List of matching tools
        """
        if server_filter:
            return list(self._tools.get(server_filter, {}).values())
        
        tools = []
        for server_tools in self._tools.values():
            tools.extend(server_tools.values())
        return tools

    def _score_tool(
        self,
        tool: ToolInfo,
        query_result: QueryResult,
    ) -> tuple[float, str]:
        """Score a tool against a parsed query.
        
        Args:
            tool: Tool to score
            query_result: Parsed query result
            
        Returns:
            Tuple of (score, reason)
        """
        scores: list[tuple[float, str]] = []
        
        # 1. Intent pattern matching (0-0.3)
        intent_score = self._score_intent_match(tool, query_result.intent)
        if intent_score > 0:
            scores.append((intent_score * 0.3, "intent pattern match"))
        
        # 2. Keyword matching in description (0-0.25)
        keyword_score = self._score_keyword_match(tool, query_result.keywords)
        if keyword_score > 0:
            scores.append((keyword_score * 0.25, "keyword match"))
        
        # 3. Entity matching (0-0.25)
        entity_score = self._score_entity_match(tool, query_result.entities)
        if entity_score > 0:
            scores.append((entity_score * 0.25, "entity match"))
        
        # 4. Direct name similarity (0-0.2)
        name_score = self._score_name_similarity(tool, query_result.normalized_query)
        if name_score > 0:
            scores.append((name_score * 0.2, "name similarity"))
        
        if not scores:
            return 0.0, "no match"
        
        total_score = sum(s[0] for s in scores)
        reasons = [s[1] for s in scores if s[0] > 0]
        reason = ", ".join(reasons)
        
        return min(total_score, 1.0), reason

    def _score_intent_match(self, tool: ToolInfo, intent: Intent) -> float:
        """Score tool by intent pattern matching.
        
        Args:
            tool: Tool to score
            intent: Detected intent
            
        Returns:
            Score from 0.0 to 1.0
        """
        tool_name_lower = tool.name.lower()
        
        # Check compiled patterns
        patterns = self._compiled_patterns.get(intent, [])
        for pattern in patterns:
            if pattern.match(tool_name_lower):
                return 1.0
        
        # Check keywords
        keywords = self._intent_keywords.get(intent, set())
        for keyword in keywords:
            if keyword in tool_name_lower:
                return 0.8
        
        # Check description
        desc_lower = tool.description.lower()
        for keyword in keywords:
            if keyword in desc_lower:
                return 0.5
        
        return 0.0

    def _score_keyword_match(self, tool: ToolInfo, keywords: list[str]) -> float:
        """Score tool by keyword matching.
        
        Args:
            tool: Tool to score
            keywords: Query keywords
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not keywords:
            return 0.0
        
        tool_text = f"{tool.name} {tool.description}".lower()
        matches = sum(1 for kw in keywords if kw in tool_text)
        
        return min(matches / len(keywords), 1.0)

    def _score_entity_match(self, tool: ToolInfo, entities: list[Entity]) -> float:
        """Score tool by entity matching.
        
        Args:
            tool: Tool to score
            entities: Extracted entities
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not entities:
            return 0.0
        
        tool_text = f"{tool.name} {tool.description} {' '.join(tool.tags)}".lower()
        matches = 0
        
        for entity in entities:
            if entity.type == "file_type":
                # Check file type keywords
                file_keywords = self._file_type_keywords.get(entity.value, set())
                if any(kw in tool_text for kw in file_keywords):
                    matches += 1
                elif entity.value in tool_text:
                    matches += 1
            elif entity.type in ("action", "target"):
                if entity.value in tool_text:
                    matches += 1
            elif entity.type in ("source_format", "target_format"):
                if entity.value in tool_text:
                    matches += 0.5
        
        return min(matches / len(entities), 1.0)

    def _score_name_similarity(self, tool: ToolInfo, query: str) -> float:
        """Score tool by name similarity to query.
        
        Args:
            tool: Tool to score
            query: Normalized query
            
        Returns:
            Score from 0.0 to 1.0
        """
        # Check if tool name appears in query
        if tool.name.lower() in query:
            return 1.0
        
        # Check partial matches
        tool_words = tool.name.lower().replace("_", " ").split()
        query_words = query.split()
        
        matches = sum(1 for tw in tool_words if any(qw in tw or tw in qw for qw in query_words))
        
        if tool_words:
            return min(matches / len(tool_words), 1.0) * 0.8
        
        return 0.0

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using SequenceMatcher.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score from 0.0 to 1.0
        """
        return SequenceMatcher(None, s1, s2).ratio()

    def _get_optional_params(self, tool: ToolInfo) -> list[str]:
        """Extract optional parameters from tool schema.
        
        Args:
            tool: Tool to extract params from
            
        Returns:
            List of optional parameter names
        """
        if not tool.input_schema:
            return []
        
        properties = tool.input_schema.get("properties", {})
        required = set(tool.required_params)
        
        return [name for name in properties.keys() if name not in required]

    def _generate_suggestions(
        self,
        query_result: QueryResult,
        matches: list[IntentMatch],
    ) -> list[str]:
        """Generate helpful suggestions based on query and matches.
        
        Args:
            query_result: Parsed query result
            matches: List of tool matches
            
        Returns:
            List of suggestion strings
        """
        suggestions: list[str] = []
        
        # Low confidence suggestion
        if query_result.confidence < 0.5:
            suggestions.append(
                "Try being more specific about what operation you want to perform."
            )
        
        # No matches suggestion
        if not matches:
            suggestions.append(
                f"No tools found for '{query_result.intent.value}' operation. "
                "Try rephrasing your query or check available tools."
            )
        
        # Multiple intents suggestion
        if query_result.secondary_intents:
            alt_intents = ", ".join(i.value for i in query_result.secondary_intents[:2])
            suggestions.append(
                f"Your query might also be interpreted as: {alt_intents}"
            )
        
        # Entity-based suggestions
        file_types = [e.value for e in query_result.entities if e.type == "file_type"]
        if file_types:
            suggestions.append(
                f"Detected file format(s): {', '.join(file_types)}"
            )
        
        # Low match score suggestion
        if matches and matches[0].score < 0.5:
            suggestions.append(
                "Match confidence is low. Consider specifying the exact tool name."
            )
        
        return suggestions[:3]  # Limit suggestions

    def get_tools_by_intent(self, intent: Intent) -> list[ToolInfo]:
        """Get all tools that match a specific intent.
        
        Args:
            intent: Intent to filter by
            
        Returns:
            List of matching tools
        """
        matching: list[ToolInfo] = []
        
        for tool in self.get_all_tools():
            score = self._score_intent_match(tool, intent)
            if score > 0.3:
                matching.append(tool)
        
        return matching

    def get_tools_by_file_type(self, file_type: str) -> list[ToolInfo]:
        """Get all tools that work with a specific file type.
        
        Args:
            file_type: File type to filter by (e.g., "json", "csv")
            
        Returns:
            List of matching tools
        """
        file_type_lower = file_type.lower()
        keywords = self._file_type_keywords.get(file_type_lower, {file_type_lower})
        
        matching: list[ToolInfo] = []
        
        for tool in self.get_all_tools():
            tool_text = f"{tool.name} {tool.description} {' '.join(tool.tags)}".lower()
            if any(kw in tool_text for kw in keywords):
                matching.append(tool)
        
        return matching

    def explain_match(self, match: IntentMatch, query_result: QueryResult) -> str:
        """Generate a human-readable explanation for a match.
        
        Args:
            match: The intent match to explain
            query_result: The original parsed query
            
        Returns:
            Human-readable explanation string
        """
        parts = [
            f"Tool '{match.tool_name}' from server '{match.server}' "
            f"matched with {match.score:.0%} confidence."
        ]
        
        if match.reason:
            parts.append(f"Matched based on: {match.reason}.")
        
        if match.required_params:
            parts.append(f"Required parameters: {', '.join(match.required_params)}.")
        
        if match.example_usage:
            parts.append(f"Example: {match.example_usage}")
        
        return " ".join(parts)
