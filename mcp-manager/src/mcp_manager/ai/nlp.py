"""Natural Language Processing for MCP tool discovery and queries."""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from .embeddings import SemanticSearcher, SemanticSearchResult


class Intent(Enum):
    """User intent classification."""
    
    SEARCH = auto()
    DISCOVER = auto()
    EXECUTE = auto()
    CALL = auto()
    HELP = auto()
    EXPLAIN = auto()
    EXAMPLE = auto()
    LIST = auto()
    INSPECT = auto()
    UNKNOWN = auto()


@dataclass
class ParsedQuery:
    """Parsed natural language query."""
    
    raw_query: str
    keywords: list[str] = field(default_factory=list)
    intent: Intent = Intent.SEARCH
    tool_name: Optional[str] = None
    server_name: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    negations: list[str] = field(default_factory=list)


class IntentClassifier:
    """Classify user intent from natural language queries."""

    # Intent patterns
    INTENT_PATTERNS: dict[Intent, list[str]] = {
        Intent.SEARCH: [
            r'\bfind\b', r'\bsearch\b', r'\blook\s+for\b', r'\bquery\b',
            r'\bwhere\b', r'\bwhich\b',
        ],
        Intent.DISCOVER: [
            r'\bdiscover\b', r'\bexplore\b', r'\bwhat\s+tools\b',
            r'\bavailable\b', r'\bshow\s+me\b',
        ],
        Intent.EXECUTE: [
            r'\brun\b', r'\bexecute\b', r'\bperform\b', r'\bdo\b',
            r'\btrigger\b', r'\bstart\b',
        ],
        Intent.CALL: [
            r'\bcall\b', r'\binvoke\b', r'\buse\b',
        ],
        Intent.HELP: [
            r'\bhelp\b', r'\bhow\s+to\b', r'\bhow\s+do\b', r'\bwhat\s+is\b',
            r'\bexplain\b', r'\bdocument\b',
        ],
        Intent.EXPLAIN: [
            r'\bexplain\b', r'\bdescribe\b', r'\btell\s+me\s+about\b',
            r'\bwhat\s+does\b',
        ],
        Intent.EXAMPLE: [
            r'\bexample\b', r'\bsamples?\b', r'\bshow\s+usage\b',
            r'\bdemonstrate\b',
        ],
        Intent.LIST: [
            r'\blist\b', r'\bshow\s+all\b', r'\benumerate\b',
            r'\bget\s+all\b',
        ],
        Intent.INSPECT: [
            r'\binspect\b', r'\bdetails?\b', r'\binfo\b',
            r'\bschema\b', r'\bparameters?\b',
        ],
    }

    def classify(self, query: str) -> Intent:
        """Classify the intent of a query.
        
        Args:
            query: Natural language query
            
        Returns:
            Classified intent
        """
        if not query:
            return Intent.SEARCH

        query_lower = query.lower()

        # Score each intent based on pattern matches
        scores: dict[Intent, int] = {intent: 0 for intent in Intent}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[intent] += 1

        # Find highest scoring intent
        max_score = max(scores.values())
        if max_score > 0:
            for intent, score in scores.items():
                if score == max_score:
                    return intent

        # Default to SEARCH for unknown queries
        return Intent.SEARCH


class QueryParser:
    """Parse natural language queries into structured format."""

    # Common tool name patterns
    TOOL_PATTERNS = [
        r'\b(?:the\s+)?(\w+_\w+)\s+tool\b',
        r'\btool\s+(\w+_\w+)\b',
        r'\busing\s+(\w+_\w+)\b',
        r'\bwith\s+(\w+_\w+)\b',
    ]

    # Server name patterns
    SERVER_PATTERNS = [
        r'\bfrom\s+server\s+(\w+)\b',
        r'\bon\s+server\s+(\w+)\b',
        r'\bserver\s+(\w+)\b',
        r'\bfrom\s+(\w+)\s+server\b',
    ]

    # Parameter patterns
    PARAM_PATTERNS = [
        r'\bwith\s+(\w+)\s*=\s*([^\s,]+)',
        r'\b(\w+)\s*:\s*([^\s,]+)',
    ]

    def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query.
        
        Args:
            query: Natural language query string
            
        Returns:
            ParsedQuery object with extracted information
        """
        if not query:
            return ParsedQuery(raw_query="")

        parsed = ParsedQuery(raw_query=query)
        query_lower = query.lower()

        # Extract keywords
        parsed.keywords = self._extract_keywords(query)

        # Classify intent
        classifier = IntentClassifier()
        parsed.intent = classifier.classify(query)

        # Extract tool name
        for pattern in self.TOOL_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                parsed.tool_name = match.group(1)
                break

        # Extract server name
        for pattern in self.SERVER_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                parsed.server_name = match.group(1)
                break

        # Extract parameters
        for pattern in self.PARAM_PATTERNS:
            for match in re.finditer(pattern, query):
                param_name = match.group(1)
                param_value = match.group(2).strip('"\'')
                parsed.parameters[param_name] = param_value

        # Extract quoted strings as special keywords
        quoted = re.findall(r'"([^"]+)"', query)
        parsed.keywords.extend(quoted)

        # Extract negations
        negation_patterns = [r'\bnot\s+(\w+)', r'\bwithout\s+(\w+)', r'\bexclude\s+(\w+)']
        for pattern in negation_patterns:
            for match in re.finditer(pattern, query_lower):
                parsed.negations.append(match.group(1))

        return parsed

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'our', 'their', 'what', 'which',
            'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if',
        }

        # Tokenize
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())
        
        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords


@dataclass
class NLPResult:
    """Result from natural language processing."""
    
    query: ParsedQuery
    tools: list[SemanticSearchResult] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    explanation: Optional[str] = None


class NaturalLanguageProcessor:
    """Process natural language queries for tool discovery."""

    def __init__(self, db_connection: Any) -> None:
        """Initialize NLP processor.
        
        Args:
            db_connection: DuckDB connection
        """
        self.conn = db_connection
        self.parser = QueryParser()
        self.classifier = IntentClassifier()
        self.searcher = SemanticSearcher(db_connection)

    def process(self, query: str) -> NLPResult:
        """Process a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            NLPResult with parsed query and matching tools
        """
        # Parse the query
        parsed = self.parser.parse(query)

        # Create result
        result = NLPResult(query=parsed)

        # Handle based on intent
        if parsed.intent in [Intent.SEARCH, Intent.DISCOVER, Intent.LIST]:
            # Search for tools
            search_query = " ".join(parsed.keywords) if parsed.keywords else query
            result.tools = self.searcher.search(
                search_query,
                limit=10,
                server=parsed.server_name,
            )

        elif parsed.intent in [Intent.EXECUTE, Intent.CALL]:
            # Find specific tool to execute
            if parsed.tool_name:
                result.tools = self._find_tool(parsed.tool_name, parsed.server_name)
            else:
                # Search based on keywords
                search_query = " ".join(parsed.keywords) if parsed.keywords else query
                result.tools = self.searcher.search(search_query, limit=5)

        elif parsed.intent in [Intent.HELP, Intent.EXPLAIN, Intent.EXAMPLE]:
            # Find tool and provide explanation
            if parsed.tool_name:
                result.tools = self._find_tool(parsed.tool_name, parsed.server_name)
            result.explanation = self._generate_explanation(parsed)

        elif parsed.intent == Intent.INSPECT:
            # Get detailed tool information
            if parsed.tool_name:
                result.tools = self._find_tool(parsed.tool_name, parsed.server_name)

        # Generate suggestions
        result.suggestions = self._generate_suggestions(parsed, result.tools)

        return result

    def _find_tool(
        self,
        tool_name: str,
        server_name: Optional[str] = None,
    ) -> list[SemanticSearchResult]:
        """Find a specific tool by name.
        
        Args:
            tool_name: Tool name to find
            server_name: Optional server filter
            
        Returns:
            List of matching tools
        """
        sql = """
            SELECT server_name, tool_name, description, required_params
            FROM mcp_tools
            WHERE enabled = true
            AND tool_name ILIKE ?
        """
        params = [f"%{tool_name}%"]

        if server_name:
            sql += " AND server_name ILIKE ?"
            params.append(f"%{server_name}%")

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

    def _generate_explanation(self, parsed: ParsedQuery) -> str:
        """Generate explanation based on parsed query.
        
        Args:
            parsed: Parsed query
            
        Returns:
            Explanation string
        """
        parts = []
        
        if parsed.tool_name:
            parts.append(f"Looking for information about '{parsed.tool_name}' tool.")
        
        if parsed.server_name:
            parts.append(f"Filtering to server '{parsed.server_name}'.")
        
        if parsed.keywords:
            parts.append(f"Key terms: {', '.join(parsed.keywords[:5])}")
        
        return " ".join(parts) if parts else "Processing your query."

    def _generate_suggestions(
        self,
        parsed: ParsedQuery,
        tools: list[SemanticSearchResult],
    ) -> list[str]:
        """Generate search suggestions.
        
        Args:
            parsed: Parsed query
            tools: Found tools
            
        Returns:
            List of suggestion strings
        """
        suggestions = []

        if not tools:
            suggestions.append("Try using more specific terms")
            suggestions.append("Use semantic search with --semantic flag")
            if parsed.keywords:
                suggestions.append(f"Search for related concepts: {', '.join(parsed.keywords[:3])}")
        else:
            if len(tools) > 5:
                suggestions.append("Try adding a server filter with --server")
            if parsed.intent == Intent.SEARCH and not parsed.server_name:
                servers = set(t.server for t in tools)
                if len(servers) > 1:
                    suggestions.append(f"Found tools on servers: {', '.join(servers)}")

        return suggestions
