"""Natural language query parsing for MCP tool discovery.

This module provides regex and pattern-based NLP for understanding user queries
and extracting intent and entities without heavy ML dependencies.
"""

from __future__ import annotations

import re
import time
from dataclasses import field
from enum import Enum
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


class Intent(str, Enum):
    """User intent categories for tool operations."""

    READ = "read"  # Read/get/fetch data
    WRITE = "write"  # Write/create/save data
    TRANSFORM = "transform"  # Convert/transform/modify data
    SEARCH = "search"  # Search/find/query data
    ANALYZE = "analyze"  # Analyze/inspect/examine data
    EXECUTE = "execute"  # Run/execute/call commands
    FILTER = "filter"  # Filter/select/subset data
    CONVERT = "convert"  # Convert between formats


class Entity(BaseModel):
    """Extracted entity from a query."""

    type: str = Field(description="Entity type (file_type, action, target, etc.)")
    value: str = Field(description="Extracted value")
    start: int = Field(description="Start position in original query")
    end: int = Field(description="End position in original query")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")

    class Config:
        frozen = True


class QueryResult(BaseModel):
    """Result of parsing a natural language query."""

    intent: Intent = Field(description="Detected primary intent")
    entities: list[Entity] = Field(default_factory=list, description="Extracted entities")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    original_query: str = Field(description="Original query string")
    normalized_query: str = Field(default="", description="Normalized/cleaned query")
    parse_time_ms: float = Field(default=0.0, description="Time taken to parse in ms")
    secondary_intents: list[Intent] = Field(
        default_factory=list,
        description="Additional detected intents"
    )
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")

    class Config:
        frozen = True


# Intent verb patterns - maps verbs to intents
INTENT_VERB_PATTERNS: dict[Intent, list[str]] = {
    Intent.READ: [
        r"\bread\b", r"\bget\b", r"\bfetch\b", r"\bretrieve\b", r"\bload\b",
        r"\bshow\b", r"\bdisplay\b", r"\bview\b", r"\blist\b", r"\bcat\b",
        r"\bopen\b", r"\baccess\b", r"\bdownload\b", r"\bpull\b",
    ],
    Intent.WRITE: [
        r"\bwrite\b", r"\bcreate\b", r"\bsave\b", r"\bstore\b", r"\binsert\b",
        r"\badd\b", r"\bput\b", r"\bupload\b", r"\bpush\b", r"\bpost\b",
        r"\bappend\b", r"\bgenerate\b", r"\bmake\b", r"\bnew\b",
    ],
    Intent.TRANSFORM: [
        r"\btransform\b", r"\bmodify\b", r"\bchange\b", r"\bupdate\b",
        r"\bedit\b", r"\balter\b", r"\bmutate\b", r"\breplace\b",
        r"\bprocess\b", r"\bmanipulate\b", r"\breformat\b",
    ],
    Intent.SEARCH: [
        r"\bsearch\b", r"\bfind\b", r"\bquery\b", r"\blookup\b", r"\blocate\b",
        r"\bdiscover\b", r"\bgrep\b", r"\bmatch\b", r"\bseek\b",
    ],
    Intent.ANALYZE: [
        r"\banalyze\b", r"\binspect\b", r"\bexamine\b", r"\bcheck\b",
        r"\breview\b", r"\baudit\b", r"\bvalidate\b", r"\bverify\b",
        r"\bdiagnose\b", r"\bdebug\b", r"\bprofile\b", r"\bmeasure\b",
        r"\bcount\b", r"\bstats\b", r"\bstatistics\b", r"\bsummarize\b",
    ],
    Intent.EXECUTE: [
        r"\brun\b", r"\bexecute\b", r"\bcall\b", r"\binvoke\b", r"\btrigger\b",
        r"\bstart\b", r"\blaunch\b", r"\bperform\b", r"\bdo\b",
        r"\bapply\b", r"\buse\b",
    ],
    Intent.FILTER: [
        r"\bfilter\b", r"\bselect\b", r"\bwhere\b", r"\bsubset\b",
        r"\bextract\b", r"\bpick\b", r"\bchoose\b", r"\bonly\b",
        r"\bexclude\b", r"\binclude\b", r"\blimit\b",
    ],
    Intent.CONVERT: [
        r"\bconvert\b", r"\btransform\b", r"\bexport\b", r"\bimport\b",
        r"\bformat\b", r"\bencode\b", r"\bdecode\b", r"\bparse\b",
        r"\bserialize\b", r"\bdeserialize\b", r"\btranslate\b",
        r"\bfrom\s+\w+\s+to\b", r"\bto\s+\w+\b",
    ],
}

# File type patterns
FILE_TYPE_PATTERNS: dict[str, list[str]] = {
    "json": [r"\bjson\b", r"\.json\b"],
    "csv": [r"\bcsv\b", r"\.csv\b", r"\bcomma.?separated\b"],
    "xml": [r"\bxml\b", r"\.xml\b"],
    "yaml": [r"\byaml\b", r"\byml\b", r"\.ya?ml\b"],
    "sql": [r"\bsql\b", r"\.sql\b", r"\bquery\b", r"\bdatabase\b"],
    "markdown": [r"\bmarkdown\b", r"\bmd\b", r"\.md\b"],
    "text": [r"\btext\b", r"\btxt\b", r"\.txt\b", r"\bplain\b"],
    "html": [r"\bhtml\b", r"\.html\b", r"\bwebpage\b"],
    "python": [r"\bpython\b", r"\bpy\b", r"\.py\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b", r"\.js\b"],
    "typescript": [r"\btypescript\b", r"\bts\b", r"\.ts\b"],
    "parquet": [r"\bparquet\b", r"\.parquet\b"],
    "arrow": [r"\barrow\b", r"\.arrow\b"],
    "excel": [r"\bexcel\b", r"\bxlsx?\b", r"\.xlsx?\b"],
    "image": [r"\bimage\b", r"\bpng\b", r"\bjpe?g\b", r"\bgif\b", r"\bsvg\b"],
    "pdf": [r"\bpdf\b", r"\.pdf\b"],
}

# Action patterns for specific operations
ACTION_PATTERNS: dict[str, list[str]] = {
    "aggregate": [r"\baggregate\b", r"\bsum\b", r"\baverage\b", r"\bavg\b", r"\bcount\b", r"\bgroup\b"],
    "sort": [r"\bsort\b", r"\border\b", r"\brank\b", r"\btop\b", r"\bbottom\b"],
    "join": [r"\bjoin\b", r"\bmerge\b", r"\bcombine\b", r"\bunion\b"],
    "split": [r"\bsplit\b", r"\bseparate\b", r"\bdivide\b", r"\bpartition\b"],
    "deduplicate": [r"\bdedup\b", r"\bdeduplicate\b", r"\bunique\b", r"\bdistinct\b"],
    "validate": [r"\bvalidate\b", r"\bcheck\b", r"\bverify\b", r"\btest\b"],
    "compress": [r"\bcompress\b", r"\bzip\b", r"\bgzip\b", r"\barchive\b"],
    "decompress": [r"\bdecompress\b", r"\bunzip\b", r"\bextract\b", r"\bunarchive\b"],
}

# Target patterns (what the action applies to)
TARGET_PATTERNS: dict[str, list[str]] = {
    "file": [r"\bfile\b", r"\bfiles\b", r"\bdocument\b", r"\bdocuments\b"],
    "directory": [r"\bdirectory\b", r"\bfolder\b", r"\bdir\b", r"\bpath\b"],
    "database": [r"\bdatabase\b", r"\bdb\b", r"\btable\b", r"\btables\b"],
    "api": [r"\bapi\b", r"\bendpoint\b", r"\burl\b", r"\bservice\b"],
    "server": [r"\bserver\b", r"\bhost\b", r"\bremote\b"],
    "data": [r"\bdata\b", r"\bdataset\b", r"\brecords\b", r"\brows\b"],
    "column": [r"\bcolumn\b", r"\bfield\b", r"\battribute\b"],
    "schema": [r"\bschema\b", r"\bstructure\b", r"\bformat\b"],
}

# Stopwords to filter out
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "mine",
    "yours", "hers", "ours", "theirs", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "then", "once", "here", "there",
    "and", "but", "or", "yet", "for", "as", "if", "because", "although",
    "with", "without", "by", "from", "in", "into", "on", "onto", "of",
    "to", "at", "about", "after", "before", "between", "through", "during",
    "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "please", "want", "like", "using", "via", "let", "me",
}


class QueryParser:
    """Parse natural language queries to extract intent and entities.
    
    This parser uses regex patterns and heuristics to understand user queries
    without heavy ML dependencies. Designed for fast parsing (<100ms).
    
    Example:
        >>> parser = QueryParser()
        >>> result = parser.parse("convert the json file to csv format")
        >>> print(result.intent)  # Intent.CONVERT
        >>> print(result.entities)  # [Entity(type='file_type', value='json'), ...]
    """

    def __init__(
        self,
        *,
        custom_intent_patterns: Optional[dict[Intent, list[str]]] = None,
        custom_file_types: Optional[dict[str, list[str]]] = None,
        custom_actions: Optional[dict[str, list[str]]] = None,
        custom_targets: Optional[dict[str, list[str]]] = None,
        custom_stopwords: Optional[set[str]] = None,
    ) -> None:
        """Initialize the query parser with optional custom patterns.
        
        Args:
            custom_intent_patterns: Additional intent verb patterns
            custom_file_types: Additional file type patterns
            custom_actions: Additional action patterns
            custom_targets: Additional target patterns
            custom_stopwords: Additional stopwords to filter
        """
        # Merge default patterns with custom ones
        self._intent_patterns = dict(INTENT_VERB_PATTERNS)
        if custom_intent_patterns:
            for intent, patterns in custom_intent_patterns.items():
                existing = self._intent_patterns.get(intent, [])
                self._intent_patterns[intent] = existing + patterns

        self._file_type_patterns = dict(FILE_TYPE_PATTERNS)
        if custom_file_types:
            self._file_type_patterns.update(custom_file_types)

        self._action_patterns = dict(ACTION_PATTERNS)
        if custom_actions:
            self._action_patterns.update(custom_actions)

        self._target_patterns = dict(TARGET_PATTERNS)
        if custom_targets:
            self._target_patterns.update(custom_targets)

        self._stopwords = STOPWORDS.copy()
        if custom_stopwords:
            self._stopwords.update(custom_stopwords)

        # Pre-compile regex patterns for performance
        self._compiled_intent_patterns: dict[Intent, list[re.Pattern[str]]] = {}
        for intent, patterns in self._intent_patterns.items():
            self._compiled_intent_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._compiled_file_types: dict[str, list[re.Pattern[str]]] = {}
        for file_type, patterns in self._file_type_patterns.items():
            self._compiled_file_types[file_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._compiled_actions: dict[str, list[re.Pattern[str]]] = {}
        for action, patterns in self._action_patterns.items():
            self._compiled_actions[action] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._compiled_targets: dict[str, list[re.Pattern[str]]] = {}
        for target, patterns in self._target_patterns.items():
            self._compiled_targets[target] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        # Common conversion pattern
        self._conversion_pattern = re.compile(
            r"(?:from\s+)?(\w+)\s+(?:to|into)\s+(\w+)",
            re.IGNORECASE
        )

        # Path/filename pattern
        self._path_pattern = re.compile(
            r"[\"']?([a-zA-Z]:[/\\]|[/\\]|\.{1,2}[/\\])?[\w\-./\\]+\.\w+[\"']?",
            re.IGNORECASE
        )

        # Quoted string pattern
        self._quoted_pattern = re.compile(r"[\"']([^\"']+)[\"']")

        # Number pattern
        self._number_pattern = re.compile(r"\b(\d+(?:\.\d+)?)\b")

        logger.debug("QueryParser initialized with {} intent patterns", len(self._intent_patterns))

    def parse(self, query: str) -> QueryResult:
        """Parse a natural language query to extract intent and entities.
        
        Args:
            query: Raw natural language query string
            
        Returns:
            QueryResult containing detected intent, entities, and metadata
            
        Example:
            >>> parser = QueryParser()
            >>> result = parser.parse("find all json files in the data folder")
            >>> result.intent
            Intent.SEARCH
        """
        start_time = time.perf_counter()
        
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return QueryResult(
                intent=Intent.SEARCH,  # Default intent
                entities=[],
                confidence=0.0,
                original_query=query or "",
                normalized_query="",
                parse_time_ms=0.0,
            )

        # Normalize the query
        normalized = self._normalize_query(query)
        logger.debug("Normalized query: '{}'", normalized)

        # Extract intent
        intent, intent_confidence, secondary_intents = self.extract_intent(normalized)
        logger.debug("Detected intent: {} (confidence: {:.2f})", intent, intent_confidence)

        # Extract entities
        entities = self.extract_entities(query, normalized)
        logger.debug("Extracted {} entities", len(entities))

        # Extract keywords
        keywords = self._extract_keywords(normalized)

        # Calculate overall confidence
        entity_boost = min(len(entities) * 0.05, 0.2)  # Max 0.2 boost from entities
        overall_confidence = min(intent_confidence + entity_boost, 1.0)

        parse_time = (time.perf_counter() - start_time) * 1000

        result = QueryResult(
            intent=intent,
            entities=entities,
            confidence=overall_confidence,
            original_query=query,
            normalized_query=normalized,
            parse_time_ms=round(parse_time, 2),
            secondary_intents=secondary_intents,
            keywords=keywords,
        )

        logger.info(
            "Parsed query in {:.2f}ms: intent={}, entities={}, confidence={:.2f}",
            parse_time,
            intent.value,
            len(entities),
            overall_confidence,
        )

        return result

    def extract_intent(self, query: str) -> tuple[Intent, float, list[Intent]]:
        """Extract the primary intent from a query.
        
        Args:
            query: Normalized query string
            
        Returns:
            Tuple of (primary_intent, confidence, secondary_intents)
        """
        intent_scores: dict[Intent, float] = {intent: 0.0 for intent in Intent}
        
        query_lower = query.lower()

        # Score each intent based on pattern matches
        for intent, patterns in self._compiled_intent_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(query_lower)
                if matches:
                    # Weight by position - earlier matches are more important
                    for match in matches:
                        if isinstance(match, str):
                            match_pos = query_lower.find(match.lower())
                        else:
                            match_pos = query_lower.find(match[0].lower()) if match else 0
                        position_weight = 1.0 - (match_pos / max(len(query_lower), 1)) * 0.3
                        intent_scores[intent] += position_weight

        # Special handling for conversion patterns
        conversion_match = self._conversion_pattern.search(query_lower)
        if conversion_match:
            intent_scores[Intent.CONVERT] += 1.5

        # Get sorted intents by score
        sorted_intents = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Primary intent
        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]

        # Calculate confidence
        if primary_score == 0:
            # No matches - default to SEARCH with low confidence
            confidence = 0.3
            primary_intent = Intent.SEARCH
        else:
            # Normalize confidence based on score
            max_possible_score = 3.0  # Approximate max
            confidence = min(primary_score / max_possible_score, 1.0)
            confidence = max(confidence, 0.5)  # Minimum 0.5 if any match

        # Secondary intents (score > 0 and not primary)
        secondary_intents = [
            intent for intent, score in sorted_intents[1:4]
            if score > 0 and score >= primary_score * 0.5
        ]

        return primary_intent, confidence, secondary_intents

    def extract_entities(self, original_query: str, normalized_query: str) -> list[Entity]:
        """Extract entities from the query.
        
        Args:
            original_query: Original user query
            normalized_query: Normalized query string
            
        Returns:
            List of extracted Entity objects
        """
        entities: list[Entity] = []
        seen_values: set[str] = set()

        # Extract file types
        for file_type, patterns in self._compiled_file_types.items():
            for pattern in patterns:
                for match in pattern.finditer(original_query):
                    value = file_type
                    if value not in seen_values:
                        entities.append(Entity(
                            type="file_type",
                            value=value,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.9,
                        ))
                        seen_values.add(value)
                        break

        # Extract actions
        for action, patterns in self._compiled_actions.items():
            for pattern in patterns:
                for match in pattern.finditer(normalized_query):
                    value = action
                    if value not in seen_values:
                        entities.append(Entity(
                            type="action",
                            value=value,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.85,
                        ))
                        seen_values.add(value)
                        break

        # Extract targets
        for target, patterns in self._compiled_targets.items():
            for pattern in patterns:
                for match in pattern.finditer(normalized_query):
                    value = target
                    if value not in seen_values:
                        entities.append(Entity(
                            type="target",
                            value=value,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.85,
                        ))
                        seen_values.add(value)
                        break

        # Extract file paths
        for match in self._path_pattern.finditer(original_query):
            path_value = match.group().strip("\"'")
            if path_value not in seen_values:
                entities.append(Entity(
                    type="path",
                    value=path_value,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                ))
                seen_values.add(path_value)

        # Extract quoted strings
        for match in self._quoted_pattern.finditer(original_query):
            quoted_value = match.group(1)
            if quoted_value not in seen_values and len(quoted_value) > 1:
                entities.append(Entity(
                    type="quoted_string",
                    value=quoted_value,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                ))
                seen_values.add(quoted_value)

        # Extract numbers
        for match in self._number_pattern.finditer(original_query):
            num_value = match.group(1)
            if num_value not in seen_values:
                entities.append(Entity(
                    type="number",
                    value=num_value,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                ))
                seen_values.add(num_value)

        # Extract conversion source/target
        conversion_match = self._conversion_pattern.search(original_query)
        if conversion_match:
            source = conversion_match.group(1).lower()
            target = conversion_match.group(2).lower()
            
            # Check if source/target are known file types
            if source in self._file_type_patterns and source not in seen_values:
                entities.append(Entity(
                    type="source_format",
                    value=source,
                    start=conversion_match.start(1),
                    end=conversion_match.end(1),
                    confidence=0.9,
                ))
                seen_values.add(f"source_{source}")
            
            if target in self._file_type_patterns and target not in seen_values:
                entities.append(Entity(
                    type="target_format",
                    value=target,
                    start=conversion_match.start(2),
                    end=conversion_match.end(2),
                    confidence=0.9,
                ))
                seen_values.add(f"target_{target}")

        # Sort by position
        entities.sort(key=lambda e: e.start)

        return entities

    def _normalize_query(self, query: str) -> str:
        """Normalize a query string for processing.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        # Remove common punctuation but keep paths intact
        # Don't remove . / \ : as they're used in paths
        normalized = re.sub(r"[,;!?(){}[\]]+", " ", normalized)
        
        # Clean up whitespace again
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        return normalized

    def _extract_keywords(self, normalized_query: str) -> list[str]:
        """Extract significant keywords from the query.
        
        Args:
            normalized_query: Normalized query string
            
        Returns:
            List of keywords
        """
        # Split into words
        words = re.findall(r"\b\w+\b", normalized_query.lower())
        
        # Filter stopwords and short words
        keywords = [
            word for word in words
            if word not in self._stopwords
            and len(word) > 2
            and not word.isdigit()
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]  # Limit to 10 keywords

    def get_intent_verbs(self, intent: Intent) -> list[str]:
        """Get the verbs associated with an intent.
        
        Args:
            intent: The intent to get verbs for
            
        Returns:
            List of verb patterns (without regex markers)
        """
        patterns = self._intent_patterns.get(intent, [])
        verbs = []
        for pattern in patterns:
            # Extract the word from the pattern
            match = re.search(r"\\b(\w+)\\b", pattern)
            if match:
                verbs.append(match.group(1))
        return verbs

    def is_ambiguous(self, result: QueryResult) -> bool:
        """Check if a query result is ambiguous.
        
        Args:
            result: QueryResult to check
            
        Returns:
            True if the result is ambiguous
        """
        # Ambiguous if low confidence
        if result.confidence < 0.5:
            return True
        
        # Ambiguous if multiple strong secondary intents
        if len(result.secondary_intents) >= 2:
            return True
        
        # Ambiguous if no entities extracted
        if not result.entities and result.confidence < 0.7:
            return True
        
        return False

    def suggest_clarification(self, result: QueryResult) -> Optional[str]:
        """Suggest a clarification question for ambiguous queries.
        
        Args:
            result: Ambiguous QueryResult
            
        Returns:
            Clarification question or None if not needed
        """
        if not self.is_ambiguous(result):
            return None

        if result.confidence < 0.4:
            return "Could you please rephrase your query? I'm not sure what you're trying to do."

        if len(result.secondary_intents) >= 2:
            intent_names = [result.intent.value] + [i.value for i in result.secondary_intents[:2]]
            return f"Do you want to {intent_names[0]}, {intent_names[1]}, or {intent_names[2]}?"

        if not result.entities:
            return f"What would you like to {result.intent.value}? Please specify the target."

        return None
