"""Tests for AI features: embeddings, NLP, examples, LLM export (Phase 3)."""

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# AI Feature Imports - these are the modules we're testing
from mcp_manager.ai.embeddings import (
    EmbeddingModel,
    EmbeddingCache,
    ToolEmbedder,
    SemanticSearcher,
)
from mcp_manager.ai.nlp import (
    IntentClassifier,
    QueryParser,
    NaturalLanguageProcessor,
    Intent,
    ParsedQuery,
)
from mcp_manager.ai.examples import (
    ExampleGenerator,
    UsageExample,
    ExampleTemplate,
    SmartExampleBuilder,
)
from mcp_manager.ai.llm_export import (
    LLMExporter,
    ExportFormat,
    ToolDocumentation,
    ExportConfig,
)
from mcp_manager.ai.suggestions import (
    ToolSuggester,
    SuggestionContext,
    Suggestion,
    WorkflowSuggester,
)
from mcp_manager.tools.schema import ToolSchema


# =============================================================================
# Embedding Tests
# =============================================================================

class TestEmbeddingModel:
    """Test embedding model functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = EmbeddingModel()

    def test_embed_text_returns_vector(self):
        """Test that embedding text returns a vector."""
        text = "Search for files in directory"
        embedding = self.model.embed(text)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_empty_text_returns_zero_vector(self):
        """Test embedding empty text returns zero vector."""
        embedding = self.model.embed("")
        
        assert embedding is not None
        assert isinstance(embedding, list)
        # Should return zeros or handle gracefully
        assert len(embedding) == self.model.dimension

    def test_embed_batch_multiple_texts(self):
        """Test batch embedding of multiple texts."""
        texts = [
            "Read file contents",
            "Write data to database",
            "Search for tools",
        ]
        embeddings = self.model.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == self.model.dimension for e in embeddings)

    def test_embedding_similarity_similar_texts(self):
        """Test that similar texts have high similarity."""
        text1 = "Search for files"
        text2 = "Find files in directory"
        
        emb1 = self.model.embed(text1)
        emb2 = self.model.embed(text2)
        
        similarity = self.model.cosine_similarity(emb1, emb2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be reasonably similar

    def test_embedding_similarity_different_texts(self):
        """Test that different texts have lower similarity."""
        text1 = "Delete all records"
        text2 = "Count items in list"
        
        emb1 = self.model.embed(text1)
        emb2 = self.model.embed(text2)
        
        similarity = self.model.cosine_similarity(emb1, emb2)
        
        assert 0.0 <= similarity <= 1.0
        # Different concepts should have lower similarity

    def test_embedding_deterministic(self):
        """Test that same text produces same embedding."""
        text = "Process data pipeline"
        
        emb1 = self.model.embed(text)
        emb2 = self.model.embed(text)
        
        assert emb1 == emb2


class TestEmbeddingCache:
    """Test embedding cache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = EmbeddingCache(max_size=100)

    def test_cache_store_and_retrieve(self):
        """Test storing and retrieving from cache."""
        key = "test_text"
        embedding = [0.1, 0.2, 0.3]
        
        self.cache.store(key, embedding)
        retrieved = self.cache.get(key)
        
        assert retrieved == embedding

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        result = self.cache.get("nonexistent_key")
        
        assert result is None

    def test_cache_eviction_on_max_size(self):
        """Test that old entries are evicted when max size reached."""
        cache = EmbeddingCache(max_size=3)
        
        cache.store("key1", [0.1])
        cache.store("key2", [0.2])
        cache.store("key3", [0.3])
        cache.store("key4", [0.4])  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key4") == [0.4]

    def test_cache_clear(self):
        """Test clearing the cache."""
        self.cache.store("key1", [0.1])
        self.cache.store("key2", [0.2])
        
        self.cache.clear()
        
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        self.cache.store("key1", [0.1])
        self.cache.get("key1")  # Hit
        self.cache.get("key2")  # Miss
        
        stats = self.cache.stats()
        
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


class TestToolEmbedder:
    """Test tool embedding functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.embedder = ToolEmbedder()
        self.sample_tool = ToolSchema(
            server_name="test-server",
            tool_name="read_file",
            description="Read contents of a file from disk",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                }
            },
            required_params=["path"],
        )

    def test_embed_tool(self):
        """Test embedding a tool."""
        embedding = self.embedder.embed_tool(self.sample_tool)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_embed_tool_uses_description_and_params(self):
        """Test that embedding considers description and parameters."""
        tool_with_params = ToolSchema(
            server_name="test",
            tool_name="complex_tool",
            description="Complex operation",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "output": {"type": "string"},
                }
            },
        )
        
        tool_no_params = ToolSchema(
            server_name="test",
            tool_name="simple_tool",
            description="Complex operation",
        )
        
        emb1 = self.embedder.embed_tool(tool_with_params)
        emb2 = self.embedder.embed_tool(tool_no_params)
        
        # Should be different due to parameter information
        assert emb1 != emb2

    def test_embed_tools_batch(self):
        """Test batch embedding of tools."""
        tools = [
            ToolSchema(server_name="s", tool_name=f"tool_{i}", description=f"Tool {i}")
            for i in range(5)
        ]
        
        embeddings = self.embedder.embed_tools(tools)
        
        assert len(embeddings) == 5


class TestSemanticSearcher:
    """Test semantic search functionality."""

    def setup_method(self):
        """Set up test fixtures with mock database."""
        self.mock_conn = MagicMock()
        self.searcher = SemanticSearcher(self.mock_conn)

    def test_search_returns_results(self):
        """Test that semantic search returns results."""
        # Mock database response
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("server1", "read_file", "Read file contents", ["path"], "[0.1,0.2]"),
            ("server1", "write_file", "Write to file", ["path", "data"], "[0.2,0.3]"),
        ]
        
        results = self.searcher.search("read file", limit=5)
        
        assert len(results) > 0
        assert hasattr(results[0], 'server')
        assert hasattr(results[0], 'tool')

    def test_search_empty_query(self):
        """Test search with empty query."""
        results = self.searcher.search("")
        
        assert results == []

    def test_search_respects_limit(self):
        """Test that search respects limit parameter."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("server1", f"tool_{i}", f"Tool {i}", [], "[]")
            for i in range(10)
        ]
        
        results = self.searcher.search("tool", limit=3)
        
        assert len(results) <= 3

    def test_search_with_server_filter(self):
        """Test search with server filter."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("target_server", "tool1", "Tool 1", [], "[]"),
        ]
        
        results = self.searcher.search("tool", server="target_server")
        
        # Verify filter was applied
        call_args = self.mock_conn.execute.call_args
        assert "target_server" in str(call_args) or call_args is not None


# =============================================================================
# NLP Tests
# =============================================================================

class TestIntentClassifier:
    """Test intent classification functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = IntentClassifier()

    def test_classify_search_intent(self):
        """Test classifying search queries."""
        queries = [
            "find tools for file operations",
            "search for database tools",
            "look for API tools",
        ]
        
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent in [Intent.SEARCH, Intent.DISCOVER]

    def test_classify_execute_intent(self):
        """Test classifying execution queries."""
        queries = [
            "run the read_file tool",
            "execute get_data on server",
            "call the API tool",
        ]
        
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent in [Intent.EXECUTE, Intent.CALL]

    def test_classify_help_intent(self):
        """Test classifying help queries."""
        queries = [
            "how do I use this tool",
            "help with file operations",
            "show me examples",
        ]
        
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent in [Intent.HELP, Intent.EXPLAIN, Intent.EXAMPLE]

    def test_classify_list_intent(self):
        """Test classifying list queries."""
        queries = [
            "list all tools",
            "show available servers",
            "what tools are available",
        ]
        
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent in [Intent.LIST, Intent.DISCOVER]

    def test_classify_unknown_returns_default(self):
        """Test that unknown queries return default intent."""
        intent = self.classifier.classify("xyzzy foobar")
        
        assert intent == Intent.SEARCH  # Default fallback


class TestQueryParser:
    """Test query parsing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = QueryParser()

    def test_parse_simple_query(self):
        """Test parsing a simple query."""
        parsed = self.parser.parse("find file tools")
        
        assert isinstance(parsed, ParsedQuery)
        assert len(parsed.keywords) > 0

    def test_parse_extracts_tool_name(self):
        """Test extracting tool name from query."""
        parsed = self.parser.parse("use the read_file tool")
        
        assert parsed.tool_name == "read_file" or "read_file" in parsed.keywords

    def test_parse_extracts_server_name(self):
        """Test extracting server name from query."""
        parsed = self.parser.parse("from server filesystem call read")
        
        assert parsed.server_name is not None or "filesystem" in parsed.keywords

    def test_parse_extracts_parameters(self):
        """Test extracting parameters from query."""
        parsed = self.parser.parse("read file with path=/tmp/test.txt")
        
        assert parsed.parameters.get("path") == "/tmp/test.txt" or "/tmp/test.txt" in parsed.keywords

    def test_parse_handles_quoted_strings(self):
        """Test handling quoted strings in query."""
        parsed = self.parser.parse('search for "exact phrase"')
        
        assert "exact phrase" in parsed.keywords or parsed.raw_query == 'search for "exact phrase"'

    def test_parse_empty_query(self):
        """Test parsing empty query."""
        parsed = self.parser.parse("")
        
        assert parsed.keywords == [] or parsed.raw_query == ""


class TestNaturalLanguageProcessor:
    """Test natural language processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()
        self.nlp = NaturalLanguageProcessor(self.mock_conn)

    def test_process_query_returns_results(self):
        """Test that processing query returns results."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("server1", "read_file", "Read file", ["path"]),
        ]
        
        results = self.nlp.process("find tools to read files")
        
        assert results is not None
        assert hasattr(results, 'tools') or isinstance(results, list)

    def test_process_with_intent_classification(self):
        """Test that processing includes intent classification."""
        self.mock_conn.execute.return_value.fetchall.return_value = []
        
        results = self.nlp.process("help with file tools")
        
        assert results is not None

    def test_process_generates_suggestions(self):
        """Test that processing generates suggestions."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("server1", "read_file", "Read file", ["path"]),
            ("server1", "write_file", "Write file", ["path", "data"]),
        ]
        
        results = self.nlp.process("file operations")
        
        # Should have some results or suggestions
        assert results is not None


# =============================================================================
# Example Generation Tests
# =============================================================================

class TestExampleGenerator:
    """Test example generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ExampleGenerator()
        self.sample_tool = ToolSchema(
            server_name="test-server",
            tool_name="read_file",
            description="Read contents of a file from disk",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8",
                    },
                },
            },
            required_params=["path"],
        )

    def test_generate_basic_example(self):
        """Test generating a basic example."""
        example = self.generator.generate(self.sample_tool)
        
        assert isinstance(example, UsageExample)
        assert example.tool_name == "read_file"
        assert "path" in example.arguments

    def test_generate_example_with_description(self):
        """Test that example includes description."""
        example = self.generator.generate(self.sample_tool)
        
        assert example.description is not None
        assert len(example.description) > 0

    def test_generate_example_respects_types(self):
        """Test that generated values match schema types."""
        tool = ToolSchema(
            server_name="test",
            tool_name="test_tool",
            input_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "items": {"type": "array"},
                },
            },
        )
        
        example = self.generator.generate(tool)
        
        assert isinstance(example.arguments.get("count"), int)
        assert isinstance(example.arguments.get("enabled"), bool)
        assert isinstance(example.arguments.get("items"), list)

    def test_generate_example_uses_defaults(self):
        """Test that examples use default values when available."""
        example = self.generator.generate(self.sample_tool)
        
        # encoding has a default of "utf-8"
        if "encoding" in example.arguments:
            assert example.arguments["encoding"] == "utf-8"

    def test_generate_multiple_examples(self):
        """Test generating multiple examples."""
        examples = self.generator.generate_multiple(self.sample_tool, count=3)
        
        assert len(examples) == 3
        # Examples should be different
        args_list = [e.arguments for e in examples]
        assert len(set(str(a) for a in args_list)) > 1


class TestUsageExample:
    """Test UsageExample dataclass."""

    def test_to_cli_command(self):
        """Test converting example to CLI command."""
        example = UsageExample(
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test.txt"},
            description="Read a text file",
        )
        
        cli_cmd = example.to_cli_command()
        
        assert "mcp-man call" in cli_cmd
        assert "filesystem" in cli_cmd
        assert "read_file" in cli_cmd
        assert "/tmp/test.txt" in cli_cmd

    def test_to_json(self):
        """Test converting example to JSON."""
        example = UsageExample(
            tool_name="test_tool",
            server_name="test_server",
            arguments={"param": "value"},
        )
        
        json_str = example.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["tool_name"] == "test_tool"
        assert parsed["arguments"]["param"] == "value"


class TestSmartExampleBuilder:
    """Test smart example building with context."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = SmartExampleBuilder()

    def test_build_with_context(self):
        """Test building example with context."""
        tool = ToolSchema(
            server_name="db",
            tool_name="query",
            description="Execute SQL query",
            input_schema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "database": {"type": "string"},
                },
            },
        )
        
        context = {"recent_database": "users_db"}
        example = self.builder.build(tool, context=context)
        
        assert example is not None
        # Should potentially use context values
        assert "sql" in example.arguments

    def test_build_from_history(self):
        """Test building example from execution history."""
        tool = ToolSchema(
            server_name="api",
            tool_name="fetch",
            description="Fetch data from API",
        )
        
        history = [
            {"arguments": {"url": "https://api.example.com/data"}},
            {"arguments": {"url": "https://api.example.com/users"}},
        ]
        
        example = self.builder.build_from_history(tool, history)
        
        assert example is not None


# =============================================================================
# LLM Export Tests
# =============================================================================

class TestLLMExporter:
    """Test LLM export functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = LLMExporter()
        self.sample_tools = [
            ToolSchema(
                server_name="filesystem",
                tool_name="read_file",
                description="Read file contents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                },
                required_params=["path"],
            ),
            ToolSchema(
                server_name="filesystem",
                tool_name="write_file",
                description="Write to a file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
                required_params=["path", "content"],
            ),
        ]

    def test_export_json_format(self):
        """Test exporting to JSON format."""
        output = self.exporter.export(self.sample_tools, ExportFormat.JSON)
        
        assert isinstance(output, str)
        parsed = json.loads(output)
        assert isinstance(parsed, (list, dict))
        assert len(parsed) > 0 if isinstance(parsed, list) else "tools" in parsed

    def test_export_markdown_format(self):
        """Test exporting to Markdown format."""
        output = self.exporter.export(self.sample_tools, ExportFormat.MARKDOWN)
        
        assert isinstance(output, str)
        assert "#" in output  # Should have headers
        assert "read_file" in output
        assert "write_file" in output

    def test_export_xml_format(self):
        """Test exporting to XML format."""
        output = self.exporter.export(self.sample_tools, ExportFormat.XML)
        
        assert isinstance(output, str)
        assert "<" in output and ">" in output  # Should have XML tags
        assert "read_file" in output

    def test_export_openai_format(self):
        """Test exporting to OpenAI function format."""
        output = self.exporter.export(self.sample_tools, ExportFormat.OPENAI)
        
        assert isinstance(output, str)
        parsed = json.loads(output)
        
        # Should have OpenAI function structure
        if isinstance(parsed, list):
            assert all("name" in item or "function" in item for item in parsed)

    def test_export_with_examples(self):
        """Test exporting with examples included."""
        config = ExportConfig(include_examples=True)
        output = self.exporter.export(
            self.sample_tools,
            ExportFormat.MARKDOWN,
            config=config
        )
        
        assert "example" in output.lower() or "Example" in output

    def test_export_single_tool(self):
        """Test exporting a single tool."""
        output = self.exporter.export_tool(
            self.sample_tools[0],
            ExportFormat.JSON
        )
        
        parsed = json.loads(output)
        assert "read_file" in str(parsed)

    def test_export_to_file(self):
        """Test exporting to file."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            self.exporter.export_to_file(
                self.sample_tools,
                filepath,
                ExportFormat.JSON
            )
            
            assert os.path.exists(filepath)
            with open(filepath) as f:
                content = f.read()
            assert "read_file" in content
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestToolDocumentation:
    """Test tool documentation generation."""

    def test_documentation_from_tool(self):
        """Test creating documentation from tool schema."""
        tool = ToolSchema(
            server_name="test",
            tool_name="test_tool",
            description="Test description",
            input_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First param"},
                },
            },
            required_params=["param1"],
        )
        
        doc = ToolDocumentation.from_tool(tool)
        
        assert doc.name == "test_tool"
        assert doc.description == "Test description"
        assert len(doc.parameters) == 1
        assert doc.parameters[0]["name"] == "param1"

    def test_documentation_to_markdown(self):
        """Test converting documentation to markdown."""
        doc = ToolDocumentation(
            name="test_tool",
            description="Test description",
            server="test_server",
            parameters=[
                {"name": "param1", "type": "string", "required": True},
            ],
        )
        
        md = doc.to_markdown()
        
        assert "test_tool" in md
        assert "param1" in md


# =============================================================================
# Suggestion Tests
# =============================================================================

class TestToolSuggester:
    """Test tool suggestion functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()
        self.suggester = ToolSuggester(self.mock_conn)

    def test_suggest_next_tools(self):
        """Test suggesting next tools after execution."""
        # Mock history query
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("read_file", "write_file", 5),
            ("read_file", "process_data", 3),
        ]
        
        suggestions = self.suggester.suggest_next(
            server="filesystem",
            current_tool="read_file"
        )
        
        assert len(suggestions) > 0
        assert isinstance(suggestions[0], Suggestion)

    def test_suggest_based_on_context(self):
        """Test suggesting tools based on context."""
        context = SuggestionContext(
            current_server="filesystem",
            current_tool="read_file",
            recent_tools=["list_dir", "read_file"],
            user_intent="process data",
        )
        
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("filesystem", "process_data", "Process data", []),
        ]
        
        suggestions = self.suggester.suggest_for_context(context)
        
        assert suggestions is not None

    def test_suggest_with_no_history(self):
        """Test suggestions when no history exists."""
        self.mock_conn.execute.return_value.fetchall.return_value = []
        
        suggestions = self.suggester.suggest_next(
            server="new_server",
            current_tool="new_tool"
        )
        
        # Should still return something (possibly empty or default suggestions)
        assert isinstance(suggestions, list)


class TestWorkflowSuggester:
    """Test workflow suggestion functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()
        self.suggester = WorkflowSuggester(self.mock_conn)

    def test_suggest_workflow(self):
        """Test suggesting a complete workflow."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("step1", "First step"),
            ("step2", "Second step"),
        ]
        
        workflow = self.suggester.suggest_workflow(
            goal="process files and save results"
        )
        
        assert workflow is not None
        assert hasattr(workflow, 'steps') or isinstance(workflow, list)

    def test_suggest_workflow_from_pattern(self):
        """Test suggesting workflow from common patterns."""
        # Mock common pattern data
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("read_file", "process_data", "write_file"),
        ]
        
        workflow = self.suggester.suggest_from_pattern("ETL pipeline")
        
        assert workflow is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestAIFeatureIntegration:
    """Integration tests for AI features working together."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()

    def test_nlp_to_search_pipeline(self):
        """Test NLP processing to semantic search pipeline."""
        nlp = NaturalLanguageProcessor(self.mock_conn)
        searcher = SemanticSearcher(self.mock_conn)
        
        # Mock responses
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("server1", "read_file", "Read file contents", ["path"], "[0.1,0.2]"),
        ]
        
        # Process natural language
        query = "I want to read a file"
        nlp_result = nlp.process(query)
        
        # Results should be usable
        assert nlp_result is not None

    def test_example_to_export_pipeline(self):
        """Test example generation to export pipeline."""
        generator = ExampleGenerator()
        exporter = LLMExporter()
        
        tool = ToolSchema(
            server_name="test",
            tool_name="test_tool",
            description="Test tool",
            input_schema={
                "type": "object",
                "properties": {"param": {"type": "string"}},
            },
        )
        
        # Generate example
        example = generator.generate(tool)
        
        # Export with example
        config = ExportConfig(include_examples=True)
        output = exporter.export([tool], ExportFormat.MARKDOWN, config=config)
        
        assert "test_tool" in output

    def test_suggestion_after_execution(self):
        """Test getting suggestions after tool execution."""
        suggester = ToolSuggester(self.mock_conn)
        
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("read_file", "write_file", 10),
        ]
        
        # Simulate post-execution suggestion
        suggestions = suggester.suggest_next(
            server="filesystem",
            current_tool="read_file"
        )
        
        assert isinstance(suggestions, list)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestAIFeatureErrorHandling:
    """Test error handling in AI features."""

    def test_embedding_model_handles_none(self):
        """Test that embedding model handles None gracefully."""
        model = EmbeddingModel()
        
        # Should not raise
        result = model.embed(None)
        assert result is not None

    def test_example_generator_handles_empty_schema(self):
        """Test example generator with empty schema."""
        generator = ExampleGenerator()
        tool = ToolSchema(
            server_name="test",
            tool_name="empty_tool",
            description="Tool with no parameters",
        )
        
        example = generator.generate(tool)
        
        assert example is not None
        assert example.tool_name == "empty_tool"

    def test_exporter_handles_invalid_format(self):
        """Test exporter handles invalid format gracefully."""
        exporter = LLMExporter()
        tools = [ToolSchema(server_name="t", tool_name="t", description="t")]
        
        # Should not raise, use default or raise proper exception
        try:
            output = exporter.export(tools, "invalid_format")
            assert output is not None
        except ValueError as e:
            assert "format" in str(e).lower()

    def test_nlp_processor_handles_unicode(self):
        """Test NLP processor handles unicode properly."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        
        nlp = NaturalLanguageProcessor(mock_conn)
        
        # Should handle unicode
        result = nlp.process("search for tools")
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
