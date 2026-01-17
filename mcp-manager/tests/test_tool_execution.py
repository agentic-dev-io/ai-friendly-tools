"""Tests for tool execution, validation, and error handling (Phase 2)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_manager.tools.schema import ToolSchema
from mcp_manager.tools.validation import (
    ToolValidator,
    ToolErrorSuggester,
    ValidationError_,
    ValidationResult,
)
from mcp_manager.tools.execution import (
    ToolExecutor,
    ExecutionContext,
    ExecutionResult,
    BatchExecutor,
)


class TestToolValidator:
    """Test parameter validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ToolValidator()

    def test_validate_required_parameters_present(self):
        """Test validation when all required parameters are present."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
            required_params=["id", "name"],
        )

        arguments = {"id": "123", "name": "test"}
        result = self.validator.validate(tool, arguments)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.validated_args == arguments

    def test_validate_required_parameters_missing(self):
        """Test validation when required parameters are missing."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="A test tool",
            required_params=["id", "name"],
        )

        arguments = {"id": "123"}  # missing 'name'
        result = self.validator.validate(tool, arguments)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].parameter == "name"
        assert "Required" in result.errors[0].message

    def test_validate_type_mismatch_string(self):
        """Test validation when parameter type doesn't match schema."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                },
            },
        )

        arguments = {"id": 123}  # should be string
        result = self.validator.validate(tool, arguments)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].parameter == "id"
        assert result.errors[0].expected_type == "string"
        assert result.errors[0].got_type == "int"

    def test_validate_type_mismatch_integer(self):
        """Test validation for integer type mismatch."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                },
            },
        )

        arguments = {"count": "not a number"}
        result = self.validator.validate(tool, arguments)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].expected_type == "integer"

    def test_validate_unexpected_parameters_warning(self):
        """Test that unexpected parameters generate warnings."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                },
            },
        )

        arguments = {"id": "123", "extra_param": "value"}
        result = self.validator.validate(tool, arguments)

        assert result.is_valid is True  # No error, just warning
        assert len(result.warnings) == 1
        assert "extra_param" in result.warnings[0]

    def test_suggest_similar_parameters(self):
        """Test parameter name suggestions."""
        suggestions = self.validator._suggest_parameters(
            "nam",  # typo for 'name'
            ["id", "name", "description"],
        )

        assert len(suggestions) > 0
        assert "name" in suggestions

    def test_find_similar_expected_params(self):
        """Test finding similar parameter names."""
        similar = self.validator._find_similar_params(
            "identifer",  # typo for 'identifier'
            {"id", "identifier", "name"},
        )

        assert similar is not None
        assert similar in ["id", "identifier"]

    def test_validate_type_boolean_rejection(self):
        """Test that boolean is not accepted as integer."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                },
            },
        )

        arguments = {"count": True}  # bool, not int
        result = self.validator.validate(tool, arguments)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].got_type == "boolean"

    def test_validate_array_type(self):
        """Test array type validation."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "items": {"type": "array"},
                },
            },
        )

        # Valid array
        result = self.validator.validate(tool, {"items": [1, 2, 3]})
        assert result.is_valid is True

        # Invalid - string instead of array
        result = self.validator.validate(tool, {"items": "not an array"})
        assert result.is_valid is False

    def test_validate_object_type(self):
        """Test object type validation."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "config": {"type": "object"},
                },
            },
        )

        # Valid object
        result = self.validator.validate(tool, {"config": {"key": "value"}})
        assert result.is_valid is True

        # Invalid - string instead of object
        result = self.validator.validate(tool, {"config": "not an object"})
        assert result.is_valid is False


class TestToolErrorSuggester:
    """Test tool suggestion on error."""

    def setup_method(self):
        """Set up test fixtures."""
        self.suggester = ToolErrorSuggester()

    def test_suggest_similar_tool_names(self):
        """Test suggestions based on tool name similarity."""
        tools = [
            ToolSchema(
                server_name="test",
                tool_name="get_data",
                description="Get data from server",
            ),
            ToolSchema(
                server_name="test",
                tool_name="get_data_v2",
                description="Get data v2",
            ),
            ToolSchema(
                server_name="test",
                tool_name="fetch_config",
                description="Fetch configuration",
            ),
        ]

        # Similar to 'get_data'
        suggestions = self.suggester.suggest_tools("get_data", tools)

        assert len(suggestions) > 0
        assert suggestions[0][0].tool_name == "get_data"

    def test_suggest_tools_by_error_message(self):
        """Test suggestions based on error message content."""
        tools = [
            ToolSchema(
                server_name="test",
                tool_name="search_database",
                description="Search data in database",
            ),
            ToolSchema(
                server_name="test",
                tool_name="fetch_config",
                description="Fetch configuration file",
            ),
        ]

        # Error message about database
        error_msg = "Cannot connect to database"
        suggestions = self.suggester.suggest_by_description(error_msg, tools)

        assert len(suggestions) > 0
        assert suggestions[0][0].tool_name == "search_database"

    def test_no_suggestions_low_similarity(self):
        """Test that suggestions with low similarity are filtered."""
        tools = [
            ToolSchema(
                server_name="test",
                tool_name="xyz",
                description="Tool xyz",
            ),
        ]

        suggestions = self.suggester.suggest_tools("abc", tools)

        # Should not suggest if similarity is too low
        assert len(suggestions) == 0 or suggestions[0][1] > 0.5

    def test_suggest_max_suggestions_limit(self):
        """Test that max_suggestions parameter is respected."""
        tools = [
            ToolSchema(
                server_name="test",
                tool_name=f"tool_{i}",
                description=f"Tool {i}",
            )
            for i in range(10)
        ]

        suggestions = self.suggester.suggest_tools(
            "tool_5", tools, max_suggestions=3
        )

        assert len(suggestions) <= 3


class TestExecutionResult:
    """Test ExecutionResult data class."""

    def test_to_dict_successful(self):
        """Test converting successful result to dict."""
        result = ExecutionResult(
            success=True,
            result={"data": "test"},
            execution_time_ms=100,
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["result"] == {"data": "test"}
        assert result_dict["execution_time_ms"] == 100

    def test_to_dict_failed(self):
        """Test converting failed result to dict."""
        result = ExecutionResult(
            success=False,
            error="Tool failed",
            execution_time_ms=150,
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is False
        assert result_dict["error"] == "Tool failed"


class TestExecutionContext:
    """Test ExecutionContext data class."""

    def test_execution_context_creation(self):
        """Test creating execution context."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="Test tool",
        )

        context = ExecutionContext(
            server="test-server",
            tool=tool,
            arguments={"param": "value"},
            user="test_user",
            session_id="session123",
        )

        assert context.server == "test-server"
        assert context.tool.tool_name == "test_tool"
        assert context.arguments == {"param": "value"}
        assert context.user == "test_user"
        assert context.session_id == "session123"


class TestToolExecutor:
    """Test tool executor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = ToolExecutor()

    def test_executor_validation_enabled(self):
        """Test that validation is performed when enabled."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="Test tool",
            required_params=["id"],
        )

        context = ExecutionContext(
            server="test-server",
            tool=tool,
            arguments={},  # missing 'id'
        )

        # Mock the connection
        mock_conn = MagicMock()

        result = self.executor.execute(
            mock_conn, context, validate=True, track_history=False
        )

        assert result.success is False
        assert result.validation_errors is not None
        assert not result.validation_errors.is_valid

    def test_executor_validation_disabled(self):
        """Test that validation can be skipped."""
        tool = ToolSchema(
            server_name="test-server",
            tool_name="test_tool",
            description="Test tool",
            required_params=["id"],
        )

        context = ExecutionContext(
            server="test-server",
            tool=tool,
            arguments={},  # missing 'id', but validation is disabled
        )

        # Mock the connection and call_tool
        mock_conn = MagicMock()

        with patch("mcp_manager.tools.execution.call_tool") as mock_call:
            mock_call.return_value = {"result": "success"}

            result = self.executor.execute(
                mock_conn, context, validate=False, track_history=False
            )

            # Should attempt to execute even with missing param
            assert mock_call.called


class TestBatchExecutor:
    """Test batch executor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool_executor = ToolExecutor()
        self.batch_executor = BatchExecutor(self.tool_executor)

    def test_sequential_execution_all_success(self):
        """Test sequential execution when all succeed."""
        tool1 = ToolSchema(
            server_name="test-server",
            tool_name="tool1",
            description="Tool 1",
        )
        tool2 = ToolSchema(
            server_name="test-server",
            tool_name="tool2",
            description="Tool 2",
        )

        contexts = [
            ExecutionContext(
                server="test-server",
                tool=tool1,
                arguments={"param": "value1"},
            ),
            ExecutionContext(
                server="test-server",
                tool=tool2,
                arguments={"param": "value2"},
            ),
        ]

        mock_conn = MagicMock()

        with patch.object(self.tool_executor, "execute") as mock_execute:
            mock_execute.return_value = ExecutionResult(
                success=True,
                result={"status": "ok"},
                execution_time_ms=50,
            )

            results = self.batch_executor.execute_sequential(mock_conn, contexts)

            assert len(results) == 2
            assert all(r.success for r in results)

    def test_sequential_execution_stop_on_error(self):
        """Test that execution stops on error when stop_on_error=True."""
        tool1 = ToolSchema(
            server_name="test-server",
            tool_name="tool1",
            description="Tool 1",
        )
        tool2 = ToolSchema(
            server_name="test-server",
            tool_name="tool2",
            description="Tool 2",
        )

        contexts = [
            ExecutionContext(
                server="test-server",
                tool=tool1,
                arguments={"param": "value1"},
            ),
            ExecutionContext(
                server="test-server",
                tool=tool2,
                arguments={"param": "value2"},
            ),
        ]

        mock_conn = MagicMock()

        with patch.object(self.tool_executor, "execute") as mock_execute:
            # First fails, second should not be called
            mock_execute.side_effect = [
                ExecutionResult(success=False, error="Failed"),
                ExecutionResult(success=True, result={"status": "ok"}),
            ]

            results = self.batch_executor.execute_sequential(
                mock_conn, contexts, stop_on_error=True
            )

            # Should only have 1 result since it stopped on error
            assert len(results) == 1
            assert not results[0].success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
