"""Tool execution with enhanced error handling and tracking."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from ..client import call_tool
from ..database import DatabaseConnectionPool
from .schema import ToolSchema, ToolHistorySchema
from .validation import ToolValidator, ToolErrorSuggester, ValidationResult
import duckdb


@dataclass
class ExecutionContext:
    """Context for a tool execution."""

    server: str
    tool: ToolSchema
    arguments: dict[str, Any]
    user: Optional[str] = None
    session_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of a tool execution."""

    success: bool
    result: Any = None
    error: Optional[str] = None
    validation_errors: Optional[ValidationResult] = None
    execution_time_ms: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    suggestions: list[tuple[ToolSchema, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class ToolExecutor:
    """Executes MCP tools with validation and error handling."""

    def __init__(self, db_pool: Optional[DatabaseConnectionPool] = None) -> None:
        """
        Initialize the executor.

        Args:
            db_pool: Optional database connection pool for history tracking
        """
        self.db_pool = db_pool
        self.validator = ToolValidator()
        self.suggester = ToolErrorSuggester()

    def execute(
        self,
        conn: duckdb.DuckDBPyConnection,
        context: ExecutionContext,
        validate: bool = True,
        track_history: bool = True,
    ) -> ExecutionResult:
        """
        Execute a tool with validation and error handling.

        Args:
            conn: DuckDB connection
            context: Execution context with tool and arguments
            validate: Whether to validate arguments before execution
            track_history: Whether to track execution in history

        Returns:
            ExecutionResult with outcome and metadata
        """
        start_time = time.time()
        result = ExecutionResult(success=False)

        try:
            # Validate arguments if requested
            if validate:
                validation_result = self.validator.validate(
                    context.tool, context.arguments
                )
                result.validation_errors = validation_result

                if not validation_result.is_valid:
                    result.error = f"Validation failed: {validation_result}"
                    logger.warning(f"Validation failed for {context.server}/{context.tool.tool_name}")
                    return result

                # Use validated arguments
                arguments = validation_result.validated_args
            else:
                arguments = context.arguments

            # Execute the tool
            logger.info(
                f"Executing {context.server}/{context.tool.tool_name} "
                f"with args {arguments}"
            )
            result.result = call_tool(conn, context.server, context.tool.tool_name, arguments)
            result.success = True

            logger.info(
                f"Successfully executed {context.server}/{context.tool.tool_name}"
            )

        except Exception as e:
            result.error = str(e)
            logger.error(
                f"Tool execution failed: {e}",
                exc_info=True,
            )

            # Get similar tools for suggestions
            try:
                available_tools = self._get_available_tools(conn, context.server)
                result.suggestions = self.suggester.suggest_tools(
                    context.tool.tool_name, available_tools
                )
            except Exception as e:
                logger.warning(f"Could not suggest similar tools: {e}")

        finally:
            # Calculate execution time
            result.execution_time_ms = int((time.time() - start_time) * 1000)

            # Track execution history if requested
            if track_history:
                try:
                    self._track_execution(
                        conn, context, result
                    )
                except Exception as e:
                    logger.warning(f"Could not track execution history: {e}")

        return result

    def _get_available_tools(
        self, conn: duckdb.DuckDBPyConnection, server: str
    ) -> list[ToolSchema]:
        """Get all available tools for a server."""
        try:
            query = """
                SELECT tool_name, description, input_schema, required_params, enabled
                FROM mcp_tools
                WHERE server_name = ? AND enabled = true
                ORDER BY tool_name
            """
            results = conn.execute(query, [server]).fetchall()

            tools = []
            for row in results:
                tool_name, description, input_schema, required_params, enabled = row
                tools.append(
                    ToolSchema(
                        server_name=server,
                        tool_name=tool_name,
                        description=description,
                        input_schema=(
                            json.loads(input_schema)
                            if isinstance(input_schema, str)
                            else input_schema
                        ),
                        required_params=(
                            required_params if isinstance(required_params, list) else []
                        ),
                        enabled=enabled,
                    )
                )
            return tools
        except Exception as e:
            logger.warning(f"Could not retrieve available tools: {e}")
            return []

    def _track_execution(
        self,
        conn: duckdb.DuckDBPyConnection,
        context: ExecutionContext,
        result: ExecutionResult,
    ) -> None:
        """Track tool execution in history."""
        try:
            query = """
                INSERT INTO mcp_tool_history 
                (server_name, tool_name, arguments, result, success, duration_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            # Serialize result
            result_json = None
            if result.result is not None:
                result_json = (
                    json.dumps(result.result)
                    if not isinstance(result.result, str)
                    else result.result
                )

            conn.execute(
                query,
                [
                    context.server,
                    context.tool.tool_name,
                    json.dumps(context.arguments),
                    result_json,
                    result.success,
                    result.execution_time_ms,
                    result.timestamp.isoformat(),
                ],
            )

            logger.debug(
                f"Tracked execution of {context.server}/{context.tool.tool_name}"
            )
        except Exception as e:
            logger.warning(f"Failed to track execution: {e}")


class BatchExecutor:
    """Execute multiple tools in sequence or parallel."""

    def __init__(self, executor: ToolExecutor) -> None:
        """
        Initialize batch executor.

        Args:
            executor: ToolExecutor instance
        """
        self.executor = executor

    def execute_sequential(
        self,
        conn: duckdb.DuckDBPyConnection,
        contexts: list[ExecutionContext],
        stop_on_error: bool = False,
    ) -> list[ExecutionResult]:
        """
        Execute tools sequentially.

        Args:
            conn: DuckDB connection
            contexts: List of execution contexts
            stop_on_error: Stop execution on first error

        Returns:
            List of execution results
        """
        results = []
        for context in contexts:
            result = self.executor.execute(conn, context)
            results.append(result)

            if stop_on_error and not result.success:
                logger.warning(f"Stopping batch execution due to error: {result.error}")
                break

        return results

    def execute_parallel(
        self,
        conn: duckdb.DuckDBPyConnection,
        contexts: list[ExecutionContext],
    ) -> list[ExecutionResult]:
        """
        Execute tools in parallel using asyncio.

        Args:
            conn: DuckDB connection
            contexts: List of execution contexts

        Returns:
            List of execution results
        """
        import asyncio

        async def run_parallel() -> list[ExecutionResult]:
            tasks = [
                asyncio.to_thread(self.executor.execute, conn, context)
                for context in contexts
            ]
            return await asyncio.gather(*tasks, return_exceptions=False)

        try:
            return asyncio.run(run_parallel())
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Fall back to sequential
            return self.execute_sequential(conn, contexts)
