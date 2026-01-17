"""Pipeline builder for multi-tool workflow execution."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import duckdb
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from ..client import call_tool
from ..tools.schema import ToolSchema


class StepStatus(str, Enum):
    """Status of a pipeline step execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(str, Enum):
    """Status of a pipeline execution."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStep(BaseModel):
    """A single step in a pipeline workflow."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(description="Human-readable name for this step")
    server: str = Field(description="MCP server name")
    tool: str = Field(description="Tool name to execute")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool"
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of step IDs this step depends on"
    )
    timeout: int = Field(
        default=30000,
        description="Timeout in milliseconds",
        ge=1000,
        le=300000
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries on failure",
        ge=0,
        le=5
    )
    retry_delay_ms: int = Field(
        default=1000,
        description="Delay between retries in milliseconds",
        ge=100,
        le=60000
    )
    condition: Optional[str] = Field(
        default=None,
        description="Optional condition expression (e.g., 'prev.success')"
    )
    transform_output: Optional[str] = Field(
        default=None,
        description="Optional JSONPath expression to transform output"
    )
    on_error: str = Field(
        default="fail",
        description="Error handling: 'fail', 'skip', or 'continue'"
    )

    @field_validator("on_error")
    @classmethod
    def validate_on_error(cls, v: str) -> str:
        """Validate on_error value."""
        valid_values = {"fail", "skip", "continue"}
        if v not in valid_values:
            raise ValueError(f"on_error must be one of {valid_values}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "server": self.server,
            "tool": self.tool,
            "args": self.args,
            "depends_on": self.depends_on,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_delay_ms": self.retry_delay_ms,
            "condition": self.condition,
            "transform_output": self.transform_output,
            "on_error": self.on_error,
        }


class StepResult(BaseModel):
    """Result of executing a pipeline step."""

    step_id: str
    step_name: str
    status: StepStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
        }


class PipelineResult(BaseModel):
    """Result of executing an entire pipeline."""

    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    step_results: list[StepResult] = Field(default_factory=list)
    total_duration_ms: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.status == PipelineStatus.COMPLETED

    @property
    def failed_steps(self) -> list[StepResult]:
        """Get list of failed steps."""
        return [r for r in self.step_results if r.status == StepStatus.FAILED]

    @property
    def completed_steps(self) -> list[StepResult]:
        """Get list of completed steps."""
        return [r for r in self.step_results if r.status == StepStatus.COMPLETED]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "step_results": [r.to_dict() for r in self.step_results],
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class ValidationError(BaseModel):
    """A validation error in the pipeline."""

    step_id: Optional[str] = None
    step_name: Optional[str] = None
    message: str
    severity: str = "error"  # "error" or "warning"


class ValidationResult(BaseModel):
    """Result of pipeline validation."""

    is_valid: bool
    errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationError] = Field(default_factory=list)

    def add_error(
        self,
        message: str,
        step_id: Optional[str] = None,
        step_name: Optional[str] = None
    ) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(
            step_id=step_id,
            step_name=step_name,
            message=message,
            severity="error"
        ))
        self.is_valid = False

    def add_warning(
        self,
        message: str,
        step_id: Optional[str] = None,
        step_name: Optional[str] = None
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationError(
            step_id=step_id,
            step_name=step_name,
            message=message,
            severity="warning"
        ))


class Pipeline(BaseModel):
    """A workflow pipeline with multiple execution steps."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Pipeline name")
    description: Optional[str] = Field(default=None)
    version: str = Field(default="1.0.0")
    steps: list[PipelineStep] = Field(default_factory=list)
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline-level variables available to all steps"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_step(
        self,
        name: str,
        server: str,
        tool: str,
        args: Optional[dict[str, Any]] = None,
        depends_on: Optional[list[str]] = None,
        timeout: int = 30000,
        **kwargs: Any
    ) -> PipelineStep:
        """
        Add a step to the pipeline.

        Args:
            name: Human-readable step name
            server: MCP server name
            tool: Tool name to execute
            args: Tool arguments
            depends_on: List of step IDs this depends on
            timeout: Timeout in milliseconds
            **kwargs: Additional step configuration

        Returns:
            The created PipelineStep
        """
        step = PipelineStep(
            name=name,
            server=server,
            tool=tool,
            args=args or {},
            depends_on=depends_on or [],
            timeout=timeout,
            **kwargs
        )
        self.steps.append(step)
        self.updated_at = datetime.now()
        logger.debug(f"Added step '{name}' (id={step.id}) to pipeline '{self.name}'")
        return step

    def remove_step(self, step_id: str) -> bool:
        """
        Remove a step from the pipeline.

        Args:
            step_id: ID of step to remove

        Returns:
            True if step was removed, False if not found
        """
        initial_count = len(self.steps)
        self.steps = [s for s in self.steps if s.id != step_id]

        if len(self.steps) < initial_count:
            # Remove from dependencies of other steps
            for step in self.steps:
                step.depends_on = [d for d in step.depends_on if d != step_id]
            self.updated_at = datetime.now()
            logger.debug(f"Removed step {step_id} from pipeline '{self.name}'")
            return True

        return False

    def get_step(self, step_id: str) -> Optional[PipelineStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_step_by_name(self, name: str) -> Optional[PipelineStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def validate_flow(
        self,
        conn: Optional[duckdb.DuckDBPyConnection] = None
    ) -> ValidationResult:
        """
        Validate the pipeline configuration and flow.

        Args:
            conn: Optional DuckDB connection for tool validation

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # Check for empty pipeline
        if not self.steps:
            result.add_error("Pipeline has no steps")
            return result

        # Build step ID set for dependency checking
        step_ids = {step.id for step in self.steps}
        step_names = [step.name for step in self.steps]

        # Check for duplicate names
        if len(step_names) != len(set(step_names)):
            result.add_warning("Pipeline has steps with duplicate names")

        # Validate each step
        for step in self.steps:
            # Check dependencies exist
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    result.add_error(
                        f"Dependency '{dep_id}' not found",
                        step_id=step.id,
                        step_name=step.name
                    )

            # Check for self-dependency
            if step.id in step.depends_on:
                result.add_error(
                    "Step cannot depend on itself",
                    step_id=step.id,
                    step_name=step.name
                )

        # Check for circular dependencies
        circular = self._detect_circular_dependencies()
        if circular:
            result.add_error(
                f"Circular dependency detected: {' -> '.join(circular)}"
            )

        # Validate tools exist if connection provided
        if conn:
            self._validate_tools(conn, result)

        return result

    def _detect_circular_dependencies(self) -> Optional[list[str]]:
        """
        Detect circular dependencies in the pipeline.

        Returns:
            List of step IDs in the cycle, or None if no cycle
        """
        # Build adjacency list
        graph: dict[str, list[str]] = {step.id: step.depends_on for step in self.steps}

        # Track visited and recursion stack
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> Optional[list[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            path.pop()
            rec_stack.remove(node)
            return None

        for step_id in graph:
            if step_id not in visited:
                cycle = dfs(step_id)
                if cycle:
                    return cycle

        return None

    def _validate_tools(
        self,
        conn: duckdb.DuckDBPyConnection,
        result: ValidationResult
    ) -> None:
        """Validate that all tools exist in the connected servers."""
        for step in self.steps:
            try:
                # Check if tool exists
                query = """
                    SELECT COUNT(*) FROM mcp_tools
                    WHERE server_name = ? AND tool_name = ? AND enabled = TRUE
                """
                count = conn.execute(query, [step.server, step.tool]).fetchone()
                if not count or count[0] == 0:
                    result.add_warning(
                        f"Tool '{step.tool}' not found on server '{step.server}'",
                        step_id=step.id,
                        step_name=step.name
                    )
            except Exception as e:
                logger.warning(f"Could not validate tool {step.tool}: {e}")

    def get_execution_order(self) -> list[list[PipelineStep]]:
        """
        Get steps in execution order (topological sort).

        Returns:
            List of lists, where each inner list contains steps
            that can be executed in parallel
        """
        if not self.steps:
            return []

        # Build in-degree map
        in_degree: dict[str, int] = {step.id: 0 for step in self.steps}
        graph: dict[str, list[str]] = {step.id: [] for step in self.steps}

        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id in graph:
                    graph[dep_id].append(step.id)
                    in_degree[step.id] += 1

        # Process levels
        result: list[list[PipelineStep]] = []
        step_map = {step.id: step for step in self.steps}

        while in_degree:
            # Find all steps with no dependencies
            ready = [sid for sid, deg in in_degree.items() if deg == 0]

            if not ready:
                # Circular dependency (should be caught by validation)
                logger.error("Circular dependency detected in execution order")
                break

            # Add current level
            level = [step_map[sid] for sid in ready]
            result.append(level)

            # Remove processed steps and update degrees
            for sid in ready:
                del in_degree[sid]
                for dependent in graph.get(sid, []):
                    if dependent in in_degree:
                        in_degree[dependent] -= 1

        return result

    def execute(
        self,
        conn: duckdb.DuckDBPyConnection,
        validate: bool = True,
        parallel: bool = False,
        progress_callback: Optional[Callable[[str, StepResult], None]] = None
    ) -> PipelineResult:
        """
        Execute the pipeline.

        Args:
            conn: DuckDB connection with MCP servers attached
            validate: Whether to validate before execution
            parallel: Whether to execute independent steps in parallel
            progress_callback: Optional callback for step completion

        Returns:
            PipelineResult with execution details
        """
        result = PipelineResult(
            pipeline_id=self.id,
            pipeline_name=self.name,
            status=PipelineStatus.NOT_STARTED,
            started_at=datetime.now()
        )

        logger.info(f"Starting pipeline '{self.name}' with {len(self.steps)} steps")

        # Validate if requested
        if validate:
            validation = self.validate_flow(conn)
            if not validation.is_valid:
                error_msgs = [e.message for e in validation.errors]
                result.status = PipelineStatus.FAILED
                result.error = f"Validation failed: {'; '.join(error_msgs)}"
                result.completed_at = datetime.now()
                logger.error(f"Pipeline validation failed: {result.error}")
                return result

        result.status = PipelineStatus.RUNNING

        # Get execution order
        execution_levels = self.get_execution_order()
        step_results: dict[str, StepResult] = {}

        try:
            for level_idx, level in enumerate(execution_levels):
                logger.debug(
                    f"Executing level {level_idx + 1}/{len(execution_levels)} "
                    f"with {len(level)} steps"
                )

                if parallel and len(level) > 1:
                    # Execute level in parallel
                    level_results = self._execute_parallel(
                        conn, level, step_results
                    )
                else:
                    # Execute sequentially
                    level_results = self._execute_sequential(
                        conn, level, step_results
                    )

                # Process results
                for step_result in level_results:
                    step_results[step_result.step_id] = step_result
                    result.step_results.append(step_result)

                    if progress_callback:
                        progress_callback(step_result.step_id, step_result)

                    # Check if we should stop
                    if step_result.status == StepStatus.FAILED:
                        step = self.get_step(step_result.step_id)
                        if step and step.on_error == "fail":
                            raise Exception(
                                f"Step '{step_result.step_name}' failed: "
                                f"{step_result.error}"
                            )

            result.status = PipelineStatus.COMPLETED
            logger.info(f"Pipeline '{self.name}' completed successfully")

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)
            logger.error(f"Pipeline '{self.name}' failed: {e}")

        finally:
            result.completed_at = datetime.now()
            result.total_duration_ms = int(
                (result.completed_at - result.started_at).total_seconds() * 1000
            )

        return result

    def _execute_sequential(
        self,
        conn: duckdb.DuckDBPyConnection,
        steps: list[PipelineStep],
        previous_results: dict[str, StepResult]
    ) -> list[StepResult]:
        """Execute steps sequentially."""
        results = []
        for step in steps:
            result = self._execute_step(conn, step, previous_results)
            results.append(result)
            previous_results[step.id] = result
        return results

    def _execute_parallel(
        self,
        conn: duckdb.DuckDBPyConnection,
        steps: list[PipelineStep],
        previous_results: dict[str, StepResult]
    ) -> list[StepResult]:
        """Execute steps in parallel using asyncio."""

        async def run_parallel() -> list[StepResult]:
            tasks = [
                asyncio.to_thread(
                    self._execute_step, conn, step, previous_results
                )
                for step in steps
            ]
            return await asyncio.gather(*tasks)

        try:
            return asyncio.run(run_parallel())
        except Exception as e:
            logger.warning(f"Parallel execution failed, falling back to sequential: {e}")
            return self._execute_sequential(conn, steps, previous_results)

    def _execute_step(
        self,
        conn: duckdb.DuckDBPyConnection,
        step: PipelineStep,
        previous_results: dict[str, StepResult]
    ) -> StepResult:
        """Execute a single pipeline step."""
        result = StepResult(
            step_id=step.id,
            step_name=step.name,
            status=StepStatus.PENDING,
            started_at=datetime.now()
        )

        # Check condition
        if step.condition:
            if not self._evaluate_condition(step.condition, previous_results):
                result.status = StepStatus.SKIPPED
                result.completed_at = datetime.now()
                logger.info(f"Step '{step.name}' skipped due to condition")
                return result

        result.status = StepStatus.RUNNING

        # Resolve arguments with variable substitution
        resolved_args = self._resolve_args(step.args, previous_results)

        # Execute with retries
        retries = 0
        last_error: Optional[str] = None

        while retries <= step.retry_count:
            try:
                start_time = time.time()

                # Call the tool
                tool_result = call_tool(conn, step.server, step.tool, resolved_args)

                result.duration_ms = int((time.time() - start_time) * 1000)

                # Transform output if specified
                if step.transform_output:
                    tool_result = self._transform_output(
                        tool_result, step.transform_output
                    )

                result.result = tool_result
                result.status = StepStatus.COMPLETED
                result.retries = retries
                result.completed_at = datetime.now()

                logger.info(
                    f"Step '{step.name}' completed in {result.duration_ms}ms"
                )
                return result

            except Exception as e:
                last_error = str(e)
                retries += 1

                if retries <= step.retry_count:
                    logger.warning(
                        f"Step '{step.name}' failed (attempt {retries}/{step.retry_count + 1}): {e}"
                    )
                    time.sleep(step.retry_delay_ms / 1000)

        # All retries exhausted
        result.status = StepStatus.FAILED
        result.error = last_error
        result.retries = retries - 1
        result.completed_at = datetime.now()
        result.duration_ms = int(
            (result.completed_at - result.started_at).total_seconds() * 1000
        )

        logger.error(f"Step '{step.name}' failed after {retries} attempts: {last_error}")
        return result

    def _evaluate_condition(
        self,
        condition: str,
        previous_results: dict[str, StepResult]
    ) -> bool:
        """
        Evaluate a condition expression.

        Supports:
        - prev.success - previous step succeeded
        - step.{id}.success - specific step succeeded
        - step.{id}.result.{path} - access step result
        """
        try:
            # Build evaluation context
            context: dict[str, Any] = {
                "prev": {},
                "step": {},
                "variables": self.variables,
            }

            # Add previous results
            for step_id, step_result in previous_results.items():
                context["step"][step_id] = {
                    "success": step_result.status == StepStatus.COMPLETED,
                    "result": step_result.result,
                    "error": step_result.error,
                }

            # Set prev to last result
            if previous_results:
                last_id = list(previous_results.keys())[-1]
                context["prev"] = context["step"][last_id]

            # Simple evaluation (not using eval for security)
            if condition == "prev.success":
                return context["prev"].get("success", False)
            elif condition == "prev.failed":
                return not context["prev"].get("success", True)
            elif condition.startswith("step."):
                parts = condition.split(".")
                if len(parts) >= 3:
                    step_id = parts[1]
                    attr = parts[2]
                    return context["step"].get(step_id, {}).get(attr, False)

            # Default to true for unknown conditions
            logger.warning(f"Unknown condition format: {condition}")
            return True

        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return True

    def _resolve_args(
        self,
        args: dict[str, Any],
        previous_results: dict[str, StepResult]
    ) -> dict[str, Any]:
        """
        Resolve variable references in arguments.

        Supports:
        - ${var.name} - pipeline variable
        - ${step.id.result} - step result
        - ${step.id.result.path} - nested result access
        """
        resolved = {}

        for key, value in args.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_string_value(value, previous_results)
            elif isinstance(value, dict):
                resolved[key] = self._resolve_args(value, previous_results)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_string_value(v, previous_results)
                    if isinstance(v, str) else v
                    for v in value
                ]
            else:
                resolved[key] = value

        return resolved

    def _resolve_string_value(
        self,
        value: str,
        previous_results: dict[str, StepResult]
    ) -> Any:
        """Resolve variable references in a string value."""
        import re

        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)

        if not matches:
            return value

        result = value
        for match in matches:
            replacement = self._get_variable_value(match, previous_results)
            if replacement is not None:
                # If entire value is a single variable, return the actual type
                if value == f"${{{match}}}":
                    return replacement
                result = result.replace(f"${{{match}}}", str(replacement))

        return result

    def _get_variable_value(
        self,
        path: str,
        previous_results: dict[str, StepResult]
    ) -> Any:
        """Get value from variable path."""
        parts = path.split(".")

        if parts[0] == "var" and len(parts) > 1:
            # Pipeline variable
            return self.variables.get(parts[1])

        elif parts[0] == "step" and len(parts) >= 3:
            # Step result
            step_id = parts[1]
            step_result = previous_results.get(step_id)
            if not step_result:
                return None

            if parts[2] == "result":
                if len(parts) > 3:
                    # Navigate nested result
                    result = step_result.result
                    for part in parts[3:]:
                        if isinstance(result, dict):
                            result = result.get(part)
                        else:
                            return None
                    return result
                return step_result.result

        return None

    def _transform_output(self, result: Any, transform: str) -> Any:
        """Transform output using a simple path expression."""
        if not transform or not result:
            return result

        try:
            parts = transform.split(".")
            current = result

            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                else:
                    return result

            return current

        except Exception as e:
            logger.warning(f"Failed to transform output: {e}")
            return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize pipeline to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_json(cls, json_str: str) -> "Pipeline":
        """Deserialize pipeline from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pipeline":
        """Create pipeline from dictionary."""
        # Parse steps
        steps = [
            PipelineStep(**step_data)
            for step_data in data.get("steps", [])
        ]

        # Parse dates
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description"),
            version=data.get("version", "1.0.0"),
            steps=steps,
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
        )

    def save(self, path: Path) -> None:
        """Save pipeline to a JSON file."""
        path.write_text(self.to_json())
        logger.info(f"Saved pipeline '{self.name}' to {path}")

    @classmethod
    def load(cls, path: Path) -> "Pipeline":
        """Load pipeline from a JSON file."""
        json_str = path.read_text()
        pipeline = cls.from_json(json_str)
        logger.info(f"Loaded pipeline '{pipeline.name}' from {path}")
        return pipeline


class PipelineBuilder:
    """Fluent builder for creating pipelines."""

    def __init__(self, name: str) -> None:
        """Initialize builder with pipeline name."""
        self._pipeline = Pipeline(name=name)
        self._last_step_id: Optional[str] = None

    def description(self, desc: str) -> "PipelineBuilder":
        """Set pipeline description."""
        self._pipeline.description = desc
        return self

    def version(self, ver: str) -> "PipelineBuilder":
        """Set pipeline version."""
        self._pipeline.version = ver
        return self

    def variable(self, name: str, value: Any) -> "PipelineBuilder":
        """Add a pipeline variable."""
        self._pipeline.variables[name] = value
        return self

    def metadata(self, key: str, value: Any) -> "PipelineBuilder":
        """Add metadata."""
        self._pipeline.metadata[key] = value
        return self

    def step(
        self,
        name: str,
        server: str,
        tool: str,
        args: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> "PipelineBuilder":
        """Add a step to the pipeline."""
        step = self._pipeline.add_step(
            name=name,
            server=server,
            tool=tool,
            args=args,
            **kwargs
        )
        self._last_step_id = step.id
        return self

    def then(
        self,
        name: str,
        server: str,
        tool: str,
        args: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> "PipelineBuilder":
        """Add a step that depends on the previous step."""
        depends_on = [self._last_step_id] if self._last_step_id else []
        step = self._pipeline.add_step(
            name=name,
            server=server,
            tool=tool,
            args=args,
            depends_on=depends_on,
            **kwargs
        )
        self._last_step_id = step.id
        return self

    def parallel(self, *steps: tuple[str, str, str, Optional[dict[str, Any]]]) -> "PipelineBuilder":
        """
        Add multiple steps that execute in parallel after the last step.

        Args:
            steps: Tuples of (name, server, tool, args)
        """
        depends_on = [self._last_step_id] if self._last_step_id else []
        step_ids = []

        for step_def in steps:
            name, server, tool = step_def[:3]
            args = step_def[3] if len(step_def) > 3 else None

            step = self._pipeline.add_step(
                name=name,
                server=server,
                tool=tool,
                args=args,
                depends_on=depends_on.copy()
            )
            step_ids.append(step.id)

        # Set last step to None so next step must explicitly set dependencies
        self._last_step_id = None
        return self

    def join(
        self,
        step_names: list[str],
        name: str,
        server: str,
        tool: str,
        args: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> "PipelineBuilder":
        """Add a step that depends on multiple previous steps."""
        depends_on = []
        for step_name in step_names:
            step = self._pipeline.get_step_by_name(step_name)
            if step:
                depends_on.append(step.id)

        step = self._pipeline.add_step(
            name=name,
            server=server,
            tool=tool,
            args=args,
            depends_on=depends_on,
            **kwargs
        )
        self._last_step_id = step.id
        return self

    def build(self) -> Pipeline:
        """Build and return the pipeline."""
        return self._pipeline
