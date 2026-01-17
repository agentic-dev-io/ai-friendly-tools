"""Workflow/pipeline management for MCP tools."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from ..tools.schema import ToolSchema


class StepStatus(str, Enum):
    """Status of a workflow step."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    
    id: str
    name: str
    server: str
    tool: str
    arguments: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    output_mapping: dict[str, str] = field(default_factory=dict)
    timeout: Optional[int] = None
    retries: int = 0
    continue_on_failure: bool = False
    condition: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "server": self.server,
            "tool": self.tool,
            "arguments": self.arguments,
            "depends_on": self.depends_on,
            "output_mapping": self.output_mapping,
            "timeout": self.timeout,
            "retries": self.retries,
            "continue_on_failure": self.continue_on_failure,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStep":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            server=data["server"],
            tool=data["tool"],
            arguments=data.get("arguments", {}),
            depends_on=data.get("depends_on", []),
            output_mapping=data.get("output_mapping", {}),
            timeout=data.get("timeout"),
            retries=data.get("retries", 0),
            continue_on_failure=data.get("continue_on_failure", False),
            condition=data.get("condition"),
        )


@dataclass
class ConditionalStep:
    """A conditional step in a workflow."""
    
    id: str
    name: str
    condition: str
    if_true: WorkflowStep
    if_false: Optional[WorkflowStep] = None

    def evaluate(self, context: dict[str, Any]) -> Optional[WorkflowStep]:
        """Evaluate the condition and return the appropriate step.
        
        Args:
            context: Execution context with variables
            
        Returns:
            The step to execute based on condition
        """
        try:
            # Simple expression evaluation
            result = eval(self.condition, {"__builtins__": {}}, context)
            if result:
                return self.if_true
            return self.if_false
        except Exception:
            # On error, default to if_false
            return self.if_false


@dataclass
class ParallelSteps:
    """A group of steps to execute in parallel."""
    
    id: str
    name: str
    steps: list[WorkflowStep]
    wait_for_all: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "parallel": True,
            "steps": [s.to_dict() for s in self.steps],
            "wait_for_all": self.wait_for_all,
        }


@dataclass
class StepResult:
    """Result of executing a workflow step."""
    
    step_id: str
    status: StepStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None


@dataclass
class WorkflowResult:
    """Result of executing a workflow."""
    
    workflow_id: str
    status: WorkflowStatus
    step_results: list[StepResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    dry_run: bool = False
    planned_steps: list[str] = field(default_factory=list)

    def is_success(self) -> bool:
        """Check if workflow completed successfully."""
        return self.status == WorkflowStatus.COMPLETED

    def get_step_result(self, step_id: str) -> Optional[StepResult]:
        """Get result for a specific step."""
        for result in self.step_results:
            if result.step_id == step_id:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "step_results": [
                {
                    "step_id": r.step_id,
                    "status": r.status.value,
                    "result": r.result,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for r in self.step_results
            ],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


@dataclass
class Workflow:
    """A workflow definition."""
    
    id: str
    name: str
    steps: list[WorkflowStep | ConditionalStep | ParallelSteps]
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            raise ValueError("Workflow ID is required")
        if not self.name:
            raise ValueError("Workflow name is required")
        if self.steps is None:
            self.steps = []
        if self.created_at is None:
            self.created_at = datetime.now()

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the workflow structure.
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        step_ids = set()
        
        for step in self.steps:
            if isinstance(step, (WorkflowStep, ParallelSteps)):
                step_ids.add(step.id)
        
        # Check dependencies
        for step in self.steps:
            if isinstance(step, WorkflowStep):
                for dep in step.depends_on:
                    if dep not in step_ids:
                        errors.append(f"Step '{step.id}' depends on unknown step '{dep}'")
        
        # Check for cycles
        cycle_errors = self._detect_cycles()
        errors.extend(cycle_errors)
        
        return len(errors) == 0, errors

    def _detect_cycles(self) -> list[str]:
        """Detect circular dependencies."""
        errors = []
        
        # Build dependency graph
        graph: dict[str, list[str]] = {}
        for step in self.steps:
            if isinstance(step, WorkflowStep):
                graph[step.id] = step.depends_on.copy()
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    errors.append("Circular dependency detected in workflow")
                    break
        
        return errors

    def get_execution_order(self) -> list[WorkflowStep]:
        """Get steps in topologically sorted order.
        
        Returns:
            List of steps in execution order
        """
        # Topological sort using Kahn's algorithm
        result = []
        steps_dict: dict[str, WorkflowStep] = {}
        in_degree: dict[str, int] = {}
        
        for step in self.steps:
            if isinstance(step, WorkflowStep):
                steps_dict[step.id] = step
                in_degree[step.id] = len(step.depends_on)
        
        # Start with steps that have no dependencies
        queue = [sid for sid, degree in in_degree.items() if degree == 0]
        
        while queue:
            step_id = queue.pop(0)
            result.append(steps_dict[step_id])
            
            # Reduce in-degree for dependent steps
            for step in self.steps:
                if isinstance(step, WorkflowStep) and step_id in step.depends_on:
                    in_degree[step.id] -= 1
                    if in_degree[step.id] == 0:
                        queue.append(step.id)
        
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [
                s.to_dict() if hasattr(s, 'to_dict') else str(s)
                for s in self.steps
            ],
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Workflow":
        """Create from JSON string."""
        data = json.loads(json_str)
        
        steps = []
        for step_data in data.get("steps", []):
            if step_data.get("parallel"):
                steps.append(ParallelSteps(
                    id=step_data["id"],
                    name=step_data["name"],
                    steps=[WorkflowStep.from_dict(s) for s in step_data["steps"]],
                    wait_for_all=step_data.get("wait_for_all", True),
                ))
            else:
                steps.append(WorkflowStep.from_dict(step_data))
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            tags=data.get("tags", []),
        )


class WorkflowRunner:
    """Execute workflows."""

    def __init__(self, db_connection: Any) -> None:
        """Initialize runner.
        
        Args:
            db_connection: DuckDB connection
        """
        self.conn = db_connection

    def run(
        self,
        workflow: Workflow,
        dry_run: bool = False,
        timeout: Optional[int] = None,
    ) -> WorkflowResult:
        """Run a workflow.
        
        Args:
            workflow: Workflow to execute
            dry_run: If True, only plan execution
            timeout: Overall timeout in seconds
            
        Returns:
            WorkflowResult with execution details
        """
        result = WorkflowResult(
            workflow_id=workflow.id,
            status=WorkflowStatus.PENDING,
            started_at=datetime.now(),
            dry_run=dry_run,
        )
        
        # Validate workflow
        is_valid, errors = workflow.validate()
        if not is_valid:
            result.status = WorkflowStatus.FAILED
            result.error = "; ".join(errors)
            return result
        
        # Get execution order
        ordered_steps = workflow.get_execution_order()
        result.planned_steps = [s.id for s in ordered_steps]
        
        if dry_run:
            result.status = WorkflowStatus.COMPLETED
            return result
        
        # Execute steps
        result.status = WorkflowStatus.RUNNING
        step_outputs: dict[str, Any] = {}
        failed_steps: set[str] = set()
        
        for step in ordered_steps:
            # Check if dependencies failed
            deps_failed = any(dep in failed_steps for dep in step.depends_on)
            
            if deps_failed and not step.continue_on_failure:
                step_result = StepResult(
                    step_id=step.id,
                    status=StepStatus.SKIPPED,
                    error="Dependency failed",
                )
                result.step_results.append(step_result)
                continue
            
            # Execute step
            try:
                step_result = self._execute_step(step, step_outputs)
                result.step_results.append(step_result)
                
                if step_result.status == StepStatus.SUCCESS:
                    step_outputs[step.id] = step_result.result
                else:
                    failed_steps.add(step.id)
                    
            except Exception as e:
                step_result = StepResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    error=str(e),
                )
                result.step_results.append(step_result)
                failed_steps.add(step.id)
        
        # Determine final status
        result.completed_at = datetime.now()
        
        if not failed_steps:
            result.status = WorkflowStatus.COMPLETED
        elif len(failed_steps) < len(ordered_steps):
            result.status = WorkflowStatus.PARTIAL
        else:
            result.status = WorkflowStatus.FAILED
        
        return result

    def _execute_step(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
    ) -> StepResult:
        """Execute a single step.
        
        Args:
            step: Step to execute
            context: Execution context with previous outputs
            
        Returns:
            StepResult
        """
        started_at = datetime.now()
        
        try:
            # Resolve argument references
            arguments = self._resolve_arguments(step.arguments, context)
            
            # Execute tool
            from ..client import call_tool
            result = call_tool(self.conn, step.server, step.tool, arguments)
            
            completed_at = datetime.now()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            
            return StepResult(
                step_id=step.id,
                status=StepStatus.SUCCESS,
                result=result,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            completed_at = datetime.now()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

    def _resolve_arguments(
        self,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve argument references to previous step outputs.
        
        Args:
            arguments: Step arguments (may contain references)
            context: Context with previous outputs
            
        Returns:
            Resolved arguments
        """
        resolved = {}
        
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Reference to previous step output
                ref = value[2:-1]
                parts = ref.split(".")
                
                if parts[0] in context:
                    resolved_value = context[parts[0]]
                    for part in parts[1:]:
                        if isinstance(resolved_value, dict):
                            resolved_value = resolved_value.get(part)
                        else:
                            break
                    resolved[key] = resolved_value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved


class WorkflowBuilder:
    """Builder for creating workflows."""

    def __init__(self) -> None:
        """Initialize builder."""
        self._name: str = ""
        self._description: str = ""
        self._steps: list[WorkflowStep | ParallelSteps] = []
        self._tags: list[str] = []

    def name(self, name: str) -> "WorkflowBuilder":
        """Set workflow name."""
        self._name = name
        return self

    def description(self, description: str) -> "WorkflowBuilder":
        """Set workflow description."""
        self._description = description
        return self

    def tag(self, tag: str) -> "WorkflowBuilder":
        """Add a tag."""
        self._tags.append(tag)
        return self

    def add_step(
        self,
        step_id: str,
        name: str,
        server: str,
        tool: str,
        arguments: dict[str, Any],
        depends_on: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "WorkflowBuilder":
        """Add a step to the workflow."""
        self._steps.append(WorkflowStep(
            id=step_id,
            name=name,
            server=server,
            tool=tool,
            arguments=arguments,
            depends_on=depends_on or [],
            **kwargs,
        ))
        return self

    def add_conditional(
        self,
        step_id: str,
        name: str,
        condition: str,
        if_true: tuple[str, str, str, str, dict],
        if_false: Optional[tuple[str, str, str, str, dict]] = None,
    ) -> "WorkflowBuilder":
        """Add a conditional step."""
        true_step = WorkflowStep(
            id=if_true[0],
            name=if_true[1],
            server=if_true[2],
            tool=if_true[3],
            arguments=if_true[4],
        )
        
        false_step = None
        if if_false:
            false_step = WorkflowStep(
                id=if_false[0],
                name=if_false[1],
                server=if_false[2],
                tool=if_false[3],
                arguments=if_false[4],
            )
        
        self._steps.append(ConditionalStep(
            id=step_id,
            name=name,
            condition=condition,
            if_true=true_step,
            if_false=false_step,
        ))
        return self

    def add_parallel(
        self,
        group_id: str,
        name: str,
        steps: list[tuple[str, str, str, str, dict]],
        wait_for_all: bool = True,
    ) -> "WorkflowBuilder":
        """Add parallel steps."""
        parallel_steps = [
            WorkflowStep(
                id=s[0],
                name=s[1],
                server=s[2],
                tool=s[3],
                arguments=s[4],
            )
            for s in steps
        ]
        
        self._steps.append(ParallelSteps(
            id=group_id,
            name=name,
            steps=parallel_steps,
            wait_for_all=wait_for_all,
        ))
        return self

    def from_template(self, template: dict[str, Any]) -> "WorkflowBuilder":
        """Build from a template dictionary."""
        self._name = template.get("name", "")
        self._description = template.get("description", "")
        self._tags = template.get("tags", [])
        
        for step_data in template.get("steps", []):
            if step_data.get("parallel"):
                self.add_parallel(
                    step_data["id"],
                    step_data["name"],
                    [(s["id"], s["name"], s["server"], s["tool"], s["arguments"])
                     for s in step_data["steps"]],
                )
            else:
                self.add_step(
                    step_data["id"],
                    step_data.get("name", step_data["id"]),
                    step_data["server"],
                    step_data["tool"],
                    step_data.get("arguments", {}),
                    depends_on=step_data.get("depends_on"),
                )
        
        return self

    def build(self) -> Workflow:
        """Build the workflow."""
        import uuid
        
        return Workflow(
            id=str(uuid.uuid4())[:8],
            name=self._name,
            description=self._description,
            steps=self._steps,
            tags=self._tags,
        )


class WorkflowRegistry:
    """Registry for storing and retrieving workflows."""

    def __init__(self, db_connection: Any) -> None:
        """Initialize registry.
        
        Args:
            db_connection: DuckDB connection
        """
        self.conn = db_connection
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure workflows table exists."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS mcp_workflows (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    description TEXT,
                    steps JSON,
                    tags VARCHAR[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        except Exception:
            pass

    def register(self, workflow: Workflow) -> bool:
        """Register a workflow.
        
        Args:
            workflow: Workflow to register
            
        Returns:
            True if successful
        """
        steps_json = json.dumps([
            s.to_dict() if hasattr(s, 'to_dict') else str(s)
            for s in workflow.steps
        ])
        
        self.conn.execute("""
            INSERT INTO mcp_workflows (id, name, description, steps, tags)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                steps = EXCLUDED.steps,
                tags = EXCLUDED.tags,
                updated_at = CURRENT_TIMESTAMP
        """, [
            workflow.id,
            workflow.name,
            workflow.description,
            steps_json,
            workflow.tags,
        ])
        
        return True

    def get(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow or None
        """
        result = self.conn.execute(
            "SELECT id, name, description, steps, created_at FROM mcp_workflows WHERE id = ?",
            [workflow_id]
        ).fetchone()
        
        if not result:
            return None
        
        wf_id, name, description, steps_json, created_at = result
        
        steps_data = json.loads(steps_json) if steps_json else []
        steps = [WorkflowStep.from_dict(s) for s in steps_data]
        
        return Workflow(
            id=wf_id,
            name=name,
            description=description or "",
            steps=steps,
        )

    def list_all(self) -> list[Workflow]:
        """List all workflows.
        
        Returns:
            List of workflows
        """
        results = self.conn.execute(
            "SELECT id, name, description, steps, created_at FROM mcp_workflows ORDER BY name"
        ).fetchall()
        
        workflows = []
        for wf_id, name, description, steps_json, created_at in results:
            steps_data = json.loads(steps_json) if steps_json else []
            steps = [WorkflowStep.from_dict(s) for s in steps_data]
            
            workflows.append(Workflow(
                id=wf_id,
                name=name,
                description=description or "",
                steps=steps,
            ))
        
        return workflows

    def delete(self, workflow_id: str) -> bool:
        """Delete a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if deleted
        """
        self.conn.execute("DELETE FROM mcp_workflows WHERE id = ?", [workflow_id])
        return True

    def update(self, workflow: Workflow) -> bool:
        """Update a workflow.
        
        Args:
            workflow: Workflow to update
            
        Returns:
            True if updated
        """
        return self.register(workflow)


class WorkflowRecommender:
    """Recommend workflows based on patterns and goals."""

    def __init__(self, db_connection: Any) -> None:
        """Initialize recommender.
        
        Args:
            db_connection: DuckDB connection
        """
        self.conn = db_connection

    def recommend_from_history(self, limit: int = 5) -> list[dict[str, Any]]:
        """Recommend workflows from execution history.
        
        Args:
            limit: Maximum recommendations
            
        Returns:
            List of workflow recommendations
        """
        # Find common tool sequences
        sql = """
            WITH sequences AS (
                SELECT 
                    h1.tool_name || ' -> ' || h2.tool_name || ' -> ' || h3.tool_name as pattern,
                    h1.tool_name as step1,
                    h2.tool_name as step2,
                    h3.tool_name as step3,
                    COUNT(*) as freq
                FROM mcp_tool_history h1
                JOIN mcp_tool_history h2 ON h2.timestamp > h1.timestamp
                JOIN mcp_tool_history h3 ON h3.timestamp > h2.timestamp
                GROUP BY h1.tool_name, h2.tool_name, h3.tool_name
                ORDER BY freq DESC
                LIMIT ?
            )
            SELECT pattern, step1, step2, step3, freq FROM sequences
        """
        
        try:
            results = self.conn.execute(sql, [limit]).fetchall()
            
            recommendations = []
            for pattern, step1, step2, step3, freq in results:
                recommendations.append({
                    "name": f"Suggested: {pattern}",
                    "steps": [step1, step2, step3],
                    "frequency": freq,
                    "confidence": min(0.9, freq / 10.0),
                })
            
            return recommendations
            
        except Exception:
            return []

    def recommend_for_goal(self, goal: str) -> list[dict[str, Any]]:
        """Recommend workflows for a goal.
        
        Args:
            goal: Goal description
            
        Returns:
            List of workflow recommendations
        """
        # Extract keywords and find matching tools
        keywords = goal.lower().split()
        keywords = [k for k in keywords if len(k) > 2]
        
        if not keywords:
            return []
        
        recommendations = []
        matching_tools = []
        
        for keyword in keywords[:5]:
            sql = """
                SELECT server_name, tool_name, description
                FROM mcp_tools
                WHERE enabled = true
                AND (tool_name ILIKE '%' || ? || '%' OR description ILIKE '%' || ? || '%')
                LIMIT 3
            """
            
            try:
                results = self.conn.execute(sql, [keyword, keyword]).fetchall()
                matching_tools.extend(results)
            except Exception:
                continue
        
        if matching_tools:
            # Deduplicate
            seen = set()
            unique_tools = []
            for server, tool, desc in matching_tools:
                if tool not in seen:
                    seen.add(tool)
                    unique_tools.append({"server": server, "tool": tool, "description": desc})
            
            recommendations.append({
                "name": f"Workflow for: {goal[:30]}",
                "steps": unique_tools[:5],
                "confidence": 0.6,
            })
        
        return recommendations

    def find_similar(self, workflow: Workflow) -> list[Workflow]:
        """Find similar workflows.
        
        Args:
            workflow: Reference workflow
            
        Returns:
            List of similar workflows
        """
        # Get tools in workflow
        workflow_tools = set()
        for step in workflow.steps:
            if isinstance(step, WorkflowStep):
                workflow_tools.add(step.tool)
        
        if not workflow_tools:
            return []
        
        # Find workflows with overlapping tools
        registry = WorkflowRegistry(self.conn)
        all_workflows = registry.list_all()
        
        similar = []
        for wf in all_workflows:
            if wf.id == workflow.id:
                continue
            
            wf_tools = set()
            for step in wf.steps:
                if isinstance(step, WorkflowStep):
                    wf_tools.add(step.tool)
            
            # Calculate Jaccard similarity
            if wf_tools:
                intersection = len(workflow_tools & wf_tools)
                union = len(workflow_tools | wf_tools)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0.3:
                    similar.append(wf)
        
        return similar

    def detect_patterns(self, limit: int = 5) -> list[dict[str, Any]]:
        """Detect common workflow patterns.
        
        Args:
            limit: Maximum patterns
            
        Returns:
            List of detected patterns
        """
        return self.recommend_from_history(limit)
