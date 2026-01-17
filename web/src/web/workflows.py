#!/usr/bin/env python3
"""Intelligent API Workflow Automation with Learning Capabilities."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from .core import Web


class StepType(str, Enum):
    """Workflow step types."""

    API_CALL = "api_call"
    CONDITION = "condition"
    LOOP = "loop"
    DELAY = "delay"
    TRANSFORM = "transform"
    EXTRACT = "extract"


class WorkflowStep(BaseModel):
    """Single step in a workflow."""

    id: str
    type: StepType
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    on_success: Optional[str] = None  # Next step ID on success
    on_error: Optional[str] = None  # Next step ID on error
    retry_count: int = 0
    retry_delay: float = 1.0


class WorkflowDefinition(BaseModel):
    """Complete workflow definition."""

    id: str
    name: str
    description: str
    api_type: str  # e.g., "unreal_remote_control", "generic"
    base_url: str
    auth: Optional[Dict[str, Any]] = None
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = Field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    last_run: Optional[str] = None


class WorkflowEngine:
    """Intelligent workflow execution engine with learning capabilities."""

    def __init__(self, web: Web):
        self.web = web
        self.db = web.db
        self._init_workflow_tables()
        self._workflow_state: Dict[str, Any] = {}

    def _init_workflow_tables(self):
        """Initialize workflow storage tables."""
        # Workflow definitions
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                api_type TEXT NOT NULL,
                base_url TEXT NOT NULL,
                auth JSON,
                steps JSON NOT NULL,
                variables JSON,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_run TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Workflow execution history
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_executions (
                id INTEGER PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                result JSON,
                error TEXT,
                execution_context JSON,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id)
            )
        """
        )

        # Learned patterns from workflow executions
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_patterns (
                id INTEGER PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_data JSON NOT NULL,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id)
            )
        """
        )

    async def save_workflow(self, workflow: WorkflowDefinition) -> None:
        """Save or update a workflow definition."""
        self.db.execute(
            """
            INSERT INTO workflows
            (id, name, description, api_type, base_url, auth, steps, variables)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                api_type = excluded.api_type,
                base_url = excluded.base_url,
                auth = excluded.auth,
                steps = excluded.steps,
                variables = excluded.variables,
                updated_at = CURRENT_TIMESTAMP
        """,
            (
                workflow.id,
                workflow.name,
                workflow.description,
                workflow.api_type,
                workflow.base_url,
                json.dumps(workflow.auth) if workflow.auth else None,
                json.dumps([s.model_dump() for s in workflow.steps]),
                json.dumps(workflow.variables),
            ),
        )
        logger.info(f"Saved workflow: {workflow.id}")

    async def load_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Load a workflow definition."""
        result = self.db.execute(
            "SELECT * FROM workflows WHERE id = ?",
            (workflow_id,)
        ).fetchone()

        if not result:
            return None

        return WorkflowDefinition(
            id=result[0],
            name=result[1],
            description=result[2] or "",
            api_type=result[3],
            base_url=result[4],
            auth=json.loads(result[5]) if result[5] else None,
            steps=[
                WorkflowStep(**step_data)
                for step_data in json.loads(result[6])
            ],
            variables=json.loads(result[7]) if result[7] else {},
            success_count=result[8] or 0,
            failure_count=result[9] or 0,
            last_run=result[10],
        )

    async def execute_workflow(
        self,
        workflow_id: str,
        input_variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow with optional input variables."""
        workflow = await self.load_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Initialize execution context
        context = {
            "workflow_id": workflow_id,
            "variables": {**workflow.variables, **(input_variables or {})},
            "results": {},
            "current_step": None,
            "errors": [],
        }

        execution_id = self._start_execution(workflow_id, context)

        try:
            # Execute steps
            current_step_id = workflow.steps[0].id if workflow.steps else None

            while current_step_id:
                step = next(
                    (s for s in workflow.steps if s.id == current_step_id),
                    None
                )
                if not step:
                    break

                context["current_step"] = step.id
                logger.info(f"Executing step: {step.name} ({step.id})")

                # Execute step with retries
                step_result = None
                for attempt in range(step.retry_count + 1):
                    try:
                        step_result = await self._execute_step(
                            step, workflow, context
                        )
                        break
                    except Exception:
                        if attempt < step.retry_count:
                            logger.warning(
                                f"Step {step.id} failed, retrying "
                                f"({attempt + 1}/{step.retry_count})"
                            )
                            await self._delay(step.retry_delay)
                        else:
                            raise

                # Store result
                context["results"][step.id] = step_result

                # Learn from execution
                await self._learn_from_step(
                    workflow_id, step, step_result, context
                )

                # Determine next step
                if step_result.get("success", False):
                    current_step_id = step.on_success
                else:
                    current_step_id = step.on_error

                if not current_step_id:
                    break

            # Mark as successful
            self._complete_execution(execution_id, "success", context)
            await self._update_workflow_stats(workflow_id, success=True)

            return {
                "success": True,
                "execution_id": execution_id,
                "results": context["results"],
                "variables": context["variables"],
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self._complete_execution(execution_id, "failed", context, str(e))
            await self._update_workflow_stats(workflow_id, success=False)
            raise

    async def _execute_step(
        self,
        step: WorkflowStep,
        workflow: WorkflowDefinition,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        if step.type == StepType.API_CALL:
            return await self._execute_api_call(step, workflow, context)
        elif step.type == StepType.CONDITION:
            return await self._execute_condition(step, context)
        elif step.type == StepType.DELAY:
            return await self._execute_delay(step, context)
        elif step.type == StepType.TRANSFORM:
            return await self._execute_transform(step, context)
        elif step.type == StepType.EXTRACT:
            return await self._execute_extract(step, context)
        else:
            raise ValueError(f"Unknown step type: {step.type}")

    async def _execute_api_call(
        self,
        step: WorkflowStep,
        workflow: WorkflowDefinition,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute an API call step."""
        config = step.config
        url = config.get("url", "")
        method = config.get("method", "GET")
        headers = config.get("headers", {})
        body = config.get("body", None)
        params = config.get("params", None)

        # Resolve variables in URL, headers, body, params
        url = self._resolve_variables(url, context["variables"])
        headers = self._resolve_variables(headers, context["variables"])
        if body:
            body = self._resolve_variables(body, context["variables"])
        if params:
            params = self._resolve_variables(params, context["variables"])

        # Build full URL
        if not url.startswith("http"):
            url = f"{workflow.base_url.rstrip('/')}/{url.lstrip('/')}"

        # Use learned patterns if available
        learned = await self._get_learned_pattern(
            workflow.id, step.id, url, method
        )
        if learned:
            logger.info(f"Using learned pattern for {step.id}")
            headers = {**learned.get("headers", {}), **headers}
            if not body and learned.get("body"):
                body = learned["body"]

        # Execute API call
        result = await self.web.execute(
            "api",
            url=url,
            method=method,
            headers=headers,
            json_data=body,
            params=params,
            auth=workflow.auth,
        )

        # Parse result
        try:
            # Extract status from result string
            if "Status:" in result:
                status_line = [
                    line for line in result.split("\n") if "Status:" in line
                ][0]
                status = int(
                    status_line.split("Status:")[1].strip().split()[0]
                )
            else:
                status = 200

            success = 200 <= status < 300

            # Extract response body if available
            response_body = None
            if "Response:" in result:
                response_part = result.split("Response:")[1].strip()
                try:
                    response_body = json.loads(response_part)
                except Exception:
                    response_body = response_part

            return {
                "success": success,
                "status": status,
                "body": response_body,
                "raw": result,
            }
        except Exception as e:
            logger.error(f"Failed to parse API result: {e}")
            return {"success": False, "error": str(e), "raw": result}

    async def _execute_condition(
        self, step: WorkflowStep, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a condition step."""
        config = step.config
        condition = config.get("condition", "")
        variable = config.get("variable", "")

        # Evaluate condition using JSONata
        value = context["variables"].get(variable)
        value_json = json.dumps(value)

        try:
            # JSONata syntax: jsonata(expression, json_data)
            result = self.db.execute(
                "SELECT jsonata(?, ?::JSON) AS result",
                (condition, value_json),
            ).fetchone()[0]

            # Convert to boolean
            success = bool(result) if result is not None else False

            return {"success": success, "result": result}
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_delay(
        self, step: WorkflowStep, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a delay step."""
        delay = step.config.get("duration", 1.0)
        await self._delay(delay)
        return {"success": True, "delayed": delay}

    async def _execute_transform(
        self, step: WorkflowStep, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a transform step using JSONata."""
        config = step.config
        expression = config.get("expression", "")
        input_var = config.get("input", "")

        input_data = context["variables"].get(input_var, {})
        input_json = json.dumps(input_data)

        try:
            result = self.db.execute(
                "SELECT jsonata(?, ?::JSON) AS result",
                (expression, input_json),
            ).fetchone()[0]

            # Store result in variable
            output_var = config.get("output", input_var)
            parsed_result = (
                json.loads(result) if isinstance(result, str) else result
            )
            context["variables"][output_var] = parsed_result

            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_extract(
        self, step: WorkflowStep, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an extract step using JSONata."""
        config = step.config
        expression = config.get("expression", "")
        input_var = config.get("input", "")

        input_data = context["variables"].get(input_var, {})
        input_json = json.dumps(input_data)

        try:
            result = self.db.execute(
                "SELECT jsonata(?, ?::JSON) AS result",
                (expression, input_json),
            ).fetchone()[0]

            # Store extracted value
            output_var = config.get("output", "extracted")
            parsed_result = (
                json.loads(result) if isinstance(result, str) else result
            )
            context["variables"][output_var] = parsed_result

            return {"success": True, "extracted": result}
        except Exception as e:
            logger.error(f"Extract failed: {e}")
            return {"success": False, "error": str(e)}

    def _resolve_variables(
        self, template: Any, variables: Dict[str, Any]
    ) -> Any:
        """Resolve variables in template using minijinja."""
        if isinstance(template, dict):
            return {
                k: self._resolve_variables(v, variables)
                for k, v in template.items()
            }
        elif isinstance(template, list):
            return [
                self._resolve_variables(item, variables) for item in template
            ]
        elif isinstance(template, str) and "{{" in template:
            # Use minijinja to resolve
            try:
                result = self.db.execute(
                    "SELECT minijinja_render(?, ?) AS result",
                    (template, json.dumps(variables)),
                ).fetchone()[0]
                # Try to parse as JSON if possible
                try:
                    return json.loads(result)
                except Exception:
                    return result
            except Exception:
                return template
        else:
            return template

    async def _get_learned_pattern(
        self, workflow_id: str, step_id: str, url: str, method: str
    ) -> Optional[Dict[str, Any]]:
        """Get learned pattern for a step."""
        result = self.db.execute(
            """
            SELECT pattern_data, success_rate
            FROM workflow_patterns
            WHERE workflow_id = ? AND step_id = ? AND pattern_type = 'api_call'
            ORDER BY success_rate DESC, usage_count DESC
            LIMIT 1
        """,
            (workflow_id, step_id),
        ).fetchone()

        if result and result[1] > 0.7:  # Only use if success rate > 70%
            return json.loads(result[0])
        return None

    async def _learn_from_step(
        self,
        workflow_id: str,
        step: WorkflowStep,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Learn from step execution and store patterns."""
        if step.type != StepType.API_CALL:
            return

        success = result.get("success", False)
        pattern_data = {
            "url": step.config.get("url", ""),
            "method": step.config.get("method", "GET"),
            "headers": step.config.get("headers", {}),
            "body": step.config.get("body"),
        }

        # Check if pattern exists
        existing = self.db.execute(
            """
            SELECT id, success_rate, usage_count
            FROM workflow_patterns
            WHERE workflow_id = ? AND step_id = ? AND pattern_type = 'api_call'
            LIMIT 1
        """,
            (workflow_id, step.id),
        ).fetchone()

        if existing:
            # Update existing pattern
            pattern_id, old_rate, count = existing
            new_count = count + 1
            new_rate = (
                (old_rate * count) + (1.0 if success else 0.0)
            ) / new_count

            self.db.execute(
                """
                UPDATE workflow_patterns
                SET success_rate = ?, usage_count = ?,
                    last_used = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (new_rate, new_count, pattern_id),
            )
        else:
            # Create new pattern
            self.db.execute(
                """
                INSERT INTO workflow_patterns
                (workflow_id, step_id, pattern_type, pattern_data,
                 success_rate)
                VALUES (?, ?, 'api_call', ?, ?)
            """,
                (
                    workflow_id,
                    step.id,
                    json.dumps(pattern_data),
                    1.0 if success else 0.0,
                ),
            )

    def _start_execution(
        self, workflow_id: str, context: Dict[str, Any]
    ) -> int:
        """Start workflow execution and return execution ID."""
        result = self.db.execute(
            """
            INSERT INTO workflow_executions
            (workflow_id, status, execution_context)
            VALUES (?, 'running', ?)
            RETURNING id
        """,
            (workflow_id, json.dumps(context)),
        ).fetchone()
        return result[0] if result else 0

    def _complete_execution(
        self,
        execution_id: int,
        status: str,
        context: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        """Complete workflow execution."""
        self.db.execute(
            """
            UPDATE workflow_executions
            SET status = ?, completed_at = CURRENT_TIMESTAMP,
                result = ?, error = ?
            WHERE id = ?
        """,
            (status, json.dumps(context.get("results", {})), error, execution_id),
        )

    async def _update_workflow_stats(
        self, workflow_id: str, success: bool
    ) -> None:
        """Update workflow statistics."""
        if success:
            self.db.execute(
                """
                UPDATE workflows
                SET success_count = success_count + 1,
                    last_run = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (workflow_id,),
            )
        else:
            self.db.execute(
                """
                UPDATE workflows
                SET failure_count = failure_count + 1,
                    last_run = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (workflow_id,),
            )

    async def _delay(self, seconds: float) -> None:
        """Delay execution."""
        import asyncio

        await asyncio.sleep(seconds)


# ============================================================================
# Unreal Remote Control Helpers
# ============================================================================


def create_unreal_remote_control_workflow(
    base_url: str = "http://localhost:30010",
    preset_name: Optional[str] = None,
) -> WorkflowDefinition:
    """Create a workflow for Unreal Remote Control API."""
    steps = []

    # Get presets
    steps.append(
        WorkflowStep(
            id="get_presets",
            type=StepType.API_CALL,
            name="Get Remote Control Presets",
            config={
                "url": "/remote/presets",
                "method": "GET",
            },
            on_success="extract_preset",
        )
    )

    # Extract preset
    steps.append(
        WorkflowStep(
            id="extract_preset",
            type=StepType.EXTRACT,
            name="Extract Preset",
            config={
                "input": "get_presets",
                "expression": preset_name
                if preset_name
                else "$[0].name",
                "output": "preset_name",
            },
            on_success="call_preset",
        )
    )

    # Call preset
    steps.append(
        WorkflowStep(
            id="call_preset",
            type=StepType.API_CALL,
            name="Call Remote Control Preset",
            config={
                "url": "/remote/preset/{{ preset_name }}",
                "method": "POST",
                "body": {},
            },
        )
    )

    return WorkflowDefinition(
        id="unreal_remote_control",
        name="Unreal Remote Control Workflow",
        description="Workflow for Unreal Engine Remote Control API",
        api_type="unreal_remote_control",
        base_url=base_url,
        steps=steps,
    )

