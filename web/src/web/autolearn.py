#!/usr/bin/env python3
"""Auto-Learning System: Discover, Explore, Enhance, Adapt, Improve, Execute, Reflect, Repeat."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from loguru import logger

from .core import Web
from .workflows import WorkflowDefinition, WorkflowEngine, WorkflowStep, StepType


class AutoLearner:
    """Automatic API discovery, exploration, and workflow optimization."""

    def __init__(self, web: Web):
        self.web = web
        self.engine = WorkflowEngine(web)
        self.db = web.db
        self._init_autolearn_tables()

    def _init_autolearn_tables(self):
        """Initialize auto-learning tables."""
        # API discovery cache
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS discovered_apis (
                id INTEGER PRIMARY KEY,
                base_url TEXT UNIQUE NOT NULL,
                api_type TEXT,
                endpoints JSON,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_explored TIMESTAMP,
                exploration_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0
            )
        """
        )

        # Exploration results
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS api_explorations (
                id INTEGER PRIMARY KEY,
                api_id INTEGER NOT NULL,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                request_config JSON,
                response_schema JSON,
                success BOOLEAN,
                response_time_ms INTEGER,
                explored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (api_id) REFERENCES discovered_apis(id)
            )
        """
        )

        # Improvement suggestions
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS improvement_suggestions (
                id INTEGER PRIMARY KEY,
                workflow_id TEXT,
                suggestion_type TEXT,
                suggestion_data JSON,
                priority INTEGER DEFAULT 5,
                applied BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id)
            )
        """
        )

    async def discover_and_explore(
        self, base_url: str, api_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Discover API endpoints and explore them."""
        logger.info(f"🔍 Discovering API: {base_url}")

        # Check if already discovered
        existing = self.db.execute(
            "SELECT id, endpoints FROM discovered_apis WHERE base_url = ?",
            (base_url,),
        ).fetchone()

        if existing:
            api_id, endpoints_json = existing
            endpoints = json.loads(endpoints_json) if endpoints_json else []
            logger.info(f"Found existing discovery with {len(endpoints)} endpoints")
        else:
            # Discover endpoints
            endpoints = await self._discover_endpoints(base_url)
            api_id = self._save_discovery(base_url, api_type, endpoints)

        # Explore each endpoint
        exploration_results = []
        for endpoint in endpoints:
            result = await self._explore_endpoint(api_id, base_url, endpoint)
            exploration_results.append(result)

        # Update discovery stats
        self._update_discovery_stats(api_id, exploration_results)

        return {
            "api_id": api_id,
            "base_url": base_url,
            "endpoints": endpoints,
            "explorations": exploration_results,
        }

    async def _discover_endpoints(self, base_url: str) -> List[str]:
        """Discover API endpoints using multiple strategies."""
        endpoints = []

        # Strategy 1: Try common OpenAPI/Swagger endpoints
        for path in ["/openapi.json", "/swagger.json", "/api-docs", "/swagger.yaml"]:
            try:
                url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
                result = self.db.execute("SELECT http_get(?) AS res", (url,)).fetchone()[
                    0
                ]
                result_json = json.dumps(result) if not isinstance(result, str) else result
                status = self.db.execute(
                    "SELECT jsonata('$.status', ?::JSON)::INT AS status",
                    (result_json,),
                ).fetchone()[0] or 0

                if status == 200:
                    logger.info(f"✅ Found OpenAPI spec at {path}")
                    # Extract endpoints from OpenAPI spec
                    body = self.db.execute(
                        "SELECT jsonata('$.body', ?::JSON) AS body", (result_json,)
                    ).fetchone()[0]
                    if body:
                        try:
                            spec = json.loads(body) if isinstance(body, str) else body
                            endpoints.extend(self._extract_endpoints_from_openapi(spec))
                        except Exception as e:
                            logger.warning(f"Failed to parse OpenAPI spec: {e}")
            except Exception:
                pass

        # Strategy 2: Try common REST endpoints
        common_paths = [
            "/api",
            "/api/v1",
            "/api/v2",
            "/health",
            "/status",
            "/info",
            "/version",
        ]
        for path in common_paths:
            try:
                url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
                result = self.db.execute("SELECT http_get(?) AS res", (url,)).fetchone()[
                    0
                ]
                result_json = json.dumps(result) if not isinstance(result, str) else result
                status = self.db.execute(
                    "SELECT jsonata('$.status', ?::JSON)::INT AS status",
                    (result_json,),
                ).fetchone()[0] or 0

                if status == 200:
                    endpoints.append(path)
                    logger.info(f"✅ Discovered endpoint: {path}")
            except Exception:
                pass

        return list(set(endpoints))  # Remove duplicates

    def _extract_endpoints_from_openapi(self, spec: Dict[str, Any]) -> List[str]:
        """Extract endpoint paths from OpenAPI specification."""
        endpoints = []
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            endpoints.append(path)
        return endpoints

    async def _explore_endpoint(
        self, api_id: int, base_url: str, endpoint: str
    ) -> Dict[str, Any]:
        """Explore a single endpoint with different methods."""
        logger.info(f"🔬 Exploring endpoint: {endpoint}")

        results = {}
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]

        for method in methods:
            try:
                url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
                start_time = asyncio.get_event_loop().time()

                # Try the request
                if method == "GET":
                    result = self.db.execute(
                        "SELECT http_get(?) AS res", (url,)
                    ).fetchone()[0]
                elif method == "POST":
                    # For POST, try with empty body
                    result = self.db.execute(
                        """
                        SELECT http_post(?, headers => MAP {'content-type': 'application/json'}, params => MAP {}) AS res
                        """,
                        (url,),
                    ).fetchone()[0]
                else:
                    # For other methods, skip for now (PUT, DELETE, PATCH need different handling)
                    continue

                elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

                result_json = json.dumps(result) if not isinstance(result, str) else result
                status = self.db.execute(
                    "SELECT jsonata('$.status', ?::JSON)::INT AS status",
                    (result_json,),
                ).fetchone()[0] or 0

                success = 200 <= status < 300

                # Extract response schema
                response_schema = None
                if success:
                    body = self.db.execute(
                        "SELECT jsonata('$.body', ?::JSON) AS body", (result_json,)
                    ).fetchone()[0]
                    if body:
                        try:
                            body_data = json.loads(body) if isinstance(body, str) else body
                            response_schema = self._infer_schema(body_data)
                        except Exception:
                            pass

                # Store exploration
                self.db.execute(
                    """
                    INSERT INTO api_explorations 
                    (api_id, endpoint, method, request_config, response_schema, success, response_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        api_id,
                        endpoint,
                        method,
                        json.dumps({}),
                        json.dumps(response_schema) if response_schema else None,
                        success,
                        elapsed_ms,
                    ),
                )

                results[method] = {
                    "success": success,
                    "status": status,
                    "response_time_ms": elapsed_ms,
                    "schema": response_schema,
                }

                if success:
                    logger.info(f"  ✅ {method} {endpoint}: {status} ({elapsed_ms}ms)")

            except Exception as e:
                logger.warning(f"  ❌ {method} {endpoint} failed: {e}")
                results[method] = {"success": False, "error": str(e)}

        return {"endpoint": endpoint, "methods": results}

    def _infer_schema(self, data: Any) -> Dict[str, Any]:
        """Infer JSON schema from data."""
        if isinstance(data, dict):
            schema = {"type": "object", "properties": {}}
            for key, value in data.items():
                schema["properties"][key] = self._infer_schema(value)
            return schema
        elif isinstance(data, list):
            if data:
                return {"type": "array", "items": self._infer_schema(data[0])}
            return {"type": "array"}
        elif isinstance(data, bool):
            return {"type": "boolean"}
        elif isinstance(data, int):
            return {"type": "integer"}
        elif isinstance(data, float):
            return {"type": "number"}
        elif isinstance(data, str):
            return {"type": "string"}
        else:
            return {"type": "null"}

    def _save_discovery(
        self, base_url: str, api_type: Optional[str], endpoints: List[str]
    ) -> int:
        """Save discovered API."""
        result = self.db.execute(
            """
            INSERT INTO discovered_apis (base_url, api_type, endpoints)
            VALUES (?, ?, ?)
            ON CONFLICT (base_url) DO UPDATE SET
                endpoints = excluded.endpoints,
                last_explored = CURRENT_TIMESTAMP,
                exploration_count = exploration_count + 1
            RETURNING id
        """,
            (base_url, api_type, json.dumps(endpoints)),
        ).fetchone()
        return result[0] if result else 0

    def _update_discovery_stats(
        self, api_id: int, exploration_results: List[Dict[str, Any]]
    ):
        """Update discovery statistics."""
        total = sum(len(r.get("methods", {})) for r in exploration_results)
        successful = sum(
            sum(1 for m in r.get("methods", {}).values() if m.get("success"))
            for r in exploration_results
        )
        success_rate = successful / total if total > 0 else 0.0

        self.db.execute(
            """
            UPDATE discovered_apis
            SET success_rate = ?, last_explored = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (success_rate, api_id),
        )

    async def enhance_workflow(
        self, workflow_id: str
    ) -> Optional[WorkflowDefinition]:
        """Enhance workflow based on learned patterns and suggestions."""
        workflow = await self.engine.load_workflow(workflow_id)
        if not workflow:
            return None

        logger.info(f"🔧 Enhancing workflow: {workflow_id}")

        # Get improvement suggestions
        suggestions = self.db.execute(
            """
            SELECT suggestion_type, suggestion_data, priority
            FROM improvement_suggestions
            WHERE workflow_id = ? AND applied = FALSE
            ORDER BY priority DESC
        """,
            (workflow_id,),
        ).fetchall()

        enhanced_steps = []
        for step in workflow.steps:
            # Apply learned patterns
            learned = await self.engine._get_learned_pattern(
                workflow_id, step.id, step.config.get("url", ""), step.config.get("method", "GET")
            )
            if learned:
                # Enhance step config with learned data
                enhanced_config = {**step.config}
                enhanced_config["headers"] = {
                    **learned.get("headers", {}),
                    **enhanced_config.get("headers", {}),
                }
                enhanced_step = WorkflowStep(
                    **{**step.model_dump(), "config": enhanced_config}
                )
                enhanced_steps.append(enhanced_step)
            else:
                enhanced_steps.append(step)

        # Create enhanced workflow
        enhanced = WorkflowDefinition(
            **{**workflow.model_dump(), "steps": enhanced_steps}
        )

        return enhanced

    async def adapt_workflow(
        self, workflow_id: str, execution_results: Dict[str, Any]
    ) -> Optional[WorkflowDefinition]:
        """Adapt workflow based on execution results."""
        workflow = await self.engine.load_workflow(workflow_id)
        if not workflow:
            return None

        logger.info(f"🔄 Adapting workflow: {workflow_id}")

        # Analyze execution results
        failed_steps = [
            step_id
            for step_id, result in execution_results.get("results", {}).items()
            if not result.get("success", False)
        ]

        # Adapt failed steps
        adapted_steps = []
        for step in workflow.steps:
            if step.id in failed_steps:
                # Increase retry count
                adapted_step = WorkflowStep(
                    **{
                        **step.model_dump(),
                        "retry_count": step.retry_count + 1,
                        "retry_delay": step.retry_delay * 1.5,
                    }
                )
                adapted_steps.append(adapted_step)
            else:
                adapted_steps.append(step)

        adapted = WorkflowDefinition(
            **{**workflow.model_dump(), "steps": adapted_steps}
        )

        return adapted

    async def improve_workflow(
        self, workflow_id: str
    ) -> Optional[WorkflowDefinition]:
        """Improve workflow by analyzing patterns and optimizing."""
        workflow = await self.engine.load_workflow(workflow_id)
        if not workflow:
            return None

        logger.info(f"⚡ Improving workflow: {workflow_id}")

        # Get execution history
        executions = self.db.execute(
            """
            SELECT result, error
            FROM workflow_executions
            WHERE workflow_id = ? AND status = 'success'
            ORDER BY completed_at DESC
            LIMIT 10
        """,
            (workflow_id,),
        ).fetchall()

        # Analyze for optimizations
        # (This is a simplified version - could be much more sophisticated)
        improved_steps = workflow.steps.copy()

        # Optimize based on success patterns
        for step in improved_steps:
            if step.type == StepType.API_CALL:
                # Check if we can optimize based on learned patterns
                pattern = self.db.execute(
                    """
                    SELECT pattern_data, success_rate
                    FROM workflow_patterns
                    WHERE workflow_id = ? AND step_id = ?
                    ORDER BY success_rate DESC
                    LIMIT 1
                """,
                    (workflow_id, step.id),
                ).fetchone()

                if pattern and pattern[1] > 0.9:  # High success rate
                    # Step is already optimal
                    continue

        return WorkflowDefinition(**{**workflow.model_dump(), "steps": improved_steps})

    async def reflect_on_execution(
        self, workflow_id: str, execution_id: int
    ) -> Dict[str, Any]:
        """Reflect on workflow execution and generate insights."""
        execution = self.db.execute(
            """
            SELECT status, result, error, execution_context
            FROM workflow_executions
            WHERE id = ?
        """,
            (execution_id,),
        ).fetchone()

        if not execution:
            return {}

        status, result_json, error, context_json = execution
        result = json.loads(result_json) if result_json else {}
        context = json.loads(context_json) if context_json else {}

        insights = {
            "success": status == "success",
            "total_steps": len(result),
            "successful_steps": sum(
                1 for r in result.values() if isinstance(r, dict) and r.get("success")
            ),
            "failed_steps": sum(
                1
                for r in result.values()
                if isinstance(r, dict) and not r.get("success")
            ),
            "suggestions": [],
        }

        # Generate suggestions
        if insights["failed_steps"] > 0:
            insights["suggestions"].append(
                {
                    "type": "retry_increase",
                    "message": "Consider increasing retry counts for failed steps",
                }
            )

        # Store insights
        self.db.execute(
            """
            INSERT INTO improvement_suggestions (workflow_id, suggestion_type, suggestion_data, priority)
            VALUES (?, ?, ?, ?)
        """,
            (
                workflow_id,
                "reflection",
                json.dumps(insights),
                5,
            ),
        )

        return insights

    async def auto_learn_cycle(
        self,
        base_url: str,
        workflow_id: Optional[str] = None,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Complete auto-learning cycle: Discover → Explore → Enhance → Adapt → Improve → Execute → Reflect → Repeat."""
        logger.info(f"🚀 Starting auto-learning cycle for: {base_url}")

        cycle_results = []

        for iteration in range(max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            logger.info(f"{'='*60}")

            # 1. DISCOVER
            logger.info("📡 Phase 1: DISCOVER")
            discovery = await self.discover_and_explore(base_url)
            cycle_results.append({"iteration": iteration + 1, "phase": "discover", "data": discovery})

            # 2. EXPLORE
            logger.info("🔬 Phase 2: EXPLORE")
            # Already done in discover_and_explore
            exploration = discovery["explorations"]
            cycle_results.append({"iteration": iteration + 1, "phase": "explore", "data": exploration})

            # 3. ENHANCE
            if workflow_id:
                logger.info("🔧 Phase 3: ENHANCE")
                enhanced = await self.enhance_workflow(workflow_id)
                if enhanced:
                    await self.engine.save_workflow(enhanced)
                    cycle_results.append({"iteration": iteration + 1, "phase": "enhance", "data": "Workflow enhanced"})

            # 4. ADAPT
            if workflow_id and iteration > 0:
                logger.info("🔄 Phase 4: ADAPT")
                # Get last execution results
                last_execution = self.db.execute(
                    """
                    SELECT id, result
                    FROM workflow_executions
                    WHERE workflow_id = ?
                    ORDER BY completed_at DESC
                    LIMIT 1
                """,
                    (workflow_id,),
                ).fetchone()

                if last_execution:
                    exec_id, result_json = last_execution
                    results = json.loads(result_json) if result_json else {}
                    adapted = await self.adapt_workflow(workflow_id, {"results": results})
                    if adapted:
                        await self.engine.save_workflow(adapted)
                        cycle_results.append({"iteration": iteration + 1, "phase": "adapt", "data": "Workflow adapted"})

            # 5. IMPROVE
            if workflow_id:
                logger.info("⚡ Phase 5: IMPROVE")
                improved = await self.improve_workflow(workflow_id)
                if improved:
                    await self.engine.save_workflow(improved)
                    cycle_results.append({"iteration": iteration + 1, "phase": "improve", "data": "Workflow improved"})

            # 6. EXECUTE
            if workflow_id:
                logger.info("▶️  Phase 6: EXECUTE")
                try:
                    result = await self.engine.execute_workflow(workflow_id)
                    cycle_results.append({"iteration": iteration + 1, "phase": "execute", "data": result})
                    logger.info(f"✅ Execution successful: {result['execution_id']}")
                except Exception as e:
                    logger.error(f"❌ Execution failed: {e}")
                    cycle_results.append({"iteration": iteration + 1, "phase": "execute", "error": str(e)})

            # 7. REFLECT
            if workflow_id:
                logger.info("🤔 Phase 7: REFLECT")
                last_execution = self.db.execute(
                    """
                    SELECT id FROM workflow_executions
                    WHERE workflow_id = ?
                    ORDER BY completed_at DESC
                    LIMIT 1
                """,
                    (workflow_id,),
                ).fetchone()

                if last_execution:
                    insights = await self.reflect_on_execution(workflow_id, last_execution[0])
                    cycle_results.append({"iteration": iteration + 1, "phase": "reflect", "data": insights})
                    logger.info(f"💡 Insights: {insights.get('successful_steps', 0)}/{insights.get('total_steps', 0)} steps successful")

            # 8. REPEAT (continue loop)
            logger.info("🔁 Phase 8: REPEAT - Continuing to next iteration...")
            await asyncio.sleep(1)  # Brief pause between iterations

        return {
            "base_url": base_url,
            "workflow_id": workflow_id,
            "iterations": max_iterations,
            "results": cycle_results,
        }

