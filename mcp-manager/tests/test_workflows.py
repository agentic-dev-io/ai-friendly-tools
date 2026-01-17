"""Tests for workflow/pipeline management functionality (Phase 3)."""

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from mcp_manager.ai.workflows import (
    Workflow,
    WorkflowStep,
    WorkflowRunner,
    WorkflowBuilder,
    WorkflowRegistry,
    WorkflowRecommender,
    StepStatus,
    WorkflowStatus,
    StepResult,
    WorkflowResult,
    ConditionalStep,
    ParallelSteps,
)
from mcp_manager.tools.schema import ToolSchema


# =============================================================================
# Workflow Model Tests
# =============================================================================

class TestWorkflowStep:
    """Test WorkflowStep model."""

    def test_create_basic_step(self):
        """Test creating a basic workflow step."""
        step = WorkflowStep(
            id="step1",
            name="Read File",
            server="filesystem",
            tool="read_file",
            arguments={"path": "/tmp/test.txt"},
        )
        
        assert step.id == "step1"
        assert step.name == "Read File"
        assert step.server == "filesystem"
        assert step.tool == "read_file"
        assert step.arguments["path"] == "/tmp/test.txt"

    def test_step_with_dependencies(self):
        """Test step with dependencies on other steps."""
        step = WorkflowStep(
            id="step2",
            name="Process Data",
            server="processor",
            tool="process",
            arguments={},
            depends_on=["step1"],
        )
        
        assert "step1" in step.depends_on

    def test_step_with_output_mapping(self):
        """Test step with output mapping."""
        step = WorkflowStep(
            id="step1",
            name="Get Data",
            server="api",
            tool="fetch",
            arguments={},
            output_mapping={"data": "result.data", "count": "result.count"},
        )
        
        assert step.output_mapping["data"] == "result.data"

    def test_step_to_dict(self):
        """Test converting step to dictionary."""
        step = WorkflowStep(
            id="step1",
            name="Test Step",
            server="test",
            tool="test_tool",
            arguments={"key": "value"},
        )
        
        step_dict = step.to_dict()
        
        assert step_dict["id"] == "step1"
        assert step_dict["server"] == "test"
        assert step_dict["tool"] == "test_tool"

    def test_step_from_dict(self):
        """Test creating step from dictionary."""
        data = {
            "id": "step1",
            "name": "Test Step",
            "server": "test",
            "tool": "test_tool",
            "arguments": {"key": "value"},
        }
        
        step = WorkflowStep.from_dict(data)
        
        assert step.id == "step1"
        assert step.arguments["key"] == "value"


class TestWorkflow:
    """Test Workflow model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.steps = [
            WorkflowStep(
                id="step1",
                name="Read Data",
                server="filesystem",
                tool="read_file",
                arguments={"path": "/data/input.txt"},
            ),
            WorkflowStep(
                id="step2",
                name="Process Data",
                server="processor",
                tool="transform",
                arguments={"format": "json"},
                depends_on=["step1"],
            ),
            WorkflowStep(
                id="step3",
                name="Save Results",
                server="filesystem",
                tool="write_file",
                arguments={"path": "/data/output.txt"},
                depends_on=["step2"],
            ),
        ]

    def test_create_workflow(self):
        """Test creating a workflow."""
        workflow = Workflow(
            id="wf1",
            name="Data Processing Pipeline",
            description="Read, process, and save data",
            steps=self.steps,
        )
        
        assert workflow.id == "wf1"
        assert len(workflow.steps) == 3
        assert workflow.name == "Data Processing Pipeline"

    def test_workflow_validates_dependencies(self):
        """Test that workflow validates step dependencies."""
        invalid_steps = [
            WorkflowStep(
                id="step1",
                name="Step 1",
                server="s",
                tool="t",
                arguments={},
                depends_on=["nonexistent"],  # Invalid dependency
            ),
        ]
        
        workflow = Workflow(
            id="wf1",
            name="Invalid Workflow",
            steps=invalid_steps,
        )
        
        # Should detect invalid dependency
        is_valid, errors = workflow.validate()
        assert not is_valid or "nonexistent" in str(errors)

    def test_workflow_detects_cycles(self):
        """Test that workflow detects circular dependencies."""
        cyclic_steps = [
            WorkflowStep(id="a", name="A", server="s", tool="t", arguments={}, depends_on=["c"]),
            WorkflowStep(id="b", name="B", server="s", tool="t", arguments={}, depends_on=["a"]),
            WorkflowStep(id="c", name="C", server="s", tool="t", arguments={}, depends_on=["b"]),
        ]
        
        workflow = Workflow(id="wf", name="Cyclic", steps=cyclic_steps)
        
        is_valid, errors = workflow.validate()
        assert not is_valid

    def test_workflow_to_json(self):
        """Test serializing workflow to JSON."""
        workflow = Workflow(
            id="wf1",
            name="Test Workflow",
            steps=self.steps[:1],
        )
        
        json_str = workflow.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["id"] == "wf1"
        assert len(parsed["steps"]) == 1

    def test_workflow_from_json(self):
        """Test deserializing workflow from JSON."""
        json_str = json.dumps({
            "id": "wf1",
            "name": "Test Workflow",
            "steps": [
                {
                    "id": "step1",
                    "name": "Step 1",
                    "server": "test",
                    "tool": "test_tool",
                    "arguments": {},
                }
            ],
        })
        
        workflow = Workflow.from_json(json_str)
        
        assert workflow.id == "wf1"
        assert len(workflow.steps) == 1

    def test_workflow_get_execution_order(self):
        """Test getting topologically sorted execution order."""
        workflow = Workflow(
            id="wf1",
            name="Test",
            steps=self.steps,
        )
        
        order = workflow.get_execution_order()
        
        # step1 should come before step2, step2 before step3
        step1_idx = next(i for i, s in enumerate(order) if s.id == "step1")
        step2_idx = next(i for i, s in enumerate(order) if s.id == "step2")
        step3_idx = next(i for i, s in enumerate(order) if s.id == "step3")
        
        assert step1_idx < step2_idx < step3_idx


class TestConditionalStep:
    """Test conditional workflow steps."""

    def test_conditional_step_creation(self):
        """Test creating a conditional step."""
        step = ConditionalStep(
            id="cond1",
            name="Conditional Step",
            condition="steps.step1.result.success == true",
            if_true=WorkflowStep(
                id="true_branch",
                name="Success Path",
                server="s",
                tool="success_tool",
                arguments={},
            ),
            if_false=WorkflowStep(
                id="false_branch",
                name="Failure Path",
                server="s",
                tool="error_handler",
                arguments={},
            ),
        )
        
        assert step.condition is not None
        assert step.if_true.id == "true_branch"
        assert step.if_false.id == "false_branch"

    def test_evaluate_condition_true(self):
        """Test evaluating a true condition."""
        step = ConditionalStep(
            id="cond1",
            name="Check",
            condition="value > 10",
            if_true=WorkflowStep(id="t", name="T", server="s", tool="t", arguments={}),
            if_false=WorkflowStep(id="f", name="F", server="s", tool="f", arguments={}),
        )
        
        context = {"value": 15}
        result = step.evaluate(context)
        
        assert result == step.if_true

    def test_evaluate_condition_false(self):
        """Test evaluating a false condition."""
        step = ConditionalStep(
            id="cond1",
            name="Check",
            condition="value > 10",
            if_true=WorkflowStep(id="t", name="T", server="s", tool="t", arguments={}),
            if_false=WorkflowStep(id="f", name="F", server="s", tool="f", arguments={}),
        )
        
        context = {"value": 5}
        result = step.evaluate(context)
        
        assert result == step.if_false


class TestParallelSteps:
    """Test parallel workflow steps."""

    def test_parallel_steps_creation(self):
        """Test creating parallel steps."""
        steps = [
            WorkflowStep(id="p1", name="P1", server="s1", tool="t1", arguments={}),
            WorkflowStep(id="p2", name="P2", server="s2", tool="t2", arguments={}),
            WorkflowStep(id="p3", name="P3", server="s3", tool="t3", arguments={}),
        ]
        
        parallel = ParallelSteps(
            id="parallel_group",
            name="Parallel Processing",
            steps=steps,
        )
        
        assert len(parallel.steps) == 3

    def test_parallel_wait_for_all(self):
        """Test parallel steps with wait_for_all option."""
        steps = [
            WorkflowStep(id="p1", name="P1", server="s", tool="t", arguments={}),
            WorkflowStep(id="p2", name="P2", server="s", tool="t", arguments={}),
        ]
        
        parallel = ParallelSteps(
            id="pg",
            name="Parallel",
            steps=steps,
            wait_for_all=True,
        )
        
        assert parallel.wait_for_all is True


# =============================================================================
# Workflow Runner Tests
# =============================================================================

class TestWorkflowRunner:
    """Test workflow execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()
        self.runner = WorkflowRunner(self.mock_conn)

    def test_run_simple_workflow(self):
        """Test running a simple workflow."""
        workflow = Workflow(
            id="wf1",
            name="Simple",
            steps=[
                WorkflowStep(
                    id="step1",
                    name="Step 1",
                    server="test",
                    tool="test_tool",
                    arguments={"param": "value"},
                ),
            ],
        )
        
        with patch.object(self.runner, '_execute_step') as mock_exec:
            mock_exec.return_value = StepResult(
                step_id="step1",
                status=StepStatus.SUCCESS,
                result={"data": "test"},
            )
            
            result = self.runner.run(workflow)
            
            assert result.status == WorkflowStatus.COMPLETED
            assert len(result.step_results) == 1

    def test_run_workflow_with_dependencies(self):
        """Test running workflow with step dependencies."""
        workflow = Workflow(
            id="wf1",
            name="With Deps",
            steps=[
                WorkflowStep(
                    id="step1",
                    name="First",
                    server="s",
                    tool="t1",
                    arguments={},
                ),
                WorkflowStep(
                    id="step2",
                    name="Second",
                    server="s",
                    tool="t2",
                    arguments={"input": "${step1.result}"},
                    depends_on=["step1"],
                ),
            ],
        )
        
        with patch.object(self.runner, '_execute_step') as mock_exec:
            mock_exec.side_effect = [
                StepResult(step_id="step1", status=StepStatus.SUCCESS, result={"value": 1}),
                StepResult(step_id="step2", status=StepStatus.SUCCESS, result={"value": 2}),
            ]
            
            result = self.runner.run(workflow)
            
            assert result.status == WorkflowStatus.COMPLETED
            assert len(result.step_results) == 2

    def test_run_workflow_step_failure(self):
        """Test workflow behavior when a step fails."""
        workflow = Workflow(
            id="wf1",
            name="Failing",
            steps=[
                WorkflowStep(id="s1", name="S1", server="s", tool="t", arguments={}),
                WorkflowStep(id="s2", name="S2", server="s", tool="t", arguments={}, depends_on=["s1"]),
            ],
        )
        
        with patch.object(self.runner, '_execute_step') as mock_exec:
            mock_exec.side_effect = [
                StepResult(step_id="s1", status=StepStatus.FAILED, error="Test error"),
                # s2 should not be called due to dependency failure
            ]
            
            result = self.runner.run(workflow)
            
            assert result.status == WorkflowStatus.FAILED
            # Only one step should have run
            executed_count = sum(1 for r in result.step_results if r.status != StepStatus.SKIPPED)
            assert executed_count <= 1

    def test_run_workflow_continue_on_failure(self):
        """Test workflow with continue_on_failure option."""
        workflow = Workflow(
            id="wf1",
            name="Continue",
            steps=[
                WorkflowStep(id="s1", name="S1", server="s", tool="t", arguments={}, continue_on_failure=True),
                WorkflowStep(id="s2", name="S2", server="s", tool="t", arguments={}),
            ],
        )
        
        with patch.object(self.runner, '_execute_step') as mock_exec:
            mock_exec.side_effect = [
                StepResult(step_id="s1", status=StepStatus.FAILED, error="Error but continue"),
                StepResult(step_id="s2", status=StepStatus.SUCCESS, result={}),
            ]
            
            result = self.runner.run(workflow)
            
            # Workflow should complete even with a failed step
            assert result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL]

    def test_run_dry_mode(self):
        """Test dry run mode (no actual execution)."""
        workflow = Workflow(
            id="wf1",
            name="Dry Run",
            steps=[
                WorkflowStep(id="s1", name="S1", server="s", tool="t", arguments={}),
            ],
        )
        
        result = self.runner.run(workflow, dry_run=True)
        
        # Should return a plan, not execute
        assert result.dry_run is True
        assert len(result.planned_steps) == 1

    def test_run_with_timeout(self):
        """Test workflow execution with timeout."""
        workflow = Workflow(
            id="wf1",
            name="Timeout",
            steps=[
                WorkflowStep(id="s1", name="S1", server="s", tool="slow_tool", arguments={}),
            ],
        )
        
        with patch.object(self.runner, '_execute_step') as mock_exec:
            mock_exec.side_effect = TimeoutError("Step timed out")
            
            result = self.runner.run(workflow, timeout=1)
            
            assert result.status == WorkflowStatus.TIMEOUT or result.step_results[0].status == StepStatus.TIMEOUT


class TestWorkflowResult:
    """Test workflow result handling."""

    def test_workflow_result_success(self):
        """Test successful workflow result."""
        result = WorkflowResult(
            workflow_id="wf1",
            status=WorkflowStatus.COMPLETED,
            step_results=[
                StepResult(step_id="s1", status=StepStatus.SUCCESS, result={"data": 1}),
                StepResult(step_id="s2", status=StepStatus.SUCCESS, result={"data": 2}),
            ],
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        
        assert result.is_success()
        assert result.get_step_result("s1").result["data"] == 1

    def test_workflow_result_to_dict(self):
        """Test converting result to dictionary."""
        result = WorkflowResult(
            workflow_id="wf1",
            status=WorkflowStatus.COMPLETED,
            step_results=[],
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["workflow_id"] == "wf1"
        assert result_dict["status"] == "completed"


# =============================================================================
# Workflow Builder Tests
# =============================================================================

class TestWorkflowBuilder:
    """Test workflow builder functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = WorkflowBuilder()

    def test_builder_add_step(self):
        """Test adding steps to builder."""
        workflow = (
            self.builder
            .name("Test Workflow")
            .add_step("step1", "First Step", "server1", "tool1", {"arg": "val"})
            .add_step("step2", "Second Step", "server2", "tool2", {}, depends_on=["step1"])
            .build()
        )
        
        assert workflow.name == "Test Workflow"
        assert len(workflow.steps) == 2

    def test_builder_with_conditional(self):
        """Test adding conditional step."""
        workflow = (
            self.builder
            .name("Conditional Workflow")
            .add_step("check", "Check Data", "s", "validate", {})
            .add_conditional(
                "branch",
                "Branch",
                condition="steps.check.result.valid == true",
                if_true=("success", "Success", "s", "process", {}),
                if_false=("error", "Error", "s", "handle_error", {}),
            )
            .build()
        )
        
        assert len(workflow.steps) == 2

    def test_builder_with_parallel(self):
        """Test adding parallel steps."""
        workflow = (
            self.builder
            .name("Parallel Workflow")
            .add_parallel(
                "parallel_group",
                "Parallel Tasks",
                [
                    ("p1", "Task 1", "s", "t1", {}),
                    ("p2", "Task 2", "s", "t2", {}),
                    ("p3", "Task 3", "s", "t3", {}),
                ],
            )
            .build()
        )
        
        assert len(workflow.steps) == 1
        parallel_step = workflow.steps[0]
        assert hasattr(parallel_step, 'steps') and len(parallel_step.steps) == 3

    def test_builder_from_template(self):
        """Test building from template."""
        template = {
            "name": "ETL Pipeline",
            "steps": [
                {"id": "extract", "name": "Extract", "server": "db", "tool": "query", "arguments": {}},
                {"id": "transform", "name": "Transform", "server": "proc", "tool": "transform", "arguments": {}, "depends_on": ["extract"]},
                {"id": "load", "name": "Load", "server": "db", "tool": "insert", "arguments": {}, "depends_on": ["transform"]},
            ],
        }
        
        workflow = self.builder.from_template(template).build()
        
        assert workflow.name == "ETL Pipeline"
        assert len(workflow.steps) == 3


# =============================================================================
# Workflow Registry Tests
# =============================================================================

class TestWorkflowRegistry:
    """Test workflow registry functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()
        self.registry = WorkflowRegistry(self.mock_conn)

    def test_register_workflow(self):
        """Test registering a workflow."""
        workflow = Workflow(
            id="wf1",
            name="Test Workflow",
            steps=[WorkflowStep(id="s1", name="S1", server="s", tool="t", arguments={})],
        )
        
        result = self.registry.register(workflow)
        
        assert result is True
        self.mock_conn.execute.assert_called()

    def test_get_workflow(self):
        """Test getting a workflow by ID."""
        self.mock_conn.execute.return_value.fetchone.return_value = (
            "wf1",
            "Test",
            "Description",
            json.dumps([{"id": "s1", "name": "S1", "server": "s", "tool": "t", "arguments": {}}]),
            datetime.now().isoformat(),
        )
        
        workflow = self.registry.get("wf1")
        
        assert workflow is not None
        assert workflow.id == "wf1"

    def test_list_workflows(self):
        """Test listing all workflows."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("wf1", "Workflow 1", "Desc 1", "[]", datetime.now().isoformat()),
            ("wf2", "Workflow 2", "Desc 2", "[]", datetime.now().isoformat()),
        ]
        
        workflows = self.registry.list_all()
        
        assert len(workflows) == 2

    def test_delete_workflow(self):
        """Test deleting a workflow."""
        result = self.registry.delete("wf1")
        
        self.mock_conn.execute.assert_called()

    def test_update_workflow(self):
        """Test updating a workflow."""
        workflow = Workflow(
            id="wf1",
            name="Updated Workflow",
            steps=[],
        )
        
        self.registry.update(workflow)
        
        self.mock_conn.execute.assert_called()


# =============================================================================
# Workflow Recommender Tests
# =============================================================================

class TestWorkflowRecommender:
    """Test workflow recommendation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()
        self.recommender = WorkflowRecommender(self.mock_conn)

    def test_recommend_from_history(self):
        """Test recommending workflows from execution history."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("read_file", "transform", "write_file"),  # Common pattern
            ("read_file", "transform", "write_file"),
            ("fetch_api", "parse_json", "save_data"),
        ]
        
        recommendations = self.recommender.recommend_from_history(limit=3)
        
        assert len(recommendations) > 0

    def test_recommend_for_goal(self):
        """Test recommending workflows for a specific goal."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("db", "query", "Query database", []),
            ("proc", "aggregate", "Aggregate data", []),
            ("report", "generate", "Generate report", []),
        ]
        
        recommendations = self.recommender.recommend_for_goal(
            "create a report from database data"
        )
        
        assert recommendations is not None

    def test_recommend_similar_workflows(self):
        """Test recommending similar workflows."""
        workflow = Workflow(
            id="wf1",
            name="Data Pipeline",
            steps=[
                WorkflowStep(id="s1", name="Read", server="fs", tool="read", arguments={}),
                WorkflowStep(id="s2", name="Process", server="proc", tool="transform", arguments={}),
            ],
        )
        
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("wf2", "Similar Pipeline", "", json.dumps([
                {"id": "a", "name": "Load", "server": "fs", "tool": "read", "arguments": {}},
                {"id": "b", "name": "Convert", "server": "proc", "tool": "convert", "arguments": {}},
            ]), ""),
        ]
        
        similar = self.recommender.find_similar(workflow)
        
        assert isinstance(similar, list)

    def test_detect_common_patterns(self):
        """Test detecting common workflow patterns."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ("read_file", "process", "write_file", 10),
            ("fetch_api", "parse", "store", 8),
        ]
        
        patterns = self.recommender.detect_patterns()
        
        assert len(patterns) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestWorkflowIntegration:
    """Integration tests for workflow functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()

    def test_build_and_run_workflow(self):
        """Test building and running a workflow."""
        builder = WorkflowBuilder()
        workflow = (
            builder
            .name("Integration Test")
            .add_step("s1", "Step 1", "server", "tool1", {"key": "value"})
            .add_step("s2", "Step 2", "server", "tool2", {}, depends_on=["s1"])
            .build()
        )
        
        runner = WorkflowRunner(self.mock_conn)
        
        with patch.object(runner, '_execute_step') as mock_exec:
            mock_exec.return_value = StepResult(
                step_id="test",
                status=StepStatus.SUCCESS,
                result={},
            )
            
            result = runner.run(workflow)
            
            assert result is not None

    def test_register_and_retrieve_workflow(self):
        """Test registering and retrieving a workflow."""
        registry = WorkflowRegistry(self.mock_conn)
        
        workflow = Workflow(
            id="test_wf",
            name="Test",
            steps=[WorkflowStep(id="s", name="S", server="s", tool="t", arguments={})],
        )
        
        # Mock the database operations
        self.mock_conn.execute.return_value.fetchone.return_value = (
            workflow.id,
            workflow.name,
            "",
            json.dumps([s.to_dict() for s in workflow.steps]),
            datetime.now().isoformat(),
        )
        
        registry.register(workflow)
        retrieved = registry.get("test_wf")
        
        assert retrieved is not None
        assert retrieved.name == workflow.name


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestWorkflowErrorHandling:
    """Test error handling in workflows."""

    def test_invalid_workflow_structure(self):
        """Test handling invalid workflow structure."""
        with pytest.raises((ValueError, TypeError)):
            Workflow(
                id=None,  # Invalid
                name="",  # Invalid
                steps=None,  # Invalid
            )

    def test_step_execution_error(self):
        """Test handling step execution errors."""
        mock_conn = MagicMock()
        runner = WorkflowRunner(mock_conn)
        
        workflow = Workflow(
            id="wf",
            name="Error Test",
            steps=[WorkflowStep(id="s", name="S", server="s", tool="t", arguments={})],
        )
        
        with patch.object(runner, '_execute_step') as mock_exec:
            mock_exec.side_effect = Exception("Execution error")
            
            result = runner.run(workflow)
            
            assert result.status == WorkflowStatus.FAILED

    def test_registry_connection_error(self):
        """Test handling registry connection errors."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("Database error")
        
        registry = WorkflowRegistry(mock_conn)
        workflow = Workflow(id="wf", name="Test", steps=[])
        
        with pytest.raises(Exception):
            registry.register(workflow)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
