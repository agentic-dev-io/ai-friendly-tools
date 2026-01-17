"""Web Intelligence Suite - Standalone web research tool."""

from .core import Web, WebConfig
from .workflows import (
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowStep,
    StepType,
    create_unreal_remote_control_workflow,
)
from .autolearn import AutoLearner

__all__ = [
    "Web",
    "WebConfig",
    "WorkflowDefinition",
    "WorkflowEngine",
    "WorkflowStep",
    "StepType",
    "create_unreal_remote_control_workflow",
    "AutoLearner",
]
__version__ = "0.1.0"
