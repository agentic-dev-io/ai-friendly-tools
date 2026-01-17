"""
MCP Agent Support - Documentation and Integration for AI Agents

Provides tools for generating AGENT.md documentation for Claude,
ChatGPT, and other AI agents using MCP-Man.
"""

from .templates import (
    AgentMarkdownGenerator,
    ServerInfo,
    ToolInfo,
    create_agent_md_from_registry,
    quick_agent_md,
)

__all__ = [
    "AgentMarkdownGenerator",
    "ToolInfo",
    "ServerInfo",
    "create_agent_md_from_registry",
    "quick_agent_md",
]
