"""
Export Module for MCP Tool Documentation.

Provides multi-format export capabilities for MCP tools, catalogs,
and workflows in JSON, Markdown, and XML formats.
"""

import json
from typing import Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field

from .prompt_builder import ToolDefinition, Pipeline, PipelineStep, ToolParameter


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    XML = "xml"


class ExportMetadata(BaseModel):
    """Metadata for exported content."""
    
    exported_at: datetime = Field(default_factory=datetime.now)
    format: ExportFormat = Field(description="Export format used")
    version: str = Field(default="1.0", description="Export schema version")
    source: str = Field(default="mcp-manager", description="Source system")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exported_at": self.exported_at.isoformat(),
            "format": self.format.value,
            "version": self.version,
            "source": self.source,
        }


class Exporter:
    """
    Multi-format exporter for MCP tools and workflows.
    
    Supports exporting to:
    - JSON: Full metadata and schema for programmatic use
    - Markdown: Human-readable documentation
    - XML: Structured format for LLM prompts
    """
    
    def __init__(self, include_metadata: bool = True):
        """
        Initialize the exporter.
        
        Args:
            include_metadata: Whether to include export metadata
        """
        self.include_metadata = include_metadata
    
    def export_tool(
        self,
        tool: ToolDefinition,
        format: ExportFormat,
        include_examples: bool = True,
    ) -> str:
        """
        Export a single tool definition.
        
        Args:
            tool: Tool definition to export
            format: Export format
            include_examples: Whether to include usage examples
            
        Returns:
            Exported tool in specified format
        """
        if format == ExportFormat.JSON:
            return self._export_tool_json(tool, include_examples)
        elif format == ExportFormat.MARKDOWN:
            return self._export_tool_markdown(tool, include_examples)
        elif format == ExportFormat.XML:
            return self._export_tool_xml(tool, include_examples)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_catalog(
        self,
        tools: list[ToolDefinition],
        format: ExportFormat,
        title: str = "MCP Tool Catalog",
        group_by_server: bool = True,
    ) -> str:
        """
        Export a catalog of tools.
        
        Args:
            tools: List of tools to export
            format: Export format
            title: Catalog title
            group_by_server: Whether to group tools by server
            
        Returns:
            Exported catalog in specified format
        """
        if format == ExportFormat.JSON:
            return self._export_catalog_json(tools, title, group_by_server)
        elif format == ExportFormat.MARKDOWN:
            return self._export_catalog_markdown(tools, title, group_by_server)
        elif format == ExportFormat.XML:
            return self._export_catalog_xml(tools, title, group_by_server)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_workflow(
        self,
        pipeline: Pipeline,
        format: ExportFormat,
        include_tool_details: bool = True,
    ) -> str:
        """
        Export a workflow pipeline.
        
        Args:
            pipeline: Pipeline to export
            format: Export format
            include_tool_details: Whether to include detailed tool info
            
        Returns:
            Exported pipeline in specified format
        """
        if format == ExportFormat.JSON:
            return self._export_workflow_json(pipeline, include_tool_details)
        elif format == ExportFormat.MARKDOWN:
            return self._export_workflow_markdown(pipeline, include_tool_details)
        elif format == ExportFormat.XML:
            return self._export_workflow_xml(pipeline, include_tool_details)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # === JSON Export Methods ===
    
    def _export_tool_json(
        self,
        tool: ToolDefinition,
        include_examples: bool = True,
    ) -> str:
        """Export tool as JSON with full metadata."""
        data: dict[str, Any] = {
            "server_name": tool.server_name,
            "tool_name": tool.tool_name,
            "full_name": tool.full_name,
            "description": tool.description,
            "parameters": [self._param_to_dict(p) for p in tool.parameters],
            "required_parameters": [p.name for p in tool.required_params],
            "optional_parameters": [p.name for p in tool.optional_params],
            "tags": tool.tags,
        }
        
        if include_examples and tool.examples:
            data["examples"] = tool.examples
        
        if self.include_metadata:
            data["_metadata"] = ExportMetadata(format=ExportFormat.JSON).to_dict()
        
        return json.dumps(data, indent=2, default=str)
    
    def _export_catalog_json(
        self,
        tools: list[ToolDefinition],
        title: str,
        group_by_server: bool,
    ) -> str:
        """Export catalog as JSON."""
        data: dict[str, Any] = {
            "title": title,
            "total_tools": len(tools),
            "servers": list(set(t.server_name for t in tools)),
        }
        
        if group_by_server:
            servers_data: dict[str, list[dict]] = {}
            for tool in tools:
                if tool.server_name not in servers_data:
                    servers_data[tool.server_name] = []
                servers_data[tool.server_name].append({
                    "tool_name": tool.tool_name,
                    "description": tool.description,
                    "parameters_count": len(tool.parameters),
                    "required_params": [p.name for p in tool.required_params],
                    "tags": tool.tags,
                })
            data["tools_by_server"] = servers_data
        else:
            data["tools"] = [
                {
                    "server_name": t.server_name,
                    "tool_name": t.tool_name,
                    "description": t.description,
                    "parameters_count": len(t.parameters),
                }
                for t in tools
            ]
        
        if self.include_metadata:
            data["_metadata"] = ExportMetadata(format=ExportFormat.JSON).to_dict()
        
        return json.dumps(data, indent=2, default=str)
    
    def _export_workflow_json(
        self,
        pipeline: Pipeline,
        include_tool_details: bool,
    ) -> str:
        """Export workflow as JSON."""
        data: dict[str, Any] = {
            "name": pipeline.name,
            "description": pipeline.description,
            "variables": pipeline.variables,
            "steps": [],
            "execution_order": pipeline.step_order,
        }
        
        for step in pipeline.steps:
            step_data: dict[str, Any] = {
                "step_id": step.step_id,
                "tool": step.tool.full_name,
                "arguments": step.arguments,
                "depends_on": step.depends_on,
                "output_mapping": step.output_mapping,
                "condition": step.condition,
                "description": step.description,
            }
            
            if include_tool_details:
                step_data["tool_details"] = {
                    "description": step.tool.description,
                    "parameters": [self._param_to_dict(p) for p in step.tool.parameters],
                }
            
            data["steps"].append(step_data)
        
        if self.include_metadata:
            data["_metadata"] = ExportMetadata(format=ExportFormat.JSON).to_dict()
        
        return json.dumps(data, indent=2, default=str)
    
    # === Markdown Export Methods ===
    
    def _export_tool_markdown(
        self,
        tool: ToolDefinition,
        include_examples: bool = True,
    ) -> str:
        """Export tool as human-readable Markdown."""
        lines: list[str] = []
        
        lines.append(f"# {tool.tool_name}")
        lines.append("")
        lines.append(f"**Server:** `{tool.server_name}`")
        lines.append(f"**Full Name:** `{tool.full_name}`")
        lines.append("")
        
        if tool.description:
            lines.append("## Description")
            lines.append("")
            lines.append(tool.description)
            lines.append("")
        
        if tool.parameters:
            lines.append("## Parameters")
            lines.append("")
            
            if tool.required_params:
                lines.append("### Required")
                lines.append("")
                lines.append("| Name | Type | Description |")
                lines.append("|------|------|-------------|")
                for param in tool.required_params:
                    desc = param.description or "-"
                    lines.append(f"| `{param.name}` | `{param.type}` | {desc} |")
                lines.append("")
            
            if tool.optional_params:
                lines.append("### Optional")
                lines.append("")
                lines.append("| Name | Type | Default | Description |")
                lines.append("|------|------|---------|-------------|")
                for param in tool.optional_params:
                    desc = param.description or "-"
                    default = str(param.default) if param.default is not None else "-"
                    lines.append(f"| `{param.name}` | `{param.type}` | {default} | {desc} |")
                lines.append("")
        
        if tool.tags:
            lines.append("## Tags")
            lines.append("")
            lines.append(", ".join(f"`{tag}`" for tag in tool.tags))
            lines.append("")
        
        if include_examples and tool.examples:
            lines.append("## Examples")
            lines.append("")
            for i, example in enumerate(tool.examples, 1):
                lines.append(f"### Example {i}")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(example, indent=2))
                lines.append("```")
                lines.append("")
        
        lines.append("## Usage")
        lines.append("")
        lines.append("```bash")
        if tool.required_params:
            example_args = {p.name: f"<{p.name}>" for p in tool.required_params}
            lines.append(f"mcp-man call {tool.server_name} {tool.tool_name} '{json.dumps(example_args)}'")
        else:
            lines.append(f"mcp-man call {tool.server_name} {tool.tool_name} '{{}}'")
        lines.append("```")
        lines.append("")
        
        if self.include_metadata:
            lines.append("---")
            lines.append(f"*Exported: {datetime.now().isoformat()}*")
        
        return "\n".join(lines)
    
    def _export_catalog_markdown(
        self,
        tools: list[ToolDefinition],
        title: str,
        group_by_server: bool,
    ) -> str:
        """Export catalog as Markdown documentation."""
        lines: list[str] = []
        
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Total Tools:** {len(tools)}")
        lines.append(f"**Servers:** {len(set(t.server_name for t in tools))}")
        lines.append("")
        
        if group_by_server:
            servers: dict[str, list[ToolDefinition]] = {}
            for tool in tools:
                if tool.server_name not in servers:
                    servers[tool.server_name] = []
                servers[tool.server_name].append(tool)
            
            lines.append("## Table of Contents")
            lines.append("")
            for server_name in sorted(servers.keys()):
                anchor = server_name.lower().replace(" ", "-")
                lines.append(f"- [{server_name}](#{anchor}) ({len(servers[server_name])} tools)")
            lines.append("")
            
            for server_name in sorted(servers.keys()):
                server_tools = servers[server_name]
                lines.append(f"## {server_name}")
                lines.append("")
                lines.append(f"*{len(server_tools)} tools available*")
                lines.append("")
                
                lines.append("| Tool | Description | Parameters |")
                lines.append("|------|-------------|------------|")
                
                for tool in sorted(server_tools, key=lambda t: t.tool_name):
                    desc = tool.description[:60] + "..." if tool.description and len(tool.description) > 60 else (tool.description or "-")
                    params = len(tool.parameters)
                    required = len(tool.required_params)
                    lines.append(f"| `{tool.tool_name}` | {desc} | {params} ({required} required) |")
                
                lines.append("")
        else:
            lines.append("## All Tools")
            lines.append("")
            lines.append("| Server | Tool | Description |")
            lines.append("|--------|------|-------------|")
            
            for tool in sorted(tools, key=lambda t: (t.server_name, t.tool_name)):
                desc = tool.description[:50] + "..." if tool.description and len(tool.description) > 50 else (tool.description or "-")
                lines.append(f"| `{tool.server_name}` | `{tool.tool_name}` | {desc} |")
            
            lines.append("")
        
        lines.append("## Quick Reference")
        lines.append("")
        lines.append("```bash")
        lines.append("# Search for tools")
        lines.append('mcp-man search "your query"')
        lines.append("")
        lines.append("# Inspect a tool")
        lines.append("mcp-man inspect <server> <tool>")
        lines.append("")
        lines.append("# Execute a tool")
        lines.append("mcp-man call <server> <tool> '{\"param\": \"value\"}'")
        lines.append("```")
        lines.append("")
        
        if self.include_metadata:
            lines.append("---")
            lines.append(f"*Generated: {datetime.now().isoformat()}*")
        
        return "\n".join(lines)
    
    def _export_workflow_markdown(
        self,
        pipeline: Pipeline,
        include_tool_details: bool,
    ) -> str:
        """Export workflow as Markdown documentation."""
        lines: list[str] = []
        
        lines.append(f"# Pipeline: {pipeline.name}")
        lines.append("")
        
        if pipeline.description:
            lines.append(pipeline.description)
            lines.append("")
        
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- **Steps:** {len(pipeline.steps)}")
        lines.append(f"- **Execution Order:** {' -> '.join(pipeline.step_order)}")
        lines.append("")
        
        if pipeline.variables:
            lines.append("## Variables")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(pipeline.variables, indent=2))
            lines.append("```")
            lines.append("")
        
        lines.append("## Steps")
        lines.append("")
        
        for i, step_id in enumerate(pipeline.step_order, 1):
            step = next((s for s in pipeline.steps if s.step_id == step_id), None)
            if not step:
                continue
            
            lines.append(f"### {i}. {step.step_id}")
            lines.append("")
            
            if step.description:
                lines.append(f"*{step.description}*")
                lines.append("")
            
            lines.append(f"**Tool:** `{step.tool.full_name}`")
            
            if step.depends_on:
                lines.append(f"**Depends on:** {', '.join(f'`{d}`' for d in step.depends_on)}")
            
            if step.condition:
                lines.append(f"**Condition:** `{step.condition}`")
            
            lines.append("")
            
            if step.arguments:
                lines.append("**Arguments:**")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(step.arguments, indent=2))
                lines.append("```")
                lines.append("")
            
            if step.output_mapping:
                lines.append("**Output Mapping:**")
                lines.append("")
                for output, variable in step.output_mapping.items():
                    lines.append(f"- `{output}` -> `${variable}`")
                lines.append("")
            
            if include_tool_details and step.tool.description:
                lines.append(f"> **Tool Description:** {step.tool.description}")
                lines.append("")
        
        lines.append("## Execution Command")
        lines.append("")
        lines.append("```bash")
        lines.append(f"# Execute pipeline: {pipeline.name}")
        for step in pipeline.steps:
            args_str = json.dumps(step.arguments) if step.arguments else "{}"
            lines.append(f"mcp-man call {step.tool.server_name} {step.tool.tool_name} '{args_str}'")
        lines.append("```")
        lines.append("")
        
        if self.include_metadata:
            lines.append("---")
            lines.append(f"*Generated: {datetime.now().isoformat()}*")
        
        return "\n".join(lines)
    
    # === XML Export Methods ===
    
    def _export_tool_xml(
        self,
        tool: ToolDefinition,
        include_examples: bool = True,
    ) -> str:
        """Export tool as structured XML for LLM prompts."""
        lines: list[str] = []
        
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(f'<tool server="{tool.server_name}" name="{tool.tool_name}">')
        
        if tool.description:
            lines.append(f"  <description>{self._escape_xml(tool.description)}</description>")
        
        if tool.parameters:
            lines.append("  <parameters>")
            for param in tool.parameters:
                required = "true" if param.required else "false"
                lines.append(f'    <parameter name="{param.name}" type="{param.type}" required="{required}">')
                if param.description:
                    lines.append(f"      <description>{self._escape_xml(param.description)}</description>")
                if param.default is not None:
                    lines.append(f"      <default>{self._escape_xml(str(param.default))}</default>")
                lines.append("    </parameter>")
            lines.append("  </parameters>")
        
        if tool.tags:
            lines.append("  <tags>")
            for tag in tool.tags:
                lines.append(f"    <tag>{self._escape_xml(tag)}</tag>")
            lines.append("  </tags>")
        
        if include_examples and tool.examples:
            lines.append("  <examples>")
            for example in tool.examples:
                lines.append("    <example>")
                lines.append(f"      <![CDATA[{json.dumps(example, indent=2)}]]>")
                lines.append("    </example>")
            lines.append("  </examples>")
        
        if self.include_metadata:
            lines.append("  <metadata>")
            lines.append(f"    <exported_at>{datetime.now().isoformat()}</exported_at>")
            lines.append("    <format>xml</format>")
            lines.append("    <source>mcp-manager</source>")
            lines.append("  </metadata>")
        
        lines.append("</tool>")
        
        return "\n".join(lines)
    
    def _export_catalog_xml(
        self,
        tools: list[ToolDefinition],
        title: str,
        group_by_server: bool,
    ) -> str:
        """Export catalog as structured XML."""
        lines: list[str] = []
        
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(f'<catalog title="{self._escape_xml(title)}" total_tools="{len(tools)}">')
        
        if group_by_server:
            servers: dict[str, list[ToolDefinition]] = {}
            for tool in tools:
                if tool.server_name not in servers:
                    servers[tool.server_name] = []
                servers[tool.server_name].append(tool)
            
            for server_name in sorted(servers.keys()):
                server_tools = servers[server_name]
                lines.append(f'  <server name="{self._escape_xml(server_name)}" tool_count="{len(server_tools)}">')
                
                for tool in sorted(server_tools, key=lambda t: t.tool_name):
                    lines.append(f'    <tool name="{self._escape_xml(tool.tool_name)}">')
                    if tool.description:
                        lines.append(f"      <description>{self._escape_xml(tool.description)}</description>")
                    lines.append(f"      <parameters_count>{len(tool.parameters)}</parameters_count>")
                    lines.append(f"      <required_count>{len(tool.required_params)}</required_count>")
                    lines.append("    </tool>")
                
                lines.append("  </server>")
        else:
            lines.append("  <tools>")
            for tool in sorted(tools, key=lambda t: (t.server_name, t.tool_name)):
                lines.append(f'    <tool server="{self._escape_xml(tool.server_name)}" name="{self._escape_xml(tool.tool_name)}">')
                if tool.description:
                    lines.append(f"      <description>{self._escape_xml(tool.description)}</description>")
                lines.append("    </tool>")
            lines.append("  </tools>")
        
        if self.include_metadata:
            lines.append("  <metadata>")
            lines.append(f"    <exported_at>{datetime.now().isoformat()}</exported_at>")
            lines.append("    <format>xml</format>")
            lines.append("    <source>mcp-manager</source>")
            lines.append("  </metadata>")
        
        lines.append("</catalog>")
        
        return "\n".join(lines)
    
    def _export_workflow_xml(
        self,
        pipeline: Pipeline,
        include_tool_details: bool,
    ) -> str:
        """Export workflow as structured XML."""
        lines: list[str] = []
        
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(f'<pipeline name="{self._escape_xml(pipeline.name)}">')
        
        if pipeline.description:
            lines.append(f"  <description>{self._escape_xml(pipeline.description)}</description>")
        
        if pipeline.variables:
            lines.append("  <variables>")
            lines.append(f"    <![CDATA[{json.dumps(pipeline.variables, indent=2)}]]>")
            lines.append("  </variables>")
        
        lines.append(f"  <execution_order>{' -> '.join(pipeline.step_order)}</execution_order>")
        
        lines.append("  <steps>")
        for step in pipeline.steps:
            depends = f' depends_on="{",".join(step.depends_on)}"' if step.depends_on else ""
            condition = f' condition="{self._escape_xml(step.condition)}"' if step.condition else ""
            
            lines.append(f'    <step id="{self._escape_xml(step.step_id)}"{depends}{condition}>')
            
            if step.description:
                lines.append(f"      <description>{self._escape_xml(step.description)}</description>")
            
            lines.append(f'      <tool server="{self._escape_xml(step.tool.server_name)}" name="{self._escape_xml(step.tool.tool_name)}">')
            
            if include_tool_details and step.tool.description:
                lines.append(f"        <description>{self._escape_xml(step.tool.description)}</description>")
            
            if include_tool_details and step.tool.parameters:
                lines.append("        <parameters>")
                for param in step.tool.parameters:
                    req = "true" if param.required else "false"
                    lines.append(f'          <parameter name="{param.name}" type="{param.type}" required="{req}"/>')
                lines.append("        </parameters>")
            
            lines.append("      </tool>")
            
            if step.arguments:
                lines.append("      <arguments>")
                lines.append(f"        <![CDATA[{json.dumps(step.arguments, indent=2)}]]>")
                lines.append("      </arguments>")
            
            if step.output_mapping:
                lines.append("      <output_mapping>")
                for output, variable in step.output_mapping.items():
                    lines.append(f'        <mapping output="{self._escape_xml(output)}" variable="{self._escape_xml(variable)}"/>')
                lines.append("      </output_mapping>")
            
            lines.append("    </step>")
        
        lines.append("  </steps>")
        
        if self.include_metadata:
            lines.append("  <metadata>")
            lines.append(f"    <exported_at>{datetime.now().isoformat()}</exported_at>")
            lines.append("    <format>xml</format>")
            lines.append("    <source>mcp-manager</source>")
            lines.append("  </metadata>")
        
        lines.append("</pipeline>")
        
        return "\n".join(lines)
    
    # === Helper Methods ===
    
    def _param_to_dict(self, param: ToolParameter) -> dict[str, Any]:
        """Convert parameter to dictionary."""
        return {
            "name": param.name,
            "type": param.type,
            "description": param.description,
            "required": param.required,
            "default": param.default,
        }
    
    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        if not text:
            return ""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )


def export_tools_to_file(
    tools: list[ToolDefinition],
    filepath: str,
    format: ExportFormat,
    **kwargs,
) -> bool:
    """
    Export tools to a file.
    
    Args:
        tools: List of tools to export
        filepath: Output file path
        format: Export format
        **kwargs: Additional arguments for export_catalog
        
    Returns:
        True if successful, False otherwise
    """
    try:
        exporter = Exporter()
        content = exporter.export_catalog(tools, format, **kwargs)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"Error exporting to {filepath}: {e}")
        return False


def export_workflow_to_file(
    pipeline: Pipeline,
    filepath: str,
    format: ExportFormat,
    **kwargs,
) -> bool:
    """
    Export workflow to a file.
    
    Args:
        pipeline: Pipeline to export
        filepath: Output file path
        format: Export format
        **kwargs: Additional arguments for export_workflow
        
    Returns:
        True if successful, False otherwise
    """
    try:
        exporter = Exporter()
        content = exporter.export_workflow(pipeline, format, **kwargs)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"Error exporting to {filepath}: {e}")
        return False
