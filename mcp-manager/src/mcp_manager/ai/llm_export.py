"""Export MCP tools for LLM consumption in various formats."""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..tools.schema import ToolSchema
from .examples import ExampleGenerator


class ExportFormat(str, Enum):
    """Supported export formats."""
    
    JSON = "json"
    MARKDOWN = "markdown"
    MD = "md"
    XML = "xml"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    YAML = "yaml"


@dataclass
class ExportConfig:
    """Configuration for export."""
    
    include_examples: bool = False
    include_schema: bool = True
    include_descriptions: bool = True
    max_description_length: int = 500
    group_by_server: bool = True
    compact: bool = False


@dataclass 
class ToolDocumentation:
    """Documentation for a single tool."""
    
    name: str
    description: str
    server: str = ""
    parameters: list[dict[str, Any]] = field(default_factory=list)
    required_params: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    returns: Optional[str] = None

    @classmethod
    def from_tool(cls, tool: ToolSchema) -> "ToolDocumentation":
        """Create documentation from tool schema.
        
        Args:
            tool: Tool schema
            
        Returns:
            ToolDocumentation instance
        """
        parameters = []
        
        if tool.input_schema and "properties" in tool.input_schema:
            for param_name, param_schema in tool.input_schema["properties"].items():
                param_doc = {
                    "name": param_name,
                    "type": param_schema.get("type", "any"),
                    "description": param_schema.get("description", ""),
                    "required": param_name in (tool.required_params or []),
                }
                
                if "default" in param_schema:
                    param_doc["default"] = param_schema["default"]
                if "enum" in param_schema:
                    param_doc["enum"] = param_schema["enum"]
                
                parameters.append(param_doc)
        
        return cls(
            name=tool.tool_name,
            description=tool.description or "",
            server=tool.server_name,
            parameters=parameters,
            required_params=tool.required_params or [],
        )

    def to_markdown(self) -> str:
        """Convert to markdown format.
        
        Returns:
            Markdown string
        """
        lines = []
        lines.append(f"### {self.name}")
        lines.append("")
        
        if self.description:
            lines.append(self.description)
            lines.append("")
        
        if self.server:
            lines.append(f"**Server:** `{self.server}`")
            lines.append("")
        
        if self.parameters:
            lines.append("**Parameters:**")
            lines.append("")
            for param in self.parameters:
                required = " (required)" if param.get("required") else ""
                default = f", default: `{param['default']}`" if "default" in param else ""
                desc = f" - {param['description']}" if param.get("description") else ""
                lines.append(f"- `{param['name']}` ({param['type']}{required}{default}){desc}")
            lines.append("")
        
        if self.examples:
            lines.append("**Examples:**")
            lines.append("")
            for example in self.examples:
                lines.append("```json")
                lines.append(json.dumps(example, indent=2))
                lines.append("```")
                lines.append("")
        
        return "\n".join(lines)


class LLMExporter:
    """Export MCP tools for LLM consumption."""

    def __init__(self) -> None:
        """Initialize exporter."""
        self.example_generator = ExampleGenerator()

    def export(
        self,
        tools: list[ToolSchema],
        format: ExportFormat | str,
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Export tools in specified format.
        
        Args:
            tools: List of tools to export
            format: Export format
            config: Export configuration
            
        Returns:
            Exported string
        """
        if isinstance(format, str):
            try:
                format = ExportFormat(format.lower())
            except ValueError:
                raise ValueError(f"Unknown export format: {format}")

        config = config or ExportConfig()

        if format == ExportFormat.JSON:
            return self._export_json(tools, config)
        elif format in [ExportFormat.MARKDOWN, ExportFormat.MD]:
            return self._export_markdown(tools, config)
        elif format == ExportFormat.XML:
            return self._export_xml(tools, config)
        elif format == ExportFormat.OPENAI:
            return self._export_openai(tools, config)
        elif format == ExportFormat.ANTHROPIC:
            return self._export_anthropic(tools, config)
        elif format == ExportFormat.YAML:
            return self._export_yaml(tools, config)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_tool(
        self,
        tool: ToolSchema,
        format: ExportFormat | str,
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Export a single tool.
        
        Args:
            tool: Tool to export
            format: Export format
            config: Export configuration
            
        Returns:
            Exported string
        """
        return self.export([tool], format, config)

    def export_to_file(
        self,
        tools: list[ToolSchema],
        filepath: str | Path,
        format: ExportFormat | str,
        config: Optional[ExportConfig] = None,
    ) -> None:
        """Export tools to a file.
        
        Args:
            tools: Tools to export
            filepath: Output file path
            format: Export format
            config: Export configuration
        """
        content = self.export(tools, format, config)
        Path(filepath).write_text(content, encoding="utf-8")

    def _export_json(self, tools: list[ToolSchema], config: ExportConfig) -> str:
        """Export to JSON format."""
        tool_list = []
        
        for tool in tools:
            tool_dict: dict[str, Any] = {
                "name": tool.tool_name,
                "server": tool.server_name,
            }
            
            if config.include_descriptions and tool.description:
                desc = tool.description
                if len(desc) > config.max_description_length:
                    desc = desc[:config.max_description_length] + "..."
                tool_dict["description"] = desc
            
            if config.include_schema and tool.input_schema:
                tool_dict["parameters"] = tool.input_schema
            
            if tool.required_params:
                tool_dict["required"] = tool.required_params
            
            if config.include_examples:
                example = self.example_generator.generate(tool)
                tool_dict["example"] = example.arguments
            
            tool_list.append(tool_dict)
        
        if config.group_by_server:
            grouped: dict[str, list] = {}
            for tool_dict in tool_list:
                server = tool_dict.pop("server", "default")
                if server not in grouped:
                    grouped[server] = []
                grouped[server].append(tool_dict)
            output = {"servers": grouped}
        else:
            output = {"tools": tool_list}
        
        indent = None if config.compact else 2
        return json.dumps(output, indent=indent)

    def _export_markdown(self, tools: list[ToolSchema], config: ExportConfig) -> str:
        """Export to Markdown format."""
        lines = ["# MCP Tools Documentation", ""]
        
        if config.group_by_server:
            # Group by server
            servers: dict[str, list[ToolSchema]] = {}
            for tool in tools:
                if tool.server_name not in servers:
                    servers[tool.server_name] = []
                servers[tool.server_name].append(tool)
            
            for server_name, server_tools in sorted(servers.items()):
                lines.append(f"## {server_name}")
                lines.append("")
                lines.append(f"*{len(server_tools)} tools available*")
                lines.append("")
                
                for tool in server_tools:
                    doc = ToolDocumentation.from_tool(tool)
                    
                    if config.include_examples:
                        example = self.example_generator.generate(tool)
                        doc.examples.append(example.arguments)
                    
                    lines.append(doc.to_markdown())
        else:
            for tool in tools:
                doc = ToolDocumentation.from_tool(tool)
                
                if config.include_examples:
                    example = self.example_generator.generate(tool)
                    doc.examples.append(example.arguments)
                
                lines.append(doc.to_markdown())
        
        return "\n".join(lines)

    def _export_xml(self, tools: list[ToolSchema], config: ExportConfig) -> str:
        """Export to XML format."""
        root = ET.Element("tools")
        
        for tool in tools:
            tool_elem = ET.SubElement(root, "tool")
            tool_elem.set("name", tool.tool_name)
            tool_elem.set("server", tool.server_name)
            
            if config.include_descriptions and tool.description:
                desc_elem = ET.SubElement(tool_elem, "description")
                desc = tool.description
                if len(desc) > config.max_description_length:
                    desc = desc[:config.max_description_length] + "..."
                desc_elem.text = desc
            
            if config.include_schema and tool.input_schema:
                params_elem = ET.SubElement(tool_elem, "parameters")
                properties = tool.input_schema.get("properties", {})
                
                for param_name, param_schema in properties.items():
                    param_elem = ET.SubElement(params_elem, "parameter")
                    param_elem.set("name", param_name)
                    param_elem.set("type", param_schema.get("type", "any"))
                    
                    if param_name in (tool.required_params or []):
                        param_elem.set("required", "true")
                    
                    if "description" in param_schema:
                        param_desc = ET.SubElement(param_elem, "description")
                        param_desc.text = param_schema["description"]
            
            if config.include_examples:
                example = self.example_generator.generate(tool)
                example_elem = ET.SubElement(tool_elem, "example")
                example_elem.text = json.dumps(example.arguments)
        
        return ET.tostring(root, encoding="unicode", method="xml")

    def _export_openai(self, tools: list[ToolSchema], config: ExportConfig) -> str:
        """Export to OpenAI function calling format."""
        functions = []
        
        for tool in tools:
            func: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.tool_name,
                    "description": tool.description or f"Execute {tool.tool_name}",
                }
            }
            
            # Build parameters schema
            if tool.input_schema:
                parameters = {
                    "type": "object",
                    "properties": tool.input_schema.get("properties", {}),
                    "required": tool.required_params or [],
                }
            else:
                parameters = {"type": "object", "properties": {}}
            
            func["function"]["parameters"] = parameters
            functions.append(func)
        
        return json.dumps(functions, indent=2)

    def _export_anthropic(self, tools: list[ToolSchema], config: ExportConfig) -> str:
        """Export to Anthropic tool use format."""
        tools_list = []
        
        for tool in tools:
            tool_def: dict[str, Any] = {
                "name": tool.tool_name,
                "description": tool.description or f"Execute {tool.tool_name}",
            }
            
            # Build input schema
            if tool.input_schema:
                tool_def["input_schema"] = {
                    "type": "object",
                    "properties": tool.input_schema.get("properties", {}),
                    "required": tool.required_params or [],
                }
            else:
                tool_def["input_schema"] = {"type": "object", "properties": {}}
            
            tools_list.append(tool_def)
        
        return json.dumps(tools_list, indent=2)

    def _export_yaml(self, tools: list[ToolSchema], config: ExportConfig) -> str:
        """Export to YAML format (simple implementation without PyYAML)."""
        lines = ["tools:"]
        
        for tool in tools:
            lines.append(f"  - name: {tool.tool_name}")
            lines.append(f"    server: {tool.server_name}")
            
            if config.include_descriptions and tool.description:
                # Escape and truncate description
                desc = tool.description.replace("\n", " ")
                if len(desc) > config.max_description_length:
                    desc = desc[:config.max_description_length] + "..."
                lines.append(f"    description: \"{desc}\"")
            
            if tool.required_params:
                lines.append("    required:")
                for param in tool.required_params:
                    lines.append(f"      - {param}")
            
            if config.include_schema and tool.input_schema:
                properties = tool.input_schema.get("properties", {})
                if properties:
                    lines.append("    parameters:")
                    for param_name, param_schema in properties.items():
                        lines.append(f"      {param_name}:")
                        lines.append(f"        type: {param_schema.get('type', 'any')}")
                        if "description" in param_schema:
                            lines.append(f"        description: \"{param_schema['description']}\"")
        
        return "\n".join(lines)
