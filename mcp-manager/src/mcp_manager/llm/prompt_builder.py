"""
Prompt Builder for LLM Integration.

Provides Claude-optimized prompt templates for tool selection, execution,
and workflow orchestration with MCP tools.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class PromptStyle(str, Enum):
    """Prompt formatting style."""
    CLAUDE = "claude"
    GENERIC = "generic"
    STRUCTURED = "structured"


class ToolParameter(BaseModel):
    """Schema for a tool parameter."""
    
    name: str = Field(description="Parameter name")
    type: str = Field(default="string", description="Parameter type")
    description: Optional[str] = Field(default=None, description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if any")
    
    def to_prompt_line(self) -> str:
        """Format parameter for prompt display."""
        req = " (required)" if self.required else ""
        default = f" [default: {self.default}]" if self.default is not None else ""
        desc = f": {self.description}" if self.description else ""
        return f"  - {self.name} ({self.type}){req}{default}{desc}"


class ToolDefinition(BaseModel):
    """Definition of a tool for prompt building."""
    
    server_name: str = Field(description="Name of the MCP server")
    tool_name: str = Field(description="Name of the tool")
    description: Optional[str] = Field(default=None, description="Tool description")
    parameters: list[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    examples: list[dict[str, Any]] = Field(default_factory=list, description="Usage examples")
    tags: list[str] = Field(default_factory=list, description="Tool categorization tags")
    
    @property
    def full_name(self) -> str:
        """Get fully qualified tool name."""
        return f"{self.server_name}:{self.tool_name}"
    
    @property
    def required_params(self) -> list[ToolParameter]:
        """Get required parameters only."""
        return [p for p in self.parameters if p.required]
    
    @property
    def optional_params(self) -> list[ToolParameter]:
        """Get optional parameters only."""
        return [p for p in self.parameters if not p.required]


class PipelineStep(BaseModel):
    """A step in a workflow pipeline."""
    
    step_id: str = Field(description="Unique step identifier")
    tool: ToolDefinition = Field(description="Tool to execute")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments to pass")
    depends_on: list[str] = Field(default_factory=list, description="Steps this depends on")
    output_mapping: dict[str, str] = Field(
        default_factory=dict, 
        description="Map output fields to variable names"
    )
    condition: Optional[str] = Field(default=None, description="Conditional execution expression")
    description: Optional[str] = Field(default=None, description="Step description")


class Pipeline(BaseModel):
    """A workflow pipeline of tool executions."""
    
    name: str = Field(description="Pipeline name")
    description: Optional[str] = Field(default=None, description="Pipeline description")
    steps: list[PipelineStep] = Field(default_factory=list, description="Pipeline steps")
    variables: dict[str, Any] = Field(default_factory=dict, description="Initial variables")
    
    @property
    def step_order(self) -> list[str]:
        """Get steps in topologically sorted order based on dependencies."""
        sorted_steps: list[str] = []
        visited: set[str] = set()
        step_map = {s.step_id: s for s in self.steps}
        
        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            visited.add(step_id)
            step = step_map.get(step_id)
            if step:
                for dep in step.depends_on:
                    visit(dep)
                sorted_steps.append(step_id)
        
        for step in self.steps:
            visit(step.step_id)
        
        return sorted_steps


class PromptBuilder:
    """
    Builds optimized prompts for LLM interaction with MCP tools.
    
    Supports Claude-optimized formatting and multiple prompt styles
    for tool selection, execution, and workflow orchestration.
    """
    
    # Claude-optimized prompt templates
    TOOL_SELECTION_TEMPLATE = """You have access to the following MCP tools to help accomplish the user's request.

## Available Tools

{tool_list}

## User Query

{query}

## Instructions

1. Analyze the user's query to understand what they need
2. Select the most appropriate tool(s) from the available options
3. If multiple tools are needed, plan the execution order
4. Consider tool dependencies and data flow between steps

Respond with your tool selection and reasoning. Use the exact tool names as shown above."""

    EXECUTION_TEMPLATE = """Execute the following MCP tool with the specified arguments.

## Tool Information

**Server:** {server_name}
**Tool:** {tool_name}
**Description:** {description}

## Parameters

{parameters}

## Arguments to Use

```json
{arguments}
```

## Expected Behavior

{expected_behavior}

Execute this tool call and report the results."""

    WORKFLOW_TEMPLATE = """Execute the following workflow pipeline of MCP tool calls.

## Pipeline: {pipeline_name}

{pipeline_description}

## Variables

{variables}

## Steps

{steps}

## Execution Instructions

1. Execute steps in dependency order
2. Pass outputs from earlier steps as inputs to later steps
3. Handle any conditional steps appropriately
4. Report results after each step
5. Stop execution if any required step fails

Begin pipeline execution."""

    TOOL_CATALOG_TEMPLATE = """# MCP Tool Catalog

The following tools are available through the MCP gateway.

{catalog}

## Usage Notes

- Use `mcp-man search "query"` to find specific tools
- Use `mcp-man inspect <server> <tool>` for detailed information
- Always verify tool parameters before execution"""

    def __init__(self, style: PromptStyle = PromptStyle.CLAUDE):
        """
        Initialize the prompt builder.
        
        Args:
            style: Prompt formatting style to use
        """
        self.style = style
    
    def build_tool_selection_prompt(
        self,
        tools: list[ToolDefinition],
        query: str,
        max_tools: int = 20,
        include_examples: bool = False,
    ) -> str:
        """
        Build a prompt for tool selection based on a user query.
        
        Args:
            tools: List of available tools
            query: User's query or request
            max_tools: Maximum number of tools to include
            include_examples: Whether to include usage examples
            
        Returns:
            Formatted prompt string for tool selection
        """
        tool_list = self._format_tool_list(tools[:max_tools], include_examples)
        
        prompt = self.TOOL_SELECTION_TEMPLATE.format(
            tool_list=tool_list,
            query=query,
        )
        
        return self.format_for_claude(prompt)
    
    def build_execution_prompt(
        self,
        tool: ToolDefinition,
        args: dict[str, Any],
        include_schema: bool = True,
    ) -> str:
        """
        Build a prompt for tool execution.
        
        Args:
            tool: Tool definition to execute
            args: Arguments to pass to the tool
            include_schema: Whether to include full parameter schema
            
        Returns:
            Formatted prompt string for execution
        """
        parameters = self._format_parameters(tool.parameters) if include_schema else "See tool documentation"
        
        import json
        arguments_json = json.dumps(args, indent=2)
        
        expected = self._generate_expected_behavior(tool)
        
        prompt = self.EXECUTION_TEMPLATE.format(
            server_name=tool.server_name,
            tool_name=tool.tool_name,
            description=tool.description or "No description available",
            parameters=parameters,
            arguments=arguments_json,
            expected_behavior=expected,
        )
        
        return self.format_for_claude(prompt)
    
    def build_workflow_prompt(
        self,
        pipeline: Pipeline,
        include_tool_details: bool = True,
    ) -> str:
        """
        Build a prompt for workflow execution.
        
        Args:
            pipeline: Pipeline definition to execute
            include_tool_details: Whether to include detailed tool info
            
        Returns:
            Formatted prompt string for workflow execution
        """
        import json
        
        variables_str = json.dumps(pipeline.variables, indent=2) if pipeline.variables else "None defined"
        
        steps_str = self._format_pipeline_steps(pipeline.steps, include_tool_details)
        
        prompt = self.WORKFLOW_TEMPLATE.format(
            pipeline_name=pipeline.name,
            pipeline_description=pipeline.description or "No description provided",
            variables=variables_str,
            steps=steps_str,
        )
        
        return self.format_for_claude(prompt)
    
    def build_catalog_prompt(
        self,
        tools: list[ToolDefinition],
        group_by_server: bool = True,
    ) -> str:
        """
        Build a catalog prompt listing all available tools.
        
        Args:
            tools: List of all available tools
            group_by_server: Whether to group tools by server
            
        Returns:
            Formatted catalog prompt
        """
        if group_by_server:
            catalog = self._format_tools_by_server(tools)
        else:
            catalog = self._format_tool_list(tools, include_examples=False)
        
        prompt = self.TOOL_CATALOG_TEMPLATE.format(catalog=catalog)
        
        return self.format_for_claude(prompt)
    
    def format_for_claude(self, content: str) -> str:
        """
        Format content optimized for Claude's processing.
        
        Applies Claude-specific optimizations:
        - Uses XML-style tags for structure
        - Applies clear section headers
        - Optimizes whitespace for readability
        
        Args:
            content: Raw content to format
            
        Returns:
            Claude-optimized formatted content
        """
        if self.style == PromptStyle.CLAUDE:
            return self._apply_claude_formatting(content)
        elif self.style == PromptStyle.STRUCTURED:
            return self._apply_structured_formatting(content)
        else:
            return content
    
    def _format_tool_list(
        self,
        tools: list[ToolDefinition],
        include_examples: bool = False,
    ) -> str:
        """Format a list of tools for prompt inclusion."""
        lines: list[str] = []
        
        for tool in tools:
            lines.append(f"### {tool.full_name}")
            lines.append("")
            
            if tool.description:
                lines.append(f"**Description:** {tool.description}")
                lines.append("")
            
            if tool.parameters:
                lines.append("**Parameters:**")
                for param in tool.parameters:
                    lines.append(param.to_prompt_line())
                lines.append("")
            
            if tool.tags:
                lines.append(f"**Tags:** {', '.join(tool.tags)}")
                lines.append("")
            
            if include_examples and tool.examples:
                lines.append("**Example:**")
                import json
                lines.append(f"```json\n{json.dumps(tool.examples[0], indent=2)}\n```")
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_tools_by_server(self, tools: list[ToolDefinition]) -> str:
        """Format tools grouped by server."""
        servers: dict[str, list[ToolDefinition]] = {}
        
        for tool in tools:
            if tool.server_name not in servers:
                servers[tool.server_name] = []
            servers[tool.server_name].append(tool)
        
        lines: list[str] = []
        
        for server_name in sorted(servers.keys()):
            server_tools = servers[server_name]
            lines.append(f"## Server: {server_name}")
            lines.append(f"*{len(server_tools)} tools available*")
            lines.append("")
            
            for tool in server_tools:
                desc = f" - {tool.description}" if tool.description else ""
                params = f" ({len(tool.parameters)} params)" if tool.parameters else ""
                lines.append(f"- **{tool.tool_name}**{params}{desc}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_parameters(self, parameters: list[ToolParameter]) -> str:
        """Format parameters for prompt display."""
        if not parameters:
            return "No parameters required."
        
        lines: list[str] = []
        
        required = [p for p in parameters if p.required]
        optional = [p for p in parameters if not p.required]
        
        if required:
            lines.append("**Required:**")
            for param in required:
                lines.append(param.to_prompt_line())
            lines.append("")
        
        if optional:
            lines.append("**Optional:**")
            for param in optional:
                lines.append(param.to_prompt_line())
        
        return "\n".join(lines)
    
    def _format_pipeline_steps(
        self,
        steps: list[PipelineStep],
        include_tool_details: bool = True,
    ) -> str:
        """Format pipeline steps for prompt display."""
        import json
        lines: list[str] = []
        
        for i, step in enumerate(steps, 1):
            lines.append(f"### Step {i}: {step.step_id}")
            lines.append("")
            
            if step.description:
                lines.append(f"*{step.description}*")
                lines.append("")
            
            lines.append(f"**Tool:** {step.tool.full_name}")
            
            if step.depends_on:
                lines.append(f"**Depends on:** {', '.join(step.depends_on)}")
            
            if step.condition:
                lines.append(f"**Condition:** {step.condition}")
            
            if step.arguments:
                lines.append(f"**Arguments:**")
                lines.append(f"```json\n{json.dumps(step.arguments, indent=2)}\n```")
            
            if step.output_mapping:
                lines.append(f"**Output mapping:** {step.output_mapping}")
            
            if include_tool_details and step.tool.parameters:
                lines.append("")
                lines.append("**Tool parameters:**")
                for param in step.tool.parameters:
                    lines.append(param.to_prompt_line())
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_expected_behavior(self, tool: ToolDefinition) -> str:
        """Generate expected behavior description for a tool."""
        lines: list[str] = []
        
        if tool.description:
            lines.append(f"This tool should {tool.description.lower()}")
        else:
            lines.append("Execute the tool with the provided arguments.")
        
        if tool.required_params:
            param_names = [p.name for p in tool.required_params]
            lines.append(f"Required inputs: {', '.join(param_names)}")
        
        return "\n".join(lines)
    
    def _apply_claude_formatting(self, content: str) -> str:
        """Apply Claude-specific formatting optimizations."""
        # Add clear section breaks
        sections = content.split("\n## ")
        if len(sections) > 1:
            formatted_sections = [sections[0]]
            for section in sections[1:]:
                formatted_sections.append(f"## {section}")
            content = "\n\n".join(formatted_sections)
        
        # Ensure proper spacing around code blocks
        content = content.replace("```\n\n", "```\n")
        content = content.replace("\n\n```", "\n```")
        
        # Clean up excessive whitespace while maintaining readability
        lines = content.split("\n")
        cleaned_lines: list[str] = []
        prev_blank = False
        
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line)
            prev_blank = is_blank
        
        return "\n".join(cleaned_lines)
    
    def _apply_structured_formatting(self, content: str) -> str:
        """Apply structured XML-style formatting."""
        # Wrap sections in XML-style tags for structured parsing
        lines = content.split("\n")
        result_lines: list[str] = []
        in_section = False
        section_name = ""
        
        for line in lines:
            if line.startswith("## "):
                if in_section:
                    result_lines.append(f"</section>")
                section_name = line[3:].strip().lower().replace(" ", "_")
                result_lines.append(f"<section name=\"{section_name}\">")
                result_lines.append(line)
                in_section = True
            else:
                result_lines.append(line)
        
        if in_section:
            result_lines.append("</section>")
        
        return "\n".join(result_lines)


def tool_from_schema(
    server_name: str,
    tool_name: str,
    description: Optional[str] = None,
    input_schema: Optional[dict[str, Any]] = None,
    required_params: Optional[list[str]] = None,
) -> ToolDefinition:
    """
    Create a ToolDefinition from schema data (ToolSchema-compatible).
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool
        description: Tool description
        input_schema: JSON schema for input parameters
        required_params: List of required parameter names
        
    Returns:
        ToolDefinition instance
    """
    parameters: list[ToolParameter] = []
    
    if input_schema and "properties" in input_schema:
        required_set = set(required_params or input_schema.get("required", []))
        
        for param_name, param_schema in input_schema["properties"].items():
            param = ToolParameter(
                name=param_name,
                type=param_schema.get("type", "string"),
                description=param_schema.get("description"),
                required=param_name in required_set,
                default=param_schema.get("default"),
            )
            parameters.append(param)
    
    return ToolDefinition(
        server_name=server_name,
        tool_name=tool_name,
        description=description,
        parameters=parameters,
    )
