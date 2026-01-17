"""
Example generation and documentation for MCP tools.

This module provides utilities for auto-generating usage examples
and Claude-friendly documentation for MCP tools based on their schemas.

Example usage:
    from mcp_manager.examples import ExampleGenerator, ExampleLevel, Example
    from mcp_manager.tools.schema import ToolSchema
    
    # Create a tool schema
    tool = ToolSchema(
        server_name="my-server",
        tool_name="get_user",
        description="Retrieves user information",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID"},
                "include_details": {"type": "boolean", "description": "Include extra details"},
            },
            "required": ["user_id"],
        },
        required_params=["user_id"],
    )
    
    # Generate examples
    generator = ExampleGenerator()
    examples = generator.generate_examples(tool)
    
    # Generate markdown documentation
    markdown = generator.generate_markdown(tool, examples)
    print(markdown)
    
    # Generate CLI command
    cli_cmd = generator.generate_cli_command(tool, {"user_id": "usr_123"})
    print(cli_cmd)  # mcp-manager execute my-server get_user --user_id=usr_123

Using templates:
    from mcp_manager.examples import (
        get_template,
        list_templates,
        apply_template,
        find_templates_for_tool,
        TemplateCategory,
    )
    
    # List all available templates
    all_templates = list_templates()
    
    # Filter by category
    file_templates = list_templates(TemplateCategory.FILE_OPERATIONS)
    
    # Find templates matching a tool
    matching = find_templates_for_tool("read_file", "Read contents of a file")
    
    # Apply a template with overrides
    args = apply_template("file_read_basic", {"path": "/custom/path.txt"})
"""

from .generator import Example, ExampleGenerator, ExampleLevel
from .templates import (
    ExampleTemplate,
    TemplateCategory,
    TemplateRegistry,
    apply_template,
    find_templates_for_tool,
    get_template,
    list_templates,
)

__all__ = [
    # Generator classes and enums
    "ExampleGenerator",
    "ExampleLevel",
    "Example",
    # Template classes and enums
    "ExampleTemplate",
    "TemplateCategory",
    "TemplateRegistry",
    # Template helper functions
    "get_template",
    "list_templates",
    "apply_template",
    "find_templates_for_tool",
]
