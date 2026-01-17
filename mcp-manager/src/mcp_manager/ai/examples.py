"""Auto-generated example functionality for MCP tools."""

import json
import random
import string
from dataclasses import dataclass, field
from typing import Any, Optional

from ..tools.schema import ToolSchema


@dataclass
class UsageExample:
    """A generated usage example for a tool."""
    
    tool_name: str
    server_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    expected_output: Optional[str] = None

    def to_cli_command(self) -> str:
        """Convert to CLI command string.
        
        Returns:
            CLI command to execute this example
        """
        args_json = json.dumps(self.arguments)
        return f"mcp-man call {self.server_name} {self.tool_name} '{args_json}'"

    def to_json(self) -> str:
        """Convert to JSON string.
        
        Returns:
            JSON representation
        """
        return json.dumps({
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "arguments": self.arguments,
            "description": self.description,
            "expected_output": self.expected_output,
        }, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "arguments": self.arguments,
            "description": self.description,
            "expected_output": self.expected_output,
        }


@dataclass
class ExampleTemplate:
    """Template for generating examples."""
    
    name: str
    description: str
    argument_generators: dict[str, Any] = field(default_factory=dict)


class ExampleGenerator:
    """Generate usage examples for MCP tools."""

    # Default sample values by type
    DEFAULT_VALUES: dict[str, Any] = {
        "string": "example_value",
        "integer": 42,
        "number": 3.14,
        "boolean": True,
        "array": ["item1", "item2"],
        "object": {"key": "value"},
    }

    # Sample values by common parameter names
    NAMED_SAMPLES: dict[str, Any] = {
        "path": "/path/to/file.txt",
        "file": "document.txt",
        "filename": "data.json",
        "directory": "/home/user/documents",
        "dir": "/tmp",
        "folder": "/var/data",
        "url": "https://api.example.com/data",
        "uri": "file:///path/to/resource",
        "query": "SELECT * FROM users",
        "sql": "SELECT id, name FROM table",
        "name": "example_name",
        "id": "12345",
        "user": "john_doe",
        "username": "admin",
        "email": "user@example.com",
        "message": "Hello, World!",
        "text": "Sample text content",
        "content": "File content goes here",
        "data": {"key": "value"},
        "config": {"option1": True, "option2": "setting"},
        "format": "json",
        "encoding": "utf-8",
        "limit": 10,
        "count": 5,
        "offset": 0,
        "page": 1,
        "size": 100,
        "timeout": 30,
        "port": 8080,
        "host": "localhost",
        "database": "mydb",
        "table": "users",
        "column": "id",
        "key": "api_key_here",
        "token": "auth_token_here",
        "password": "********",
        "secret": "********",
    }

    def generate(self, tool: ToolSchema) -> UsageExample:
        """Generate a usage example for a tool.
        
        Args:
            tool: Tool schema
            
        Returns:
            Generated UsageExample
        """
        arguments = self._generate_arguments(tool)
        description = self._generate_description(tool)

        return UsageExample(
            tool_name=tool.tool_name,
            server_name=tool.server_name,
            arguments=arguments,
            description=description,
        )

    def generate_multiple(
        self,
        tool: ToolSchema,
        count: int = 3,
    ) -> list[UsageExample]:
        """Generate multiple usage examples.
        
        Args:
            tool: Tool schema
            count: Number of examples to generate
            
        Returns:
            List of UsageExample objects
        """
        examples = []
        for i in range(count):
            arguments = self._generate_arguments(tool, variation=i)
            description = self._generate_description(tool, variation=i)
            
            examples.append(UsageExample(
                tool_name=tool.tool_name,
                server_name=tool.server_name,
                arguments=arguments,
                description=description,
            ))
        
        return examples

    def _generate_arguments(
        self,
        tool: ToolSchema,
        variation: int = 0,
    ) -> dict[str, Any]:
        """Generate argument values for a tool.
        
        Args:
            tool: Tool schema
            variation: Variation index for generating different examples
            
        Returns:
            Dictionary of argument values
        """
        arguments: dict[str, Any] = {}

        if not tool.input_schema:
            return arguments

        properties = tool.input_schema.get("properties", {})
        required = set(tool.required_params or [])

        for param_name, param_schema in properties.items():
            value = self._generate_value(param_name, param_schema, variation)
            
            # Include all required params, and some optional ones
            if param_name in required or random.random() > 0.3:
                arguments[param_name] = value

        return arguments

    def _generate_value(
        self,
        param_name: str,
        param_schema: dict[str, Any],
        variation: int = 0,
    ) -> Any:
        """Generate a value for a parameter.
        
        Args:
            param_name: Parameter name
            param_schema: Parameter JSON schema
            variation: Variation index
            
        Returns:
            Generated value
        """
        # Check for default value
        if "default" in param_schema:
            return param_schema["default"]

        # Check for enum values
        if "enum" in param_schema:
            enum_values = param_schema["enum"]
            return enum_values[variation % len(enum_values)]

        # Check for named samples
        param_lower = param_name.lower()
        for key, value in self.NAMED_SAMPLES.items():
            if key in param_lower:
                if variation > 0 and isinstance(value, str):
                    return f"{value}_{variation}"
                return value

        # Generate by type
        param_type = param_schema.get("type", "string")
        
        if param_type == "string":
            return self._generate_string(param_name, param_schema, variation)
        elif param_type == "integer":
            return self._generate_integer(param_schema, variation)
        elif param_type == "number":
            return self._generate_number(param_schema, variation)
        elif param_type == "boolean":
            return variation % 2 == 0
        elif param_type == "array":
            return self._generate_array(param_schema, variation)
        elif param_type == "object":
            return self._generate_object(param_schema, variation)
        else:
            return self.DEFAULT_VALUES.get(param_type, "value")

    def _generate_string(
        self,
        param_name: str,
        schema: dict[str, Any],
        variation: int,
    ) -> str:
        """Generate a string value."""
        # Check for pattern
        if "pattern" in schema:
            # Simple pattern handling
            return f"pattern_match_{variation}"

        # Check for format
        fmt = schema.get("format", "")
        if fmt == "email":
            return f"user{variation}@example.com"
        elif fmt == "uri" or fmt == "url":
            return f"https://example.com/resource/{variation}"
        elif fmt == "date":
            return "2024-01-15"
        elif fmt == "date-time":
            return "2024-01-15T10:30:00Z"

        # Generate based on min/max length
        min_len = schema.get("minLength", 1)
        max_len = schema.get("maxLength", 50)
        length = min(max(min_len, 10), max_len)
        
        if variation == 0:
            return f"example_{param_name}"
        else:
            return f"example_{param_name}_{variation}"

    def _generate_integer(self, schema: dict[str, Any], variation: int) -> int:
        """Generate an integer value."""
        minimum = schema.get("minimum", 1)
        maximum = schema.get("maximum", 1000)
        return minimum + (variation * 10) % (maximum - minimum + 1)

    def _generate_number(self, schema: dict[str, Any], variation: int) -> float:
        """Generate a number value."""
        minimum = schema.get("minimum", 0.0)
        maximum = schema.get("maximum", 100.0)
        return minimum + (variation * 0.5) % (maximum - minimum)

    def _generate_array(self, schema: dict[str, Any], variation: int) -> list[Any]:
        """Generate an array value."""
        items_schema = schema.get("items", {"type": "string"})
        min_items = schema.get("minItems", 1)
        max_items = schema.get("maxItems", 5)
        
        count = min(max(min_items, 2), max_items)
        
        return [
            self._generate_value(f"item_{i}", items_schema, variation)
            for i in range(count)
        ]

    def _generate_object(self, schema: dict[str, Any], variation: int) -> dict[str, Any]:
        """Generate an object value."""
        properties = schema.get("properties", {})
        
        if not properties:
            return {"key": f"value_{variation}"}
        
        result = {}
        for prop_name, prop_schema in properties.items():
            result[prop_name] = self._generate_value(prop_name, prop_schema, variation)
        
        return result

    def _generate_description(self, tool: ToolSchema, variation: int = 0) -> str:
        """Generate a description for the example.
        
        Args:
            tool: Tool schema
            variation: Variation index
            
        Returns:
            Description string
        """
        base_desc = tool.description or f"Execute {tool.tool_name}"
        
        if variation == 0:
            return f"Basic usage: {base_desc}"
        elif variation == 1:
            return f"Alternative usage: {base_desc}"
        else:
            return f"Example {variation}: {base_desc}"


class SmartExampleBuilder:
    """Build examples using context and history."""

    def __init__(self) -> None:
        """Initialize the builder."""
        self.generator = ExampleGenerator()

    def build(
        self,
        tool: ToolSchema,
        context: Optional[dict[str, Any]] = None,
    ) -> UsageExample:
        """Build an example using context.
        
        Args:
            tool: Tool schema
            context: Optional context dictionary
            
        Returns:
            UsageExample with context-aware values
        """
        example = self.generator.generate(tool)
        
        if context:
            # Override arguments with context values
            for key, value in context.items():
                # Check if key matches any parameter
                if tool.input_schema and "properties" in tool.input_schema:
                    properties = tool.input_schema["properties"]
                    for param_name in properties:
                        if key.lower() in param_name.lower() or param_name.lower() in key.lower():
                            example.arguments[param_name] = value
        
        return example

    def build_from_history(
        self,
        tool: ToolSchema,
        history: list[dict[str, Any]],
    ) -> UsageExample:
        """Build example from execution history.
        
        Args:
            tool: Tool schema
            history: List of historical executions
            
        Returns:
            UsageExample based on common patterns
        """
        if not history:
            return self.generator.generate(tool)

        # Analyze history to find common argument patterns
        arg_frequencies: dict[str, dict[str, int]] = {}
        
        for entry in history:
            args = entry.get("arguments", {})
            for key, value in args.items():
                if key not in arg_frequencies:
                    arg_frequencies[key] = {}
                
                value_str = str(value)
                if value_str not in arg_frequencies[key]:
                    arg_frequencies[key][value_str] = 0
                arg_frequencies[key][value_str] += 1

        # Use most common values
        arguments = {}
        for param, value_counts in arg_frequencies.items():
            most_common = max(value_counts.items(), key=lambda x: x[1])
            # Try to preserve the original type
            try:
                arguments[param] = json.loads(most_common[0])
            except (json.JSONDecodeError, TypeError):
                arguments[param] = most_common[0]

        return UsageExample(
            tool_name=tool.tool_name,
            server_name=tool.server_name,
            arguments=arguments,
            description=f"Common usage pattern for {tool.tool_name}",
        )

    def build_variations(
        self,
        tool: ToolSchema,
        count: int = 3,
        context: Optional[dict[str, Any]] = None,
    ) -> list[UsageExample]:
        """Build multiple variations of examples.
        
        Args:
            tool: Tool schema
            count: Number of variations
            context: Optional context
            
        Returns:
            List of UsageExample variations
        """
        examples = []
        
        for i in range(count):
            example = self.generator.generate(tool)
            
            # Apply context if provided
            if context:
                for key, value in context.items():
                    if key in example.arguments:
                        if isinstance(value, list) and i < len(value):
                            example.arguments[key] = value[i]
                        else:
                            example.arguments[key] = value
            
            example.description = f"Variation {i+1}: {example.description}"
            examples.append(example)
        
        return examples
