"""Example generation for MCP tools with realistic sample data."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from ..tools.schema import ToolSchema


class ExampleLevel(Enum):
    """Complexity level for generated examples."""
    
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class Example:
    """A single usage example for a tool."""
    
    level: ExampleLevel
    description: str
    arguments: dict[str, Any]
    expected_output: Optional[dict[str, Any]] = None
    use_case: str = ""
    notes: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert example to dictionary representation."""
        return {
            "level": self.level.value,
            "description": self.description,
            "arguments": self.arguments,
            "expected_output": self.expected_output,
            "use_case": self.use_case,
            "notes": self.notes,
        }


class ExampleGenerator:
    """Generates usage examples for MCP tools based on their schemas."""
    
    # Sample data pools for realistic value generation
    SAMPLE_STRINGS: dict[str, list[str]] = {
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "username": ["alice_dev", "bob123", "charlie_code", "diana_ml", "eve_data"],
        "email": ["alice@example.com", "bob@company.org", "charlie@dev.io"],
        "filename": ["report.pdf", "data.csv", "config.json", "README.md", "script.py"],
        "path": ["/home/user/documents", "/var/log", "/tmp/output", "./data"],
        "url": ["https://api.example.com", "https://data.company.org/v1"],
        "id": ["usr_12345", "doc_abc123", "txn_xyz789", "item_001"],
        "query": ["SELECT * FROM users", "status:active", "type:report"],
        "message": ["Hello, World!", "Processing complete", "Task executed successfully"],
        "description": ["A sample description", "Detailed explanation here", "Brief note"],
        "title": ["Monthly Report", "Data Analysis", "System Status", "User Guide"],
        "content": ["Lorem ipsum dolor sit amet", "Sample content for testing"],
        "format": ["json", "csv", "xml", "yaml", "markdown"],
        "status": ["active", "pending", "completed", "failed", "cancelled"],
        "type": ["user", "admin", "system", "guest", "service"],
        "category": ["finance", "technology", "health", "education", "general"],
        "tag": ["important", "urgent", "review", "draft", "archived"],
        "default": ["sample_value", "test_data", "example_input"],
    }
    
    SAMPLE_NUMBERS: dict[str, tuple[int, int]] = {
        "id": (1, 10000),
        "count": (1, 100),
        "limit": (10, 1000),
        "offset": (0, 500),
        "page": (1, 50),
        "size": (1, 1024),
        "port": (1024, 65535),
        "timeout": (1000, 60000),
        "retries": (1, 10),
        "age": (18, 80),
        "quantity": (1, 100),
        "price": (1, 10000),
        "score": (0, 100),
        "default": (1, 100),
    }
    
    SAMPLE_ARRAYS: dict[str, list[list[Any]]] = {
        "ids": [["id_1", "id_2", "id_3"], ["usr_001", "usr_002"]],
        "tags": [["urgent", "review"], ["draft", "archived", "important"]],
        "fields": [["name", "email", "status"], ["id", "created_at"]],
        "values": [[1, 2, 3], ["a", "b", "c"]],
        "files": [["file1.txt", "file2.csv"], ["report.pdf", "data.json"]],
        "default": [["item1", "item2"], ["value_a", "value_b", "value_c"]],
    }
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the example generator.
        
        Args:
            seed: Optional random seed for reproducible examples
        """
        if seed is not None:
            random.seed(seed)
    
    def generate_examples(self, tool: ToolSchema) -> list[Example]:
        """
        Generate examples at all complexity levels for a tool.
        
        Args:
            tool: The tool schema to generate examples for
            
        Returns:
            List of Example objects at different complexity levels
        """
        examples: list[Example] = []
        
        if not tool.input_schema:
            # Tool takes no parameters
            examples.append(Example(
                level=ExampleLevel.SIMPLE,
                description=f"Basic invocation of {tool.tool_name}",
                arguments={},
                use_case=f"Call {tool.tool_name} with no arguments",
            ))
            return examples
        
        properties = tool.input_schema.get("properties", {})
        required = tool.required_params or tool.input_schema.get("required", [])
        
        # Generate SIMPLE example - only required params
        if required:
            simple_args = self._generate_args_for_params(
                {k: v for k, v in properties.items() if k in required},
                complexity="simple"
            )
            examples.append(Example(
                level=ExampleLevel.SIMPLE,
                description=f"Minimal example with required parameters only",
                arguments=simple_args,
                use_case=self._generate_use_case(tool, "simple"),
                expected_output=self._generate_expected_output(tool, simple_args),
            ))
        
        # Generate INTERMEDIATE example - required + some optional
        optional_params = {k: v for k, v in properties.items() if k not in required}
        if optional_params:
            # Include half of optional params
            selected_optional = dict(list(optional_params.items())[:len(optional_params)//2 + 1])
            intermediate_args = self._generate_args_for_params(
                {**{k: v for k, v in properties.items() if k in required}, **selected_optional},
                complexity="intermediate"
            )
            examples.append(Example(
                level=ExampleLevel.INTERMEDIATE,
                description=f"Extended example with common optional parameters",
                arguments=intermediate_args,
                use_case=self._generate_use_case(tool, "intermediate"),
                expected_output=self._generate_expected_output(tool, intermediate_args),
                notes=self._generate_notes(tool, selected_optional),
            ))
        
        # Generate ADVANCED example - all params with complex values
        all_args = self._generate_args_for_params(properties, complexity="advanced")
        examples.append(Example(
            level=ExampleLevel.ADVANCED,
            description=f"Complete example with all parameters",
            arguments=all_args,
            use_case=self._generate_use_case(tool, "advanced"),
            expected_output=self._generate_expected_output(tool, all_args),
            notes=self._generate_notes(tool, properties, advanced=True),
        ))
        
        return examples
    
    def generate_sample_values(
        self,
        param_type: str,
        param_name: str,
        param_schema: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Generate a realistic sample value for a parameter.
        
        Args:
            param_type: JSON schema type (string, integer, number, boolean, array, object)
            param_name: Name of the parameter for context-aware generation
            param_schema: Optional full schema for the parameter
            
        Returns:
            A sample value appropriate for the parameter type and name
        """
        param_schema = param_schema or {}
        
        # Handle enum values
        if "enum" in param_schema:
            return random.choice(param_schema["enum"])
        
        # Handle const values
        if "const" in param_schema:
            return param_schema["const"]
        
        # Handle default values
        if "default" in param_schema:
            return param_schema["default"]
        
        # Generate based on type
        if param_type == "string":
            return self._generate_string_value(param_name, param_schema)
        elif param_type == "integer":
            return self._generate_integer_value(param_name, param_schema)
        elif param_type == "number":
            return self._generate_number_value(param_name, param_schema)
        elif param_type == "boolean":
            return random.choice([True, False])
        elif param_type == "array":
            return self._generate_array_value(param_name, param_schema)
        elif param_type == "object":
            return self._generate_object_value(param_name, param_schema)
        elif param_type == "null":
            return None
        else:
            # Default fallback
            return self._generate_string_value(param_name, {})
    
    def generate_markdown(self, tool: ToolSchema, examples: list[Example]) -> str:
        """
        Generate Claude-friendly markdown documentation for a tool.
        
        Args:
            tool: The tool schema
            examples: List of examples to include
            
        Returns:
            Formatted markdown string
        """
        lines: list[str] = []
        
        # Header
        lines.append(f"# {tool.tool_name}")
        lines.append("")
        
        # Description
        if tool.description:
            lines.append(tool.description)
            lines.append("")
        
        # Server info
        lines.append(f"**Server:** `{tool.server_name}`")
        lines.append("")
        
        # Parameters section
        if tool.input_schema and tool.input_schema.get("properties"):
            lines.append("## Parameters")
            lines.append("")
            lines.append("| Parameter | Type | Required | Description |")
            lines.append("|-----------|------|----------|-------------|")
            
            properties = tool.input_schema.get("properties", {})
            required = set(tool.required_params or tool.input_schema.get("required", []))
            
            for name, schema in properties.items():
                param_type = schema.get("type", "any")
                is_required = "Yes" if name in required else "No"
                description = schema.get("description", "-")
                lines.append(f"| `{name}` | `{param_type}` | {is_required} | {description} |")
            
            lines.append("")
        
        # Examples section
        lines.append("## Examples")
        lines.append("")
        
        for i, example in enumerate(examples, 1):
            level_emoji = {
                ExampleLevel.SIMPLE: "",
                ExampleLevel.INTERMEDIATE: "",
                ExampleLevel.ADVANCED: "",
            }.get(example.level, "")
            
            lines.append(f"### {level_emoji} Example {i}: {example.level.value.title()}")
            lines.append("")
            lines.append(f"**{example.description}**")
            lines.append("")
            
            if example.use_case:
                lines.append(f"*Use case:* {example.use_case}")
                lines.append("")
            
            # Arguments
            lines.append("**Arguments:**")
            lines.append("")
            lines.append("```json")
            lines.append(self._format_json(example.arguments))
            lines.append("```")
            lines.append("")
            
            # Expected output
            if example.expected_output:
                lines.append("**Expected Output:**")
                lines.append("")
                lines.append("```json")
                lines.append(self._format_json(example.expected_output))
                lines.append("```")
                lines.append("")
            
            # Notes
            if example.notes:
                lines.append("**Notes:**")
                for note in example.notes:
                    lines.append(f"- {note}")
                lines.append("")
        
        return "\n".join(lines)
    
    def generate_cli_command(
        self,
        tool: ToolSchema,
        args: dict[str, Any],
        cli_prefix: str = "mcp-manager",
    ) -> str:
        """
        Generate a CLI command string for executing the tool.
        
        Args:
            tool: The tool schema
            args: Arguments to pass
            cli_prefix: CLI command prefix
            
        Returns:
            Formatted CLI command string
        """
        parts = [cli_prefix, "execute", tool.server_name, tool.tool_name]
        
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    parts.append(f"--{key}")
            elif isinstance(value, (list, dict)):
                import json
                parts.append(f"--{key}={json.dumps(value)}")
            elif isinstance(value, str) and (" " in value or '"' in value):
                parts.append(f'--{key}="{value}"')
            else:
                parts.append(f"--{key}={value}")
        
        return " ".join(parts)
    
    def _generate_args_for_params(
        self,
        properties: dict[str, Any],
        complexity: str = "simple",
    ) -> dict[str, Any]:
        """Generate argument values for a set of parameters."""
        args: dict[str, Any] = {}
        
        for name, schema in properties.items():
            param_type = schema.get("type", "string")
            
            # For simple complexity, use basic values
            if complexity == "simple":
                args[name] = self.generate_sample_values(param_type, name, schema)
            elif complexity == "intermediate":
                args[name] = self.generate_sample_values(param_type, name, schema)
            else:
                # Advanced - use more complex/realistic values
                args[name] = self._generate_advanced_value(param_type, name, schema)
        
        return args
    
    def _generate_advanced_value(
        self,
        param_type: str,
        param_name: str,
        param_schema: dict[str, Any],
    ) -> Any:
        """Generate more complex values for advanced examples."""
        if param_type == "string" and "format" in param_schema:
            format_type = param_schema["format"]
            if format_type == "date-time":
                return datetime.now().isoformat()
            elif format_type == "date":
                return datetime.now().strftime("%Y-%m-%d")
            elif format_type == "time":
                return datetime.now().strftime("%H:%M:%S")
            elif format_type == "email":
                return f"user_{random.randint(1, 999)}@example.com"
            elif format_type == "uri":
                return f"https://api.example.com/v2/resources/{random.randint(1, 9999)}"
            elif format_type == "uuid":
                return self._generate_uuid()
        
        return self.generate_sample_values(param_type, param_name, param_schema)
    
    def _generate_string_value(
        self,
        param_name: str,
        param_schema: dict[str, Any],
    ) -> str:
        """Generate a string value based on parameter name and schema."""
        name_lower = param_name.lower()
        
        # Check for format hints in schema
        if "format" in param_schema:
            format_type = param_schema["format"]
            if format_type == "date-time":
                return datetime.now().isoformat()
            elif format_type == "date":
                return datetime.now().strftime("%Y-%m-%d")
            elif format_type == "email":
                return random.choice(self.SAMPLE_STRINGS["email"])
            elif format_type == "uri":
                return random.choice(self.SAMPLE_STRINGS["url"])
            elif format_type == "uuid":
                return self._generate_uuid()
        
        # Match by name pattern
        for pattern, values in self.SAMPLE_STRINGS.items():
            if pattern in name_lower:
                return random.choice(values)
        
        # Check common suffixes
        if name_lower.endswith("_id") or name_lower.endswith("id"):
            return random.choice(self.SAMPLE_STRINGS["id"])
        if name_lower.endswith("_name") or name_lower.endswith("name"):
            return random.choice(self.SAMPLE_STRINGS["name"])
        if name_lower.endswith("_path") or name_lower.endswith("path"):
            return random.choice(self.SAMPLE_STRINGS["path"])
        if name_lower.endswith("_url") or name_lower.endswith("url"):
            return random.choice(self.SAMPLE_STRINGS["url"])
        
        # Default string
        return random.choice(self.SAMPLE_STRINGS["default"])
    
    def _generate_integer_value(
        self,
        param_name: str,
        param_schema: dict[str, Any],
    ) -> int:
        """Generate an integer value based on parameter name and schema."""
        name_lower = param_name.lower()
        
        # Check schema constraints
        minimum = param_schema.get("minimum", 0)
        maximum = param_schema.get("maximum", 10000)
        
        # Match by name pattern
        for pattern, (low, high) in self.SAMPLE_NUMBERS.items():
            if pattern in name_lower:
                return random.randint(max(low, minimum), min(high, maximum))
        
        # Default range
        default_low, default_high = self.SAMPLE_NUMBERS["default"]
        return random.randint(max(default_low, minimum), min(default_high, maximum))
    
    def _generate_number_value(
        self,
        param_name: str,
        param_schema: dict[str, Any],
    ) -> float:
        """Generate a floating-point number value."""
        int_value = self._generate_integer_value(param_name, param_schema)
        return round(int_value + random.random(), 2)
    
    def _generate_array_value(
        self,
        param_name: str,
        param_schema: dict[str, Any],
    ) -> list[Any]:
        """Generate an array value based on parameter name and schema."""
        name_lower = param_name.lower()
        
        # Check for items schema
        items_schema = param_schema.get("items", {})
        items_type = items_schema.get("type", "string")
        
        # Determine array length
        min_items = param_schema.get("minItems", 1)
        max_items = param_schema.get("maxItems", 5)
        length = random.randint(min_items, max_items)
        
        # Generate items
        if items_type == "string":
            # Check for matching sample arrays
            for pattern, arrays in self.SAMPLE_ARRAYS.items():
                if pattern in name_lower:
                    return random.choice(arrays)[:length]
            return [self._generate_string_value(param_name, items_schema) for _ in range(length)]
        elif items_type == "integer":
            return [self._generate_integer_value(param_name, items_schema) for _ in range(length)]
        elif items_type == "number":
            return [self._generate_number_value(param_name, items_schema) for _ in range(length)]
        elif items_type == "boolean":
            return [random.choice([True, False]) for _ in range(length)]
        elif items_type == "object":
            return [self._generate_object_value(param_name, items_schema) for _ in range(length)]
        
        # Default
        return random.choice(self.SAMPLE_ARRAYS["default"])[:length]
    
    def _generate_object_value(
        self,
        param_name: str,
        param_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate an object value based on parameter schema."""
        properties = param_schema.get("properties", {})
        
        if not properties:
            # Generate a simple generic object
            return {
                "key": "value",
                "data": random.choice(self.SAMPLE_STRINGS["default"]),
            }
        
        result: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            result[prop_name] = self.generate_sample_values(prop_type, prop_name, prop_schema)
        
        return result
    
    def _generate_uuid(self) -> str:
        """Generate a random UUID-like string."""
        segments = [
            ''.join(random.choices(string.hexdigits.lower(), k=8)),
            ''.join(random.choices(string.hexdigits.lower(), k=4)),
            ''.join(random.choices(string.hexdigits.lower(), k=4)),
            ''.join(random.choices(string.hexdigits.lower(), k=4)),
            ''.join(random.choices(string.hexdigits.lower(), k=12)),
        ]
        return '-'.join(segments)
    
    def _generate_use_case(self, tool: ToolSchema, complexity: str) -> str:
        """Generate a use case description for an example."""
        tool_name = tool.tool_name.replace("_", " ").title()
        
        use_cases = {
            "simple": [
                f"Quick execution of {tool_name} with minimal configuration",
                f"Basic {tool_name} operation for common scenarios",
                f"Standard {tool_name} call with required inputs only",
            ],
            "intermediate": [
                f"Enhanced {tool_name} with additional options",
                f"Customized {tool_name} execution for specific needs",
                f"Extended {tool_name} with filtering parameters",
            ],
            "advanced": [
                f"Full-featured {tool_name} with all available options",
                f"Production-ready {tool_name} configuration",
                f"Comprehensive {tool_name} for complex workflows",
            ],
        }
        
        return random.choice(use_cases.get(complexity, use_cases["simple"]))
    
    def _generate_expected_output(
        self,
        tool: ToolSchema,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate an expected output structure for an example."""
        return {
            "success": True,
            "data": {
                "tool": tool.tool_name,
                "server": tool.server_name,
                "result": "Operation completed successfully",
            },
            "metadata": {
                "execution_time_ms": random.randint(10, 500),
                "timestamp": datetime.now().isoformat(),
            },
        }
    
    def _generate_notes(
        self,
        tool: ToolSchema,
        params: dict[str, Any],
        advanced: bool = False,
    ) -> list[str]:
        """Generate helpful notes for an example."""
        notes: list[str] = []
        
        for name, schema in params.items():
            param_type = schema.get("type", "string")
            
            if "enum" in schema:
                notes.append(f"`{name}` accepts: {', '.join(repr(v) for v in schema['enum'])}")
            
            if param_type == "integer":
                if "minimum" in schema or "maximum" in schema:
                    min_val = schema.get("minimum", "unbounded")
                    max_val = schema.get("maximum", "unbounded")
                    notes.append(f"`{name}` range: {min_val} to {max_val}")
            
            if param_type == "string" and "format" in schema:
                notes.append(f"`{name}` expects {schema['format']} format")
        
        if advanced:
            notes.append("This example demonstrates all available parameters")
            if tool.description:
                notes.append(f"Tool purpose: {tool.description[:100]}")
        
        return notes[:5]  # Limit to 5 notes
    
    def _format_json(self, obj: Any, indent: int = 2) -> str:
        """Format an object as pretty-printed JSON."""
        import json
        return json.dumps(obj, indent=indent, default=str)
