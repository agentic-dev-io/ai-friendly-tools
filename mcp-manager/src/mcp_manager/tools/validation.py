"""Tool call validation and error handling for MCP tools."""

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Optional

from loguru import logger
from pydantic import ValidationError, create_model, Field

from .schema import ToolSchema


@dataclass
class ValidationError_:
    """Represents a validation error for a tool parameter."""

    parameter: str
    message: str
    expected_type: Optional[str] = None
    got_type: Optional[str] = None
    suggestions: list[str] = None

    def __post_init__(self) -> None:
        if self.suggestions is None:
            self.suggestions = []

    def __str__(self) -> str:
        msg = f"Parameter '{self.parameter}': {self.message}"
        if self.expected_type and self.got_type:
            msg += f"\n  Expected type: {self.expected_type}, got: {self.got_type}"
        if self.suggestions:
            msg += f"\n  Suggestions: {', '.join(self.suggestions)}"
        return msg


@dataclass
class ValidationResult:
    """Result of tool parameter validation."""

    is_valid: bool
    errors: list[ValidationError_]
    warnings: list[str]
    validated_args: dict[str, Any]

    def __str__(self) -> str:
        lines = []
        if self.errors:
            lines.append("[Validation Errors]")
            for error in self.errors:
                lines.append(str(error))
        if self.warnings:
            lines.append("[Warnings]")
            for warning in self.warnings:
                lines.append(f"  {warning}")
        return "\n".join(lines) if lines else "Valid"


class ToolValidator:
    """Validator for tool parameters based on JSON schema."""

    def __init__(self) -> None:
        """Initialize the validator."""
        self._type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

    def validate(
        self,
        tool: ToolSchema,
        arguments: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate tool arguments against the tool's schema.

        Args:
            tool: ToolSchema instance
            arguments: Arguments to validate

        Returns:
            ValidationResult with errors, warnings, and validated args
        """
        errors: list[ValidationError_] = []
        warnings: list[str] = []
        validated_args = arguments.copy()

        # Check for required parameters
        if tool.required_params:
            for param in tool.required_params:
                if param not in arguments:
                    errors.append(
                        ValidationError_(
                            parameter=param,
                            message="Required parameter missing",
                            suggestions=self._suggest_parameters(param, arguments.keys()),
                        )
                    )

        # Validate against input schema if available
        if tool.input_schema:
            schema_errors = self._validate_against_schema(
                tool.input_schema, arguments, tool.tool_name
            )
            errors.extend(schema_errors)

        # Check for unexpected parameters
        if tool.input_schema and "properties" in tool.input_schema:
            expected_props = set(tool.input_schema["properties"].keys())
            provided_props = set(arguments.keys())
            unexpected = provided_props - expected_props

            for param in unexpected:
                similar = self._find_similar_params(param, expected_props)
                warnings.append(
                    f"Unexpected parameter '{param}'"
                    + (f", did you mean '{similar}'?" if similar else "?")
                )

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validated_args=validated_args if is_valid else {},
        )

    def _validate_against_schema(
        self,
        schema: dict[str, Any],
        arguments: dict[str, Any],
        tool_name: str,
    ) -> list[ValidationError_]:
        """Validate arguments against JSON schema."""
        errors: list[ValidationError_] = []

        if schema.get("type") != "object":
            return errors

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for param_name, param_schema in properties.items():
            if param_name not in arguments:
                if param_name in required:
                    errors.append(
                        ValidationError_(
                            parameter=param_name,
                            message="Required by schema",
                        )
                    )
                continue

            param_value = arguments[param_name]
            param_error = self._validate_parameter_type(
                param_name, param_value, param_schema
            )
            if param_error:
                errors.append(param_error)

        return errors

    def _validate_parameter_type(
        self,
        param_name: str,
        value: Any,
        param_schema: dict[str, Any],
    ) -> Optional[ValidationError_]:
        """Validate a single parameter type."""
        expected_type = param_schema.get("type")
        got_type = type(value).__name__

        if expected_type == "string" and not isinstance(value, str):
            return ValidationError_(
                parameter=param_name,
                message=f"Expected string, got {type(value).__name__}",
                expected_type="string",
                got_type=got_type,
            )

        if expected_type == "integer" and not isinstance(value, int):
            if isinstance(value, bool):
                # bool is subclass of int in Python, but we want to reject it
                return ValidationError_(
                    parameter=param_name,
                    message="Expected integer, got boolean",
                    expected_type="integer",
                    got_type="boolean",
                )
            return ValidationError_(
                parameter=param_name,
                message=f"Expected integer, got {type(value).__name__}",
                expected_type="integer",
                got_type=got_type,
            )

        if expected_type == "number" and not isinstance(value, (int, float)):
            return ValidationError_(
                parameter=param_name,
                message=f"Expected number, got {type(value).__name__}",
                expected_type="number",
                got_type=got_type,
            )

        if expected_type == "boolean" and not isinstance(value, bool):
            return ValidationError_(
                parameter=param_name,
                message=f"Expected boolean, got {type(value).__name__}",
                expected_type="boolean",
                got_type=got_type,
            )

        if expected_type == "array" and not isinstance(value, list):
            return ValidationError_(
                parameter=param_name,
                message=f"Expected array, got {type(value).__name__}",
                expected_type="array",
                got_type=got_type,
            )

        if expected_type == "object" and not isinstance(value, dict):
            return ValidationError_(
                parameter=param_name,
                message=f"Expected object, got {type(value).__name__}",
                expected_type="object",
                got_type=got_type,
            )

        return None

    def _suggest_parameters(
        self, missing_param: str, available_params: list[str]
    ) -> list[str]:
        """Suggest similar parameter names."""
        if not available_params:
            return []

        # Calculate similarity ratio for each available parameter
        similarities = [
            (param, SequenceMatcher(None, missing_param, param).ratio())
            for param in available_params
        ]

        # Sort by similarity and return top 3
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [param for param, ratio in similarities[:3] if ratio > 0.6]

    def _find_similar_params(self, param: str, expected_params: set[str]) -> Optional[str]:
        """Find the most similar expected parameter."""
        if not expected_params:
            return None

        similarities = [
            (exp_param, SequenceMatcher(None, param, exp_param).ratio())
            for exp_param in expected_params
        ]

        similarities.sort(key=lambda x: x[1], reverse=True)
        best_match, ratio = similarities[0] if similarities else (None, 0)

        return best_match if ratio > 0.6 else None


class ToolErrorSuggester:
    """Suggests similar tools when a tool call fails."""

    def __init__(self) -> None:
        """Initialize the suggester."""
        pass

    def suggest_tools(
        self,
        tool_name: str,
        available_tools: list[ToolSchema],
        max_suggestions: int = 3,
    ) -> list[tuple[ToolSchema, float]]:
        """
        Suggest similar tools based on tool name.

        Args:
            tool_name: Name of the tool that failed
            available_tools: List of available tools to search
            max_suggestions: Maximum number of suggestions

        Returns:
            List of (ToolSchema, similarity_score) tuples
        """
        if not available_tools:
            return []

        similarities = [
            (tool, SequenceMatcher(None, tool_name, tool.tool_name).ratio())
            for tool in available_tools
        ]

        # Sort by similarity and filter
        similarities.sort(key=lambda x: x[1], reverse=True)
        suggestions = [
            (tool, score)
            for tool, score in similarities[:max_suggestions]
            if score > 0.5
        ]

        return suggestions

    def suggest_by_description(
        self,
        error_message: str,
        available_tools: list[ToolSchema],
        max_suggestions: int = 3,
    ) -> list[tuple[ToolSchema, float]]:
        """
        Suggest tools based on error message and tool descriptions.

        Args:
            error_message: Error message from failed tool call
            available_tools: List of available tools to search
            max_suggestions: Maximum number of suggestions

        Returns:
            List of (ToolSchema, similarity_score) tuples
        """
        if not available_tools or not error_message:
            return []

        # Extract keywords from error message
        error_words = error_message.lower().split()

        similarities = []
        for tool in available_tools:
            description = (tool.description or "").lower()
            # Count how many error words appear in description
            match_count = sum(1 for word in error_words if word in description)
            score = match_count / len(error_words) if error_words else 0
            similarities.append((tool, score))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        suggestions = [
            (tool, score)
            for tool, score in similarities[:max_suggestions]
            if score > 0
        ]

        return suggestions
