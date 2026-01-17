"""Template registry for common tool usage patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class TemplateCategory(Enum):
    """Categories for example templates."""
    
    FILE_OPERATIONS = "file_operations"
    API_CALLS = "api_calls"
    DATA_PROCESSING = "data_processing"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    MESSAGING = "messaging"
    SEARCH = "search"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ExampleTemplate:
    """A reusable template for tool usage examples."""
    
    name: str
    description: str
    sample_args: dict[str, Any]
    sample_output: Optional[dict[str, Any]] = None
    category: TemplateCategory = TemplateCategory.CUSTOM
    tags: list[str] = field(default_factory=list)
    applicable_patterns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert template to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "sample_args": self.sample_args,
            "sample_output": self.sample_output,
            "category": self.category.value,
            "tags": self.tags,
            "applicable_patterns": self.applicable_patterns,
            "notes": self.notes,
        }
    
    def matches_tool(self, tool_name: str, description: str = "") -> bool:
        """
        Check if this template is applicable to a tool.
        
        Args:
            tool_name: Name of the tool to check
            description: Optional tool description
            
        Returns:
            True if template likely applies to this tool
        """
        combined = f"{tool_name.lower()} {description.lower()}"
        
        for pattern in self.applicable_patterns:
            if pattern.lower() in combined:
                return True
        
        return False


class TemplateRegistry:
    """Registry for managing and applying example templates."""
    
    _templates: dict[str, ExampleTemplate] = {}
    _category_index: dict[TemplateCategory, list[str]] = {}
    
    @classmethod
    def register(cls, template: ExampleTemplate) -> None:
        """
        Register a new template.
        
        Args:
            template: The template to register
        """
        cls._templates[template.name] = template
        
        if template.category not in cls._category_index:
            cls._category_index[template.category] = []
        
        if template.name not in cls._category_index[template.category]:
            cls._category_index[template.category].append(template.name)
    
    @classmethod
    def get_template(cls, name: str) -> Optional[ExampleTemplate]:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            ExampleTemplate if found, None otherwise
        """
        return cls._templates.get(name)
    
    @classmethod
    def list_templates(
        cls,
        category: Optional[TemplateCategory] = None,
    ) -> list[ExampleTemplate]:
        """
        List all templates, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of matching templates
        """
        if category is not None:
            names = cls._category_index.get(category, [])
            return [cls._templates[name] for name in names if name in cls._templates]
        
        return list(cls._templates.values())
    
    @classmethod
    def find_matching_templates(
        cls,
        tool_name: str,
        description: str = "",
    ) -> list[ExampleTemplate]:
        """
        Find templates that match a given tool.
        
        Args:
            tool_name: Name of the tool
            description: Optional tool description
            
        Returns:
            List of matching templates
        """
        matches: list[ExampleTemplate] = []
        
        for template in cls._templates.values():
            if template.matches_tool(tool_name, description):
                matches.append(template)
        
        return matches
    
    @classmethod
    def apply_template(
        cls,
        template_name: str,
        overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Apply a template with optional argument overrides.
        
        Args:
            template_name: Name of the template to apply
            overrides: Optional dict of argument overrides
            
        Returns:
            Dictionary of arguments to use
            
        Raises:
            ValueError: If template not found
        """
        template = cls.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        args = template.sample_args.copy()
        if overrides:
            args.update(overrides)
        
        return args
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered templates."""
        cls._templates.clear()
        cls._category_index.clear()
    
    @classmethod
    def template_count(cls) -> int:
        """Return the number of registered templates."""
        return len(cls._templates)


# ============================================================================
# Built-in Templates: File Operations
# ============================================================================

FILE_READ_TEMPLATE = ExampleTemplate(
    name="file_read_basic",
    description="Read contents of a file",
    sample_args={
        "path": "/home/user/documents/report.txt",
        "encoding": "utf-8",
    },
    sample_output={
        "success": True,
        "content": "File contents here...",
        "size": 1024,
        "encoding": "utf-8",
    },
    category=TemplateCategory.FILE_OPERATIONS,
    tags=["read", "file", "text"],
    applicable_patterns=["read_file", "get_file", "file_read", "read_content"],
    notes=["Supports text and binary modes", "Default encoding is UTF-8"],
)

FILE_WRITE_TEMPLATE = ExampleTemplate(
    name="file_write_basic",
    description="Write content to a file",
    sample_args={
        "path": "/home/user/documents/output.txt",
        "content": "Hello, World!\nThis is sample content.",
        "encoding": "utf-8",
        "overwrite": True,
    },
    sample_output={
        "success": True,
        "path": "/home/user/documents/output.txt",
        "bytes_written": 42,
    },
    category=TemplateCategory.FILE_OPERATIONS,
    tags=["write", "file", "create"],
    applicable_patterns=["write_file", "save_file", "file_write", "create_file"],
    notes=["Set overwrite=False to prevent overwriting existing files"],
)

FILE_LIST_TEMPLATE = ExampleTemplate(
    name="file_list_directory",
    description="List files in a directory",
    sample_args={
        "path": "/home/user/documents",
        "pattern": "*.txt",
        "recursive": False,
    },
    sample_output={
        "success": True,
        "files": ["report.txt", "notes.txt", "readme.txt"],
        "count": 3,
    },
    category=TemplateCategory.FILE_OPERATIONS,
    tags=["list", "directory", "find"],
    applicable_patterns=["list_files", "list_dir", "get_files", "directory_list"],
    notes=["Use recursive=True to include subdirectories"],
)

FILE_DELETE_TEMPLATE = ExampleTemplate(
    name="file_delete",
    description="Delete a file or directory",
    sample_args={
        "path": "/home/user/temp/old_file.txt",
        "recursive": False,
    },
    sample_output={
        "success": True,
        "deleted_path": "/home/user/temp/old_file.txt",
    },
    category=TemplateCategory.FILE_OPERATIONS,
    tags=["delete", "remove", "file"],
    applicable_patterns=["delete_file", "remove_file", "file_delete", "rm"],
    notes=["Use recursive=True for directories with contents"],
)

# ============================================================================
# Built-in Templates: API Calls
# ============================================================================

API_GET_TEMPLATE = ExampleTemplate(
    name="api_get_request",
    description="Make a GET request to an API endpoint",
    sample_args={
        "url": "https://api.example.com/v1/users",
        "headers": {"Authorization": "Bearer token123"},
        "params": {"limit": 10, "offset": 0},
        "timeout": 30,
    },
    sample_output={
        "status_code": 200,
        "data": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "headers": {"Content-Type": "application/json"},
    },
    category=TemplateCategory.API_CALLS,
    tags=["api", "http", "get", "request"],
    applicable_patterns=["http_get", "api_get", "fetch", "get_request"],
    notes=["Always include appropriate authentication headers"],
)

API_POST_TEMPLATE = ExampleTemplate(
    name="api_post_request",
    description="Make a POST request to an API endpoint",
    sample_args={
        "url": "https://api.example.com/v1/users",
        "headers": {
            "Authorization": "Bearer token123",
            "Content-Type": "application/json",
        },
        "body": {"name": "Charlie", "email": "charlie@example.com"},
        "timeout": 30,
    },
    sample_output={
        "status_code": 201,
        "data": {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
    },
    category=TemplateCategory.API_CALLS,
    tags=["api", "http", "post", "create"],
    applicable_patterns=["http_post", "api_post", "post_request", "create"],
    notes=["Request body is automatically serialized to JSON"],
)

API_PUT_TEMPLATE = ExampleTemplate(
    name="api_put_request",
    description="Make a PUT request to update a resource",
    sample_args={
        "url": "https://api.example.com/v1/users/123",
        "headers": {"Authorization": "Bearer token123"},
        "body": {"name": "Charlie Updated", "email": "charlie_new@example.com"},
    },
    sample_output={
        "status_code": 200,
        "data": {"id": 123, "name": "Charlie Updated", "updated_at": "2024-01-15T10:30:00Z"},
    },
    category=TemplateCategory.API_CALLS,
    tags=["api", "http", "put", "update"],
    applicable_patterns=["http_put", "api_put", "update", "put_request"],
    notes=["PUT typically replaces the entire resource"],
)

API_DELETE_TEMPLATE = ExampleTemplate(
    name="api_delete_request",
    description="Make a DELETE request to remove a resource",
    sample_args={
        "url": "https://api.example.com/v1/users/123",
        "headers": {"Authorization": "Bearer token123"},
    },
    sample_output={
        "status_code": 204,
        "data": None,
    },
    category=TemplateCategory.API_CALLS,
    tags=["api", "http", "delete", "remove"],
    applicable_patterns=["http_delete", "api_delete", "delete_request"],
    notes=["204 No Content is the typical successful response"],
)

# ============================================================================
# Built-in Templates: Data Processing
# ============================================================================

DATA_TRANSFORM_TEMPLATE = ExampleTemplate(
    name="data_transform",
    description="Transform data from one format to another",
    sample_args={
        "input_data": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
        "input_format": "json",
        "output_format": "csv",
        "options": {"include_header": True},
    },
    sample_output={
        "success": True,
        "output": "name,age\nAlice,30\nBob,25",
        "records_processed": 2,
    },
    category=TemplateCategory.DATA_PROCESSING,
    tags=["transform", "convert", "format"],
    applicable_patterns=["transform", "convert", "format_data", "data_convert"],
    notes=["Supports JSON, CSV, XML, YAML formats"],
)

DATA_FILTER_TEMPLATE = ExampleTemplate(
    name="data_filter",
    description="Filter data based on conditions",
    sample_args={
        "data": [
            {"name": "Alice", "age": 30, "active": True},
            {"name": "Bob", "age": 25, "active": False},
            {"name": "Charlie", "age": 35, "active": True},
        ],
        "filters": {"active": True, "age__gte": 28},
    },
    sample_output={
        "success": True,
        "filtered_data": [
            {"name": "Alice", "age": 30, "active": True},
            {"name": "Charlie", "age": 35, "active": True},
        ],
        "original_count": 3,
        "filtered_count": 2,
    },
    category=TemplateCategory.DATA_PROCESSING,
    tags=["filter", "query", "select"],
    applicable_patterns=["filter", "query_data", "select", "data_filter"],
    notes=["Supports __gte, __lte, __gt, __lt, __contains modifiers"],
)

DATA_AGGREGATE_TEMPLATE = ExampleTemplate(
    name="data_aggregate",
    description="Aggregate data with grouping and calculations",
    sample_args={
        "data": [
            {"category": "A", "value": 100},
            {"category": "B", "value": 200},
            {"category": "A", "value": 150},
        ],
        "group_by": "category",
        "aggregations": {"value": ["sum", "avg", "count"]},
    },
    sample_output={
        "success": True,
        "results": [
            {"category": "A", "value_sum": 250, "value_avg": 125.0, "value_count": 2},
            {"category": "B", "value_sum": 200, "value_avg": 200.0, "value_count": 1},
        ],
    },
    category=TemplateCategory.DATA_PROCESSING,
    tags=["aggregate", "group", "statistics"],
    applicable_patterns=["aggregate", "group_by", "summarize", "statistics"],
    notes=["Available aggregations: sum, avg, min, max, count"],
)

DATA_VALIDATE_TEMPLATE = ExampleTemplate(
    name="data_validate",
    description="Validate data against a schema",
    sample_args={
        "data": {"name": "Alice", "email": "alice@example.com", "age": 30},
        "schema": {
            "type": "object",
            "required": ["name", "email"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "email": {"type": "string", "format": "email"},
                "age": {"type": "integer", "minimum": 0},
            },
        },
    },
    sample_output={
        "valid": True,
        "errors": [],
    },
    category=TemplateCategory.DATA_PROCESSING,
    tags=["validate", "schema", "check"],
    applicable_patterns=["validate", "check_data", "verify", "schema_validate"],
    notes=["Uses JSON Schema validation"],
)

# ============================================================================
# Built-in Templates: Database Operations
# ============================================================================

DB_QUERY_TEMPLATE = ExampleTemplate(
    name="database_query",
    description="Execute a database query",
    sample_args={
        "query": "SELECT id, name, email FROM users WHERE active = $1 LIMIT $2",
        "params": [True, 100],
        "database": "main",
    },
    sample_output={
        "success": True,
        "rows": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ],
        "row_count": 2,
        "execution_time_ms": 15,
    },
    category=TemplateCategory.DATABASE,
    tags=["query", "sql", "select", "database"],
    applicable_patterns=["query", "execute_query", "sql", "select"],
    notes=["Always use parameterized queries to prevent SQL injection"],
)

DB_INSERT_TEMPLATE = ExampleTemplate(
    name="database_insert",
    description="Insert data into a database table",
    sample_args={
        "table": "users",
        "data": {"name": "Charlie", "email": "charlie@example.com", "active": True},
        "returning": ["id"],
    },
    sample_output={
        "success": True,
        "inserted_id": 3,
        "rows_affected": 1,
    },
    category=TemplateCategory.DATABASE,
    tags=["insert", "create", "database"],
    applicable_patterns=["insert", "create_record", "add_record", "db_insert"],
    notes=["Use returning to get auto-generated values"],
)

DB_UPDATE_TEMPLATE = ExampleTemplate(
    name="database_update",
    description="Update records in a database table",
    sample_args={
        "table": "users",
        "data": {"active": False, "updated_at": "2024-01-15T10:30:00Z"},
        "where": {"id": 123},
    },
    sample_output={
        "success": True,
        "rows_affected": 1,
    },
    category=TemplateCategory.DATABASE,
    tags=["update", "modify", "database"],
    applicable_patterns=["update", "modify_record", "db_update", "edit"],
    notes=["Always include a WHERE clause to avoid updating all rows"],
)

# ============================================================================
# Built-in Templates: Search Operations
# ============================================================================

SEARCH_TEXT_TEMPLATE = ExampleTemplate(
    name="search_text",
    description="Perform full-text search",
    sample_args={
        "query": "machine learning algorithms",
        "fields": ["title", "content", "tags"],
        "limit": 20,
        "offset": 0,
        "highlight": True,
    },
    sample_output={
        "success": True,
        "results": [
            {
                "id": "doc_123",
                "title": "Introduction to Machine Learning",
                "score": 0.95,
                "highlights": ["...about <em>machine learning</em> <em>algorithms</em>..."],
            }
        ],
        "total_count": 42,
    },
    category=TemplateCategory.SEARCH,
    tags=["search", "text", "fulltext", "fts"],
    applicable_patterns=["search", "find", "query", "fulltext"],
    notes=["Results are sorted by relevance score"],
)

SEARCH_SEMANTIC_TEMPLATE = ExampleTemplate(
    name="search_semantic",
    description="Perform semantic/vector search",
    sample_args={
        "query": "How do neural networks learn patterns?",
        "collection": "documents",
        "top_k": 10,
        "min_score": 0.7,
    },
    sample_output={
        "success": True,
        "results": [
            {
                "id": "doc_456",
                "content": "Neural networks learn through backpropagation...",
                "similarity": 0.92,
                "metadata": {"source": "textbook.pdf"},
            }
        ],
        "query_embedding_time_ms": 45,
        "search_time_ms": 12,
    },
    category=TemplateCategory.SEARCH,
    tags=["search", "semantic", "vector", "embeddings"],
    applicable_patterns=["semantic_search", "vector_search", "similarity", "embedding"],
    notes=["Uses embedding model for semantic similarity"],
)

# ============================================================================
# Built-in Templates: System Operations
# ============================================================================

SYSTEM_EXEC_TEMPLATE = ExampleTemplate(
    name="system_execute_command",
    description="Execute a system command",
    sample_args={
        "command": "ls",
        "args": ["-la", "/home/user"],
        "timeout": 30,
        "working_dir": "/home/user",
    },
    sample_output={
        "success": True,
        "stdout": "total 48\ndrwxr-xr-x  5 user user 4096 Jan 15 10:30 .\n...",
        "stderr": "",
        "exit_code": 0,
        "execution_time_ms": 25,
    },
    category=TemplateCategory.SYSTEM,
    tags=["system", "command", "shell", "exec"],
    applicable_patterns=["execute", "run_command", "shell", "exec", "system"],
    notes=["Always validate and sanitize command arguments"],
)

SYSTEM_ENV_TEMPLATE = ExampleTemplate(
    name="system_environment",
    description="Get or set environment variables",
    sample_args={
        "action": "get",
        "name": "PATH",
    },
    sample_output={
        "success": True,
        "name": "PATH",
        "value": "/usr/local/bin:/usr/bin:/bin",
    },
    category=TemplateCategory.SYSTEM,
    tags=["environment", "env", "config"],
    applicable_patterns=["get_env", "set_env", "environment", "env_var"],
    notes=["Some environment variables may be restricted"],
)

# ============================================================================
# Built-in Templates: Authentication
# ============================================================================

AUTH_LOGIN_TEMPLATE = ExampleTemplate(
    name="auth_login",
    description="Authenticate a user",
    sample_args={
        "username": "alice@example.com",
        "password": "********",
        "remember": True,
    },
    sample_output={
        "success": True,
        "token": "eyJhbGciOiJIUzI1NiIs...",
        "expires_at": "2024-01-16T10:30:00Z",
        "user": {"id": 1, "name": "Alice", "role": "admin"},
    },
    category=TemplateCategory.AUTHENTICATION,
    tags=["auth", "login", "authenticate"],
    applicable_patterns=["login", "authenticate", "sign_in", "auth"],
    notes=["Never log or expose actual passwords"],
)

AUTH_VERIFY_TEMPLATE = ExampleTemplate(
    name="auth_verify_token",
    description="Verify an authentication token",
    sample_args={
        "token": "eyJhbGciOiJIUzI1NiIs...",
    },
    sample_output={
        "valid": True,
        "user_id": 1,
        "expires_at": "2024-01-16T10:30:00Z",
        "scopes": ["read", "write"],
    },
    category=TemplateCategory.AUTHENTICATION,
    tags=["auth", "token", "verify"],
    applicable_patterns=["verify_token", "validate_token", "check_auth"],
    notes=["Returns user context if token is valid"],
)

# ============================================================================
# Built-in Templates: Messaging
# ============================================================================

MSG_SEND_TEMPLATE = ExampleTemplate(
    name="message_send",
    description="Send a message",
    sample_args={
        "to": "user_123",
        "content": "Hello! This is a test message.",
        "type": "text",
        "metadata": {"priority": "normal"},
    },
    sample_output={
        "success": True,
        "message_id": "msg_abc123",
        "sent_at": "2024-01-15T10:30:00Z",
        "status": "delivered",
    },
    category=TemplateCategory.MESSAGING,
    tags=["message", "send", "notification"],
    applicable_patterns=["send_message", "notify", "message", "post_message"],
    notes=["Supports text, markdown, and rich content types"],
)

MSG_SUBSCRIBE_TEMPLATE = ExampleTemplate(
    name="message_subscribe",
    description="Subscribe to a message channel",
    sample_args={
        "channel": "notifications",
        "filters": {"type": ["alert", "update"]},
    },
    sample_output={
        "success": True,
        "subscription_id": "sub_xyz789",
        "channel": "notifications",
    },
    category=TemplateCategory.MESSAGING,
    tags=["subscribe", "channel", "pubsub"],
    applicable_patterns=["subscribe", "listen", "watch", "pubsub"],
    notes=["Use filters to receive only relevant messages"],
)


def _register_builtin_templates() -> None:
    """Register all built-in templates."""
    builtin_templates = [
        # File Operations
        FILE_READ_TEMPLATE,
        FILE_WRITE_TEMPLATE,
        FILE_LIST_TEMPLATE,
        FILE_DELETE_TEMPLATE,
        # API Calls
        API_GET_TEMPLATE,
        API_POST_TEMPLATE,
        API_PUT_TEMPLATE,
        API_DELETE_TEMPLATE,
        # Data Processing
        DATA_TRANSFORM_TEMPLATE,
        DATA_FILTER_TEMPLATE,
        DATA_AGGREGATE_TEMPLATE,
        DATA_VALIDATE_TEMPLATE,
        # Database
        DB_QUERY_TEMPLATE,
        DB_INSERT_TEMPLATE,
        DB_UPDATE_TEMPLATE,
        # Search
        SEARCH_TEXT_TEMPLATE,
        SEARCH_SEMANTIC_TEMPLATE,
        # System
        SYSTEM_EXEC_TEMPLATE,
        SYSTEM_ENV_TEMPLATE,
        # Authentication
        AUTH_LOGIN_TEMPLATE,
        AUTH_VERIFY_TEMPLATE,
        # Messaging
        MSG_SEND_TEMPLATE,
        MSG_SUBSCRIBE_TEMPLATE,
    ]
    
    for template in builtin_templates:
        TemplateRegistry.register(template)


# Helper functions for template access
def get_template(name: str) -> Optional[ExampleTemplate]:
    """
    Get a template by name.
    
    Args:
        name: Template name
        
    Returns:
        ExampleTemplate if found, None otherwise
    """
    return TemplateRegistry.get_template(name)


def list_templates(category: Optional[TemplateCategory] = None) -> list[ExampleTemplate]:
    """
    List all templates, optionally filtered by category.
    
    Args:
        category: Optional category filter
        
    Returns:
        List of matching templates
    """
    return TemplateRegistry.list_templates(category)


def apply_template(
    template_name: str,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Apply a template with optional argument overrides.
    
    Args:
        template_name: Name of the template to apply
        overrides: Optional dict of argument overrides
        
    Returns:
        Dictionary of arguments to use
        
    Raises:
        ValueError: If template not found
    """
    return TemplateRegistry.apply_template(template_name, overrides)


def find_templates_for_tool(
    tool_name: str,
    description: str = "",
) -> list[ExampleTemplate]:
    """
    Find templates that match a given tool.
    
    Args:
        tool_name: Name of the tool
        description: Optional tool description
        
    Returns:
        List of matching templates
    """
    return TemplateRegistry.find_matching_templates(tool_name, description)


# Register built-in templates on module import
_register_builtin_templates()
