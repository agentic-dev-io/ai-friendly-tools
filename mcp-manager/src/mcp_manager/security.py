"""Security module for MCP Gateway with allowlist validation."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SecurityConfig(BaseModel):
    """Security configuration for MCP operations."""

    allowed_commands: list[str] = Field(
        default_factory=list,
        description="Colon-delimited executable paths allowed for MCP servers"
    )
    allowed_urls: list[str] = Field(
        default_factory=list,
        description="Space-delimited URL prefixes allowed for MCP servers"
    )
    lock_servers: bool = Field(
        default=True,
        description="Lock server configuration to prevent runtime changes"
    )
    log_level: str = Field(
        default="info",
        description="MCP logging level (trace, debug, info, warn, error, off)"
    )
    log_file: str = Field(
        default="",
        description="Path to MCP log file (empty for no file logging)"
    )
    console_logging: bool = Field(
        default=False,
        description="Enable MCP logging to console/stderr"
    )


def validate_command(cmd: str, config: SecurityConfig) -> bool:
    """
    Validate command against allowlist.
    
    Args:
        cmd: Command path to validate
        config: Security configuration with allowlist
        
    Returns:
        True if command is allowed, False otherwise
    """
    if not config.allowed_commands:
        return False

    # Normalize both paths to POSIX format for comparison
    cmd_posix = Path(cmd).as_posix()

    for allowed in config.allowed_commands:
        allowed_posix = Path(allowed).as_posix()
        if cmd_posix == allowed_posix:
            return True

    return False


def validate_url(url: str, config: SecurityConfig) -> bool:
    """
    Validate URL against allowlist of prefixes.
    
    Args:
        url: URL to validate
        config: Security configuration with allowlist
        
    Returns:
        True if URL prefix is allowed, False otherwise
    """
    if not config.allowed_urls:
        return False

    for allowed_prefix in config.allowed_urls:
        if url.startswith(allowed_prefix):
            return True

    return False


def apply_security_settings(conn: Any, config: SecurityConfig) -> None:
    """
    Apply security settings to DuckDB connection.
    
    Args:
        conn: DuckDB connection
        config: Security configuration to apply
    """
    # Format allowed commands as colon-delimited string
    # Use pathlib to normalize paths for cross-platform compatibility
    if config.allowed_commands:
        # Convert all paths to POSIX format (forward slashes)
        from pathlib import Path as PathLib
        normalized_commands = [PathLib(cmd).as_posix() for cmd in config.allowed_commands]
        commands_str = ":".join(normalized_commands)
        conn.execute(f"SET allowed_mcp_commands = '{commands_str}'")

    # Format allowed URLs as space-delimited string
    if config.allowed_urls:
        urls_str = " ".join(config.allowed_urls)
        conn.execute(f"SET allowed_mcp_urls = '{urls_str}'")

    # Set lock status
    conn.execute(f"SET mcp_lock_servers = {str(config.lock_servers).lower()}")

    # Set logging configuration
    conn.execute(f"SET mcp_log_level = '{config.log_level}'")

    if config.log_file:
        conn.execute(f"SET mcp_log_file = '{config.log_file}'")

    conn.execute(f"SET mcp_console_logging = {str(config.console_logging).lower()}")

