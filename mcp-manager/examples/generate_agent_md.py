#!/usr/bin/env python3
"""
Example: Generate AGENT.md for MCP-Man

This example demonstrates how to use the AgentMarkdownGenerator
to create comprehensive AGENT.md documentation for AI agents.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_manager.agent import (
    AgentMarkdownGenerator,
    ServerInfo,
    ToolInfo,
    quick_agent_md,
)


def example_basic():
    """Example 1: Basic usage with quick_agent_md helper."""
    print("=" * 60)
    print("Example 1: Quick Agent MD Generation")
    print("=" * 60)
    
    # Define servers and their tools
    server_tools = {
        "filesystem": [
            ("read_file", "Read contents from a file"),
            ("write_file", "Write content to a file"),
            ("list_directory", "List files in a directory"),
            ("create_directory", "Create a new directory"),
            ("delete_file", "Delete a file"),
        ],
        "database": [
            ("query", "Execute SQL query on DuckDB"),
            ("create_table", "Create a new table"),
            ("insert_data", "Insert data into table"),
            ("list_tables", "List all tables in database"),
        ],
        "web": [
            ("fetch_url", "Fetch content from URL"),
            ("parse_html", "Parse HTML document"),
            ("search_web", "Search the web"),
        ],
    }
    
    # Generate markdown
    md = quick_agent_md(server_tools)
    print(md[:500] + "\n... (truncated)\n")
    
    return md


def example_advanced():
    """Example 2: Advanced usage with full control."""
    print("\n" + "=" * 60)
    print("Example 2: Advanced Generator with Detailed Tools")
    print("=" * 60)
    
    # Create generator
    generator = AgentMarkdownGenerator()
    
    # Define tools with parameters
    filesystem_tools = [
        ToolInfo(
            name="read_file",
            description="Read contents from a file",
            parameters=["path", "encoding"],
            required_params=["path"]
        ),
        ToolInfo(
            name="write_file",
            description="Write content to a file",
            parameters=["path", "content", "overwrite"],
            required_params=["path", "content"]
        ),
        ToolInfo(
            name="list_directory",
            description="List files in a directory",
            parameters=["path", "recursive"],
            required_params=["path"]
        ),
    ]
    
    database_tools = [
        ToolInfo(
            name="query",
            description="Execute SQL query on DuckDB",
            parameters=["sql", "params"],
            required_params=["sql"]
        ),
        ToolInfo(
            name="create_table",
            description="Create a new table",
            parameters=["name", "columns"],
            required_params=["name", "columns"]
        ),
    ]
    
    # Add to generator
    fs_server = ServerInfo(name="filesystem", tool_count=len(filesystem_tools))
    generator.add_server(fs_server)
    for tool in filesystem_tools:
        generator.add_tool_to_server("filesystem", tool)
    
    db_server = ServerInfo(name="database", tool_count=len(database_tools))
    generator.add_server(db_server)
    for tool in database_tools:
        generator.add_tool_to_server("database", tool)
    
    # Generate complete documentation
    md = generator.generate_agent_md(
        title="MCP-Man Agent Documentation",
        include_quick_ref=True,
        include_servers=True
    )
    
    print(f"Generated {len(md)} characters of documentation")
    print("\nFirst 600 characters:")
    print(md[:600])
    print("\n... (truncated)\n")
    
    return md


def example_save():
    """Example 3: Save to file."""
    print("\n" + "=" * 60)
    print("Example 3: Save to File")
    print("=" * 60)
    
    server_tools = {
        "core": [
            ("search", "Search for tools"),
            ("inspect", "Inspect tool details"),
            ("call", "Execute a tool"),
        ],
        "system": [
            ("health", "Check system health"),
            ("status", "Get status"),
        ],
    }
    
    # Generate and save
    generator = AgentMarkdownGenerator()
    
    for server_name, tools_list in server_tools.items():
        tools = [ToolInfo(name=n, description=d) for n, d in tools_list]
        server = ServerInfo(name=server_name, tool_count=len(tools))
        generator.add_server(server)
        for tool in tools:
            generator.add_tool_to_server(server_name, tool)
    
    # Save to file
    output_path = Path(__file__).parent / "AGENT.md"
    success = generator.generate_and_save(
        output_path,
        title="MCP-Man Example Documentation"
    )
    
    if success:
        print(f"✓ Successfully saved to {output_path}")
        print(f"  File size: {output_path.stat().st_size} bytes")
    else:
        print(f"✗ Failed to save to {output_path}")
    
    return success


def example_sections():
    """Example 4: Generate specific sections."""
    print("\n" + "=" * 60)
    print("Example 4: Individual Sections")
    print("=" * 60)
    
    generator = AgentMarkdownGenerator()
    
    # Generate specific sections
    print("\n1. Quick Reference:")
    print("-" * 40)
    print(generator.generate_quick_reference()[:400])
    
    print("\n2. Error Handling:")
    print("-" * 40)
    print(generator.generate_error_handling_section()[:400])
    
    print("\n3. Workflow:")
    print("-" * 40)
    print(generator.generate_workflow_section()[:400])
    
    print("\n4. Commands:")
    print("-" * 40)
    print(generator.generate_commands_section()[:400])


def main():
    """Run all examples."""
    print("\n🎯 MCP-Man AGENT.md Generator Examples\n")
    
    # Run examples
    md1 = example_basic()
    md2 = example_advanced()
    success = example_save()
    example_sections()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print(f"\n✓ Generated {len(md1)} chars (basic)")
    print(f"✓ Generated {len(md2)} chars (advanced)")
    print(f"{'✓' if success else '✗'} File save {'succeeded' if success else 'failed'}")


if __name__ == "__main__":
    main()
