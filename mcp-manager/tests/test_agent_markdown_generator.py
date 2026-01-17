"""
Unit tests for AgentMarkdownGenerator

Tests the AGENT.md generation functionality for AI agents.
"""

import tempfile
from pathlib import Path

import pytest

# Adjust import path for tests
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_manager.agent import (
    AgentMarkdownGenerator,
    ServerInfo,
    ToolInfo,
    create_agent_md_from_registry,
    quick_agent_md,
)


class TestToolInfo:
    """Test ToolInfo data class."""
    
    def test_tool_info_creation(self):
        """Test creating a ToolInfo instance."""
        tool = ToolInfo(
            name="test_tool",
            description="Test tool description",
            parameters=["param1", "param2"]
        )
        assert tool.name == "test_tool"
        assert tool.description == "Test tool description"
        assert len(tool.parameters) == 2
    
    def test_tool_info_format_params_with_params(self):
        """Test formatting parameters."""
        tool = ToolInfo(
            name="test",
            description="desc",
            parameters=["a", "b", "c"]
        )
        assert tool.format_params() == "a, b, c"
    
    def test_tool_info_format_params_empty(self):
        """Test formatting with no parameters."""
        tool = ToolInfo(name="test", description="desc")
        assert tool.format_params() == ""
    
    def test_tool_info_to_markdown_with_params(self):
        """Test markdown conversion with parameters."""
        tool = ToolInfo(
            name="read_file",
            description="Reads a file",
            parameters=["path", "encoding"]
        )
        md = tool.to_markdown()
        assert "read_file" in md
        assert "Reads a file" in md
        assert "path, encoding" in md
        assert md.startswith("- **")
    
    def test_tool_info_to_markdown_no_params(self):
        """Test markdown conversion without parameters."""
        tool = ToolInfo(name="test", description="Test tool")
        md = tool.to_markdown()
        assert "test" in md
        assert "Test tool" in md
        assert "params:" not in md


class TestServerInfo:
    """Test ServerInfo data class."""
    
    def test_server_info_creation(self):
        """Test creating a ServerInfo instance."""
        server = ServerInfo(name="test_server", tool_count=5)
        assert server.name == "test_server"
        assert server.status == "active"
        assert server.tool_count == 5
        assert server.tools == []
    
    def test_server_info_with_tools(self):
        """Test ServerInfo with tools list."""
        tools = [
            ToolInfo(name="t1", description="Tool 1"),
            ToolInfo(name="t2", description="Tool 2"),
        ]
        server = ServerInfo(name="server", tools=tools)
        assert len(server.tools) == 2
        assert server.tool_count == 2
    
    def test_server_info_timestamp(self):
        """Test that last_checked timestamp is set."""
        server = ServerInfo(name="server")
        assert server.last_checked is not None
        assert "T" in server.last_checked  # ISO format


class TestAgentMarkdownGenerator:
    """Test AgentMarkdownGenerator class."""
    
    def test_generator_creation(self):
        """Test creating a generator instance."""
        gen = AgentMarkdownGenerator()
        assert gen.servers == {}
        assert gen.generated_at is not None
    
    def test_add_server(self):
        """Test adding a server."""
        gen = AgentMarkdownGenerator()
        server = ServerInfo(name="test", tool_count=3)
        gen.add_server(server)
        
        assert "test" in gen.servers
        assert gen.servers["test"].name == "test"
    
    def test_add_tool_to_server(self):
        """Test adding a tool to a server."""
        gen = AgentMarkdownGenerator()
        tool = ToolInfo(name="test_tool", description="Test")
        
        gen.add_tool_to_server("test_server", tool)
        
        assert "test_server" in gen.servers
        assert len(gen.servers["test_server"].tools) == 1
        assert gen.servers["test_server"].tools[0].name == "test_tool"
    
    def test_add_tool_creates_server_if_not_exists(self):
        """Test that adding a tool creates server if needed."""
        gen = AgentMarkdownGenerator()
        tool = ToolInfo(name="tool", description="desc")
        
        gen.add_tool_to_server("new_server", tool)
        
        assert "new_server" in gen.servers
    
    def test_generate_quick_reference(self):
        """Test generating quick reference section."""
        gen = AgentMarkdownGenerator()
        ref = gen.generate_quick_reference()
        
        assert "Quick Command Reference" in ref
        assert "mcp-man search" in ref
        assert "mcp-man tools" in ref
        assert "mcp-man inspect" in ref
        assert "mcp-man call" in ref
        assert "```bash" in ref
    
    def test_generate_workflow_section(self):
        """Test generating workflow section."""
        gen = AgentMarkdownGenerator()
        workflow = gen.generate_workflow_section()
        
        assert "Empfohlener Workflow" in workflow
        assert "Search" in workflow
        assert "Inspect" in workflow
        assert "Call" in workflow
    
    def test_generate_commands_section(self):
        """Test generating commands section."""
        gen = AgentMarkdownGenerator()
        commands = gen.generate_commands_section()
        
        assert "Verfügbare Befehle" in commands
        assert "Tool Discovery" in commands
        assert "Tool Execution" in commands
        assert "System Management" in commands
    
    def test_generate_error_handling_section(self):
        """Test generating error handling section."""
        gen = AgentMarkdownGenerator()
        errors = gen.generate_error_handling_section()
        
        assert "Error Handling" in errors
        assert "Tool Not Found" in errors
        assert "Connection Failed" in errors
        assert "Invalid Parameters" in errors
    
    def test_generate_server_section_empty(self):
        """Test generating server section with no tools."""
        gen = AgentMarkdownGenerator()
        section = gen.generate_server_section("empty_server", tools=[])
        
        assert "empty_server" in section
        assert "No tools available" in section
    
    def test_generate_server_section_with_tools(self):
        """Test generating server section with tools."""
        tools = [
            ToolInfo(name="t1", description="Tool 1"),
            ToolInfo(name="t2", description="Tool 2"),
        ]
        gen = AgentMarkdownGenerator()
        section = gen.generate_server_section("test_server", tools=tools)
        
        assert "test_server" in section
        assert "Tool 1" in section
        assert "Tool 2" in section
    
    def test_generate_server_section_max_tools(self):
        """Test max_tools limit in server section."""
        tools = [
            ToolInfo(name=f"tool_{i}", description=f"Tool {i}")
            for i in range(15)
        ]
        gen = AgentMarkdownGenerator()
        section = gen.generate_server_section("server", tools=tools, max_tools=10)
        
        # Should show only 10 tools
        assert "tool_0" in section
        assert "tool_9" in section
        assert "... and 5 more tools" in section
    
    def test_generate_agent_md_basic(self):
        """Test generating complete AGENT.md."""
        gen = AgentMarkdownGenerator()
        md = gen.generate_agent_md(title="Test Documentation")
        
        assert "Test Documentation" in md
        assert "Wichtig: IMMER Zuerst Suchen!" in md
        assert "Empfohlener Workflow" in md
        assert "Verfügbare Befehle" in md
        assert "Error Handling" in md
        assert len(md) > 1000  # Should be substantial
    
    def test_generate_agent_md_with_servers(self):
        """Test AGENT.md generation with server data."""
        gen = AgentMarkdownGenerator()
        
        for server_name in ["filesystem", "database"]:
            tools = [
                ToolInfo(name=f"{server_name}_t1", description="Tool 1"),
                ToolInfo(name=f"{server_name}_t2", description="Tool 2"),
            ]
            server = ServerInfo(name=server_name, tool_count=len(tools))
            gen.add_server(server)
            for tool in tools:
                gen.add_tool_to_server(server_name, tool)
        
        md = gen.generate_agent_md(include_servers=True)
        
        assert "filesystem" in md
        assert "database" in md
        assert "2 Servers" in md or "Total:" in md
    
    def test_generate_agent_md_without_quick_ref(self):
        """Test AGENT.md without quick reference."""
        gen = AgentMarkdownGenerator()
        md = gen.generate_agent_md(include_quick_ref=False)
        
        # Should still have main content
        assert "Empfohlener Workflow" in md
        assert "Verfügbare Befehle" in md
    
    def test_generate_and_save(self):
        """Test generating and saving to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "AGENT.md"
            
            gen = AgentMarkdownGenerator()
            tool = ToolInfo(name="test", description="Test tool")
            gen.add_tool_to_server("test_server", tool)
            
            success = gen.generate_and_save(output_path)
            
            assert success is True
            assert output_path.exists()
            
            content = output_path.read_text()
            assert len(content) > 0
            assert "test_server" in content
    
    def test_save_to_file_creates_parent_directory(self):
        """Test that save_to_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "deep" / "dir" / "AGENT.md"
            
            gen = AgentMarkdownGenerator()
            success = gen.save_to_file("Test content", output_path)
            
            assert success is True
            assert output_path.exists()
            assert output_path.parent.exists()
    
    def test_save_to_file_content(self):
        """Test that save_to_file writes correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.md"
            test_content = "# Test\n\nThis is test content."
            
            gen = AgentMarkdownGenerator()
            gen.save_to_file(test_content, output_path)
            
            saved_content = output_path.read_text()
            assert saved_content == test_content


class TestHelperFunctions:
    """Test convenience helper functions."""
    
    def test_quick_agent_md(self):
        """Test quick_agent_md helper function."""
        server_tools = {
            "server1": [
                ("tool1", "Description 1"),
                ("tool2", "Description 2"),
            ],
            "server2": [
                ("tool3", "Description 3"),
            ],
        }
        
        md = quick_agent_md(server_tools)
        
        assert "server1" in md
        assert "server2" in md
        assert "tool1" in md
        assert "Description 1" in md
        assert len(md) > 1000
    
    def test_create_agent_md_from_registry(self):
        """Test create_agent_md_from_registry function."""
        servers = {
            "fs": [
                ToolInfo(name="read", description="Read file"),
                ToolInfo(name="write", description="Write file"),
            ],
            "db": [
                ToolInfo(name="query", description="Execute query"),
            ],
        }
        
        md = create_agent_md_from_registry(servers)
        
        assert "fs" in md
        assert "db" in md
        assert "read" in md
        assert "query" in md
    
    def test_create_agent_md_from_registry_with_save(self):
        """Test create_agent_md_from_registry with file save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "AGENT.md"
            servers = {
                "test": [
                    ToolInfo(name="test_tool", description="Test"),
                ]
            }
            
            md = create_agent_md_from_registry(servers, output_path)
            
            assert output_path.exists()
            saved_content = output_path.read_text()
            assert saved_content == md


class TestMarkdownContent:
    """Test the actual markdown content quality."""
    
    def test_markdown_has_proper_structure(self):
        """Test that generated markdown has proper structure."""
        gen = AgentMarkdownGenerator()
        md = gen.generate_agent_md()
        
        # Check for proper markdown structure
        assert md.startswith("#")  # Starts with heading
        assert "##" in md  # Has subheadings
        assert "```" in md  # Has code blocks
        assert "\n" in md  # Has line breaks
    
    def test_markdown_has_warnings(self):
        """Test that markdown includes important warnings."""
        gen = AgentMarkdownGenerator()
        md = gen.generate_agent_md()
        
        assert "IMMER" in md  # German: "always"
        assert "Suchen" in md  # German: "search"
        assert "⚠️" in md or "Wichtig" in md  # Warning or important
    
    def test_markdown_is_valid_utf8(self):
        """Test that markdown is valid UTF-8."""
        gen = AgentMarkdownGenerator()
        md = gen.generate_agent_md()
        
        # Should encode/decode without errors
        encoded = md.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert decoded == md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
