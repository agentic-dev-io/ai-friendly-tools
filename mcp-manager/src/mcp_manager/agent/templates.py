"""
Agent Markdown Generator for MCP-Man Documentation

Generates AGENT.md markdown documentation for Claude and other AI agents,
providing comprehensive workflow instructions, CLI commands, and tool information.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime


@dataclass
class ToolInfo:
    """Information about a single MCP tool."""
    
    name: str
    description: str
    parameters: list[str] | None = None
    required_params: list[str] | None = None
    server: str | None = None
    
    def format_params(self) -> str:
        """Format parameters for display."""
        if not self.parameters:
            return ""
        return ", ".join(self.parameters)
    
    def to_markdown(self) -> str:
        """Convert tool info to markdown line."""
        params = self.format_params()
        if params:
            return f"- **{self.name}**: {self.description} `params: {params}`"
        return f"- **{self.name}**: {self.description}"


@dataclass
class ServerInfo:
    """Information about an MCP server."""
    
    name: str
    status: str = "active"
    tool_count: int = 0
    tools: list[ToolInfo] | None = None
    last_checked: str | None = None
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.last_checked is None:
            self.last_checked = datetime.now().isoformat()


class AgentMarkdownGenerator:
    """Generate AGENT.md documentation for AI agents using MCP-Man."""
    
    def __init__(self):
        """Initialize the markdown generator."""
        self.servers: dict[str, ServerInfo] = {}
        self.generated_at = datetime.now().isoformat()
    
    def add_server(self, server: ServerInfo) -> None:
        """Add server information to generator."""
        self.servers[server.name] = server
    
    def add_tool_to_server(self, server_name: str, tool: ToolInfo) -> None:
        """Add a tool to a specific server."""
        if server_name not in self.servers:
            self.servers[server_name] = ServerInfo(name=server_name)
        
        tool.server = server_name
        self.servers[server_name].tools.append(tool)
        self.servers[server_name].tool_count = len(self.servers[server_name].tools)
    
    def generate_server_section(
        self, 
        server_name: str, 
        tools: list[ToolInfo] | None = None,
        max_tools: int = 10
    ) -> str:
        """
        Generate markdown section for a specific server.
        
        Args:
            server_name: Name of the server
            tools: List of tools (if None, uses stored tools)
            max_tools: Maximum number of tools to show
            
        Returns:
            Formatted markdown section for the server
        """
        if tools is None and server_name in self.servers:
            tools = self.servers[server_name].tools[:max_tools]
        
        if not tools:
            return f"### {server_name}\n\nNo tools available.\n"
        
        section = f"### {server_name}\n\n"
        section += f"**Status:** Active | **Available Tools:** {len(tools)}\n\n"
        section += "**Most Important Tools:**\n\n"
        
        for tool in tools[:max_tools]:
            section += f"{tool.to_markdown()}\n"
        
        if len(tools) > max_tools:
            section += f"\n*... and {len(tools) - max_tools} more tools. Use `mcp-man tools {server_name}` to see all.*\n"
        
        section += "\n"
        return section
    
    def generate_quick_reference(self) -> str:
        """
        Generate quick command reference section.
        
        Returns:
            Formatted markdown section with copy-paste ready commands
        """
        reference = "## Quick Command Reference\n\n"
        reference += "Copy-paste ready commands for common tasks:\n\n"
        reference += "```bash\n"
        reference += "# Search for tools\n"
        reference += 'mcp-man search "your query"\n\n'
        reference += "# Search semantically\n"
        reference += 'mcp-man search "your query" --semantic\n\n'
        reference += "# List all tools from a server\n"
        reference += "mcp-man tools <server_name>\n\n"
        reference += "# Inspect a specific tool\n"
        reference += "mcp-man inspect <server_name> <tool_name>\n\n"
        reference += "# Inspect with usage example\n"
        reference += "mcp-man inspect <server_name> <tool_name> --example\n\n"
        reference += "# Execute a tool\n"
        reference += 'mcp-man call <server_name> <tool_name> \'{"param1": "value1"}\'\n\n'
        reference += "# Verify all connections\n"
        reference += "mcp-man verify\n\n"
        reference += "# Check system health\n"
        reference += "mcp-man health\n\n"
        reference += "# Refresh tool index\n"
        reference += "mcp-man refresh\n"
        reference += "```\n\n"
        return reference
    
    def generate_error_handling_section(self) -> str:
        """
        Generate error handling and troubleshooting section.
        
        Returns:
            Formatted markdown section with error handling tips
        """
        section = "## Error Handling & Troubleshooting\n\n"
        
        section += "### Common Issues & Solutions\n\n"
        
        section += "**Tool Not Found Error**\n"
        section += "- Error: `Tool 'xyz' not found on server 'abc'`\n"
        section += "- **Solution:** Tool names vary between servers. Use `mcp-man search` first!\n"
        section += "  ```bash\n"
        section += '  mcp-man search "tool description"\n'
        section += "  ```\n"
        section += "- **Never guess** tool names - always search first\n\n"
        
        section += "**Connection Failed**\n"
        section += "- Error: `Failed to connect to server`\n"
        section += "- **Solution:** Check server status and connections:\n"
        section += "  ```bash\n"
        section += "  mcp-man health\n"
        section += "  mcp-man verify\n"
        section += "  ```\n\n"
        
        section += "**Invalid Parameters**\n"
        section += "- Error: `Invalid parameter type or missing required field`\n"
        section += "- **Solution:** Inspect the tool to see exact schema:\n"
        section += "  ```bash\n"
        section += "  mcp-man inspect <server> <tool> --example\n"
        section += "  ```\n"
        section += "- Use the example output as a template\n\n"
        
        section += "**Search Returns No Results**\n"
        section += "- **Solution:** Try different search terms or use semantic search:\n"
        section += "  ```bash\n"
        section += '  mcp-man search "keyword" --semantic\n'
        section += "  ```\n\n"
        
        section += "**Tool Execution Timeout**\n"
        section += "- Error: `Tool execution exceeded timeout`\n"
        section += "- **Reason:** Tool is taking too long\n"
        section += "- **Solution:** Check tool parameters or try simpler queries\n\n"
        
        section += "### Getting Help\n\n"
        section += "- Always use `mcp-man search` before calling tools\n"
        section += "- Use `mcp-man inspect <server> <tool> --example` for usage examples\n"
        section += "- Run `mcp-man health` to diagnose connection issues\n"
        section += "- Tool names NEVER match between servers - search first!\n\n"
        
        return section
    
    def generate_workflow_section(self) -> str:
        """
        Generate recommended workflow section.
        
        Returns:
            Formatted markdown section with workflow instructions
        """
        section = "## Empfohlener Workflow\n\n"
        section += "Follow this 3-step workflow for reliable tool usage:\n\n"
        
        section += "### 1. Search nach Tools\n\n"
        section += "Never guess tool names! Always search first:\n\n"
        section += "```bash\n"
        section += '# Full-text search (BM25)\n'
        section += 'mcp-man search "what you want to do"\n\n'
        section += "# Semantic search (AI-powered)\n"
        section += 'mcp-man search "your description" --semantic\n'
        section += "```\n\n"
        section += "**Why:** Tool names vary across servers. Searching ensures you find the right tool.\n\n"
        
        section += "### 2. Inspect für Details\n\n"
        section += "Once found, inspect the tool to understand its parameters:\n\n"
        section += "```bash\n"
        section += "mcp-man inspect <server_name> <tool_name>\n"
        section += "mcp-man inspect <server_name> <tool_name> --example  # With example\n"
        section += "```\n\n"
        section += "**What you get:** Full schema, required parameters, and usage examples.\n\n"
        
        section += "### 3. Call mit Parametern\n\n"
        section += "Execute the tool with the correct parameters:\n\n"
        section += "```bash\n"
        section += 'mcp-man call <server_name> <tool_name> \'{"param1": "value1", "param2": "value2"}\'\n'
        section += "```\n\n"
        section += "**Important:** Always use the exact parameter names from the schema.\n\n"
        
        return section
    
    def generate_commands_section(self) -> str:
        """
        Generate available commands section.
        
        Returns:
            Formatted markdown section with all CLI commands
        """
        section = "## Verfügbare Befehle\n\n"
        
        section += "### Tool Discovery\n\n"
        section += '- `mcp-man search "<query>"` - BM25 Volltextsuche über alle Tools\n'
        section += '- `mcp-man search "<query>" --semantic` - Semantische Suche mit AI\n'
        section += "- `mcp-man tools <server>` - Alle Tools eines Servers auflisten\n"
        section += "- `mcp-man tools <server> --json` - JSON output für Parsing\n\n"
        
        section += "### Tool Execution\n\n"
        section += "- `mcp-man inspect <server> <tool>` - Zeige Tool-Schema\n"
        section += "- `mcp-man inspect <server> <tool> --example` - Mit Beispiel\n"
        section += '- `mcp-man call <server> <tool> \'...\' ` - Führe Tool aus\n'
        section += '- `mcp-man call <server> <tool> \'...\' --raw` - Roh-Output\n\n'
        
        section += "### System Management\n\n"
        section += "- `mcp-man verify` - Teste alle Server-Verbindungen\n"
        section += "- `mcp-man health` - Zeige System-Status\n"
        section += "- `mcp-man refresh` - Aktualisiere Tool-Index\n"
        section += "- `mcp-man status` - Verbindungs-Status anzeigen\n\n"
        
        section += "### Configuration\n\n"
        section += "- `mcp-man config show` - Zeige aktuelle Config\n"
        section += "- `mcp-man config update` - Aktualisiere Config\n\n"
        
        return section
    
    def generate_server_overview_section(self) -> str:
        """
        Generate server overview section.
        
        Returns:
            Formatted markdown section with server statistics
        """
        if not self.servers:
            return ""
        
        section = "## Verbundene Server\n\n"
        section += f"**Total:** {len(self.servers)} Servers, "
        section += f"**Total Tools:** {sum(s.tool_count for s in self.servers.values())}\n\n"
        
        section += "| Server | Status | Tools |\n"
        section += "|--------|--------|-------|\n"
        
        for server_name, server in sorted(self.servers.items()):
            section += f"| {server_name} | {server.status} | {server.tool_count} |\n"
        
        section += "\n"
        return section
    
    def generate_agent_md(
        self,
        title: str = "MCP-Man (DuckDB-Powered MCP Gateway)",
        include_quick_ref: bool = True,
        include_servers: bool = True
    ) -> str:
        """
        Generate complete AGENT.md markdown documentation.
        
        Args:
            title: Main title for the document
            include_quick_ref: Include quick reference section
            include_servers: Include server details section
            
        Returns:
            Complete markdown document as string
        """
        md = f"# {title}\n\n"
        
        md += "## ⚠️ Wichtig: IMMER Zuerst Suchen!\n\n"
        md += "Tool-Namen variieren zwischen Servern. **NIE raten** - immer `mcp-man search` nutzen.\n\n"
        md += "```bash\n"
        md += '# Richtig: Tool suchen\n'
        md += 'mcp-man search "was du brauchst"\n\n'
        md += "# Falsch: Tool-Namen raten\n"
        md += "mcp-man call server some_random_tool '{}'\n"
        md += "```\n\n"
        
        md += self.generate_workflow_section()
        md += self.generate_commands_section()
        
        if include_servers and self.servers:
            md += self.generate_server_overview_section()
            md += "## Server Details\n\n"
            for server_name, server in sorted(self.servers.items()):
                md += self.generate_server_section(server_name, server.tools)
        
        if include_quick_ref:
            md += self.generate_quick_reference()
        
        md += self.generate_error_handling_section()
        
        md += "## Wichtige Konzepte\n\n"
        md += "### BM25 vs Semantic Search\n\n"
        md += "- **BM25:** Schnelle Volltextsuche, ideal für genaue Keyword-Matches\n"
        md += "- **Semantic:** AI-powered Suche, besser für konzeptuelle Queries\n\n"
        
        md += "### Tool Naming\n\n"
        md += "⚠️ **Kritisch:** Tool-Namen sind unterschiedlich pro Server!\n\n"
        md += "- Server A könnte `get_data` haben\n"
        md += "- Server B könnte `fetch_information` haben\n"
        md += "- Beide machen das gleiche, heißen aber unterschiedlich\n\n"
        md += "**Immer:** Use `mcp-man search` to find the right tool for your task!\n\n"
        
        md += "### Parameter Format\n\n"
        md += "Tools nehmen Parameter in JSON-Format entgegen:\n\n"
        md += "```bash\n"
        md += 'mcp-man call <server> <tool> \'{"param1": "value1", "param2": 42}\'\n'
        md += "```\n\n"
        md += "- Strings in Anführungszeichen\n"
        md += "- Zahlen ohne Anführungszeichen\n"
        md += "- Booleans: `true` oder `false` (Kleinbuchstaben)\n"
        md += "- Arrays: `[\"item1\", \"item2\"]`\n\n"
        
        md += "---\n\n"
        md += f"*Generated: {self.generated_at}*\n"
        
        return md
    
    def save_to_file(self, content: str, path: Path | str) -> bool:
        """
        Save generated markdown to file.
        
        Args:
            content: Markdown content to save
            path: File path to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(path)
            
            # Create parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            path.write_text(content, encoding="utf-8")
            
            return True
        except Exception as e:
            print(f"Error saving to {path}: {e}")
            return False
    
    def generate_and_save(
        self,
        output_path: Path | str,
        title: str = "MCP-Man (DuckDB-Powered MCP Gateway)",
        **kwargs: Any
    ) -> bool:
        """
        Generate AGENT.md and save to file in one step.
        
        Args:
            output_path: Path to save AGENT.md to
            title: Document title
            **kwargs: Additional arguments for generate_agent_md
            
        Returns:
            True if successful, False otherwise
        """
        md_content = self.generate_agent_md(title=title, **kwargs)
        return self.save_to_file(md_content, output_path)


# Convenience functions for common use cases

def create_agent_md_from_registry(
    servers: dict[str, list[ToolInfo]],
    output_path: Path | str | None = None
) -> str:
    """
    Create AGENT.md from a registry of servers and tools.
    
    Args:
        servers: Dictionary mapping server names to lists of tools
        output_path: Optional path to save the generated markdown
        
    Returns:
        Generated markdown string
    """
    generator = AgentMarkdownGenerator()
    
    for server_name, tools in servers.items():
        server = ServerInfo(name=server_name, tool_count=len(tools))
        generator.add_server(server)
        
        for tool in tools:
            generator.add_tool_to_server(server_name, tool)
    
    md = generator.generate_agent_md()
    
    if output_path:
        generator.save_to_file(md, output_path)
    
    return md


def quick_agent_md(server_tools: dict[str, list[tuple[str, str]]]) -> str:
    """
    Quick helper to generate AGENT.md from simple server/tool data.
    
    Args:
        server_tools: Dict of {server_name: [(tool_name, description), ...]}
        
    Returns:
        Generated markdown string
    """
    servers = {}
    
    for server_name, tools_list in server_tools.items():
        tools = [
            ToolInfo(name=name, description=desc)
            for name, desc in tools_list
        ]
        servers[server_name] = tools
    
    return create_agent_md_from_registry(servers)
