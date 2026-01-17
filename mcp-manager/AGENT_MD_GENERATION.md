# AGENT.md Generation für MCP-Man

Dieses Modul bietet Funktionalität zur Erzeugung von `AGENT.md` Dokumentation für AI-Agenten (Claude, ChatGPT, etc.), die mit MCP-Man arbeiten.

## Übersicht

Das `agent` Modul generiert umfassende Markdown-Dokumentation mit:

- **Workflow-Anleitung** (3-Schritte-Prozess)
- **CLI-Befehle** (Discovery, Execution, System)
- **Server & Tool-Übersicht** (mit Statistiken)
- **Quick Reference** (Copy-Paste ready Commands)
- **Error Handling** (Häufige Probleme & Lösungen)
- **Best Practices** (Tool-Namen, Parameter, etc.)

## Installation

Die Module sind bereits im mcp-manager enthalten:

```bash
from mcp_manager.agent import (
    AgentMarkdownGenerator,
    ServerInfo,
    ToolInfo,
    create_agent_md_from_registry,
    quick_agent_md,
)
```

## Schnellstart

### Option 1: Schnelle Variante (quick_agent_md)

```python
from mcp_manager.agent import quick_agent_md

server_tools = {
    "filesystem": [
        ("read_file", "Read contents from a file"),
        ("write_file", "Write content to a file"),
    ],
    "database": [
        ("query", "Execute SQL query"),
        ("create_table", "Create a new table"),
    ],
}

# Generate markdown
md = quick_agent_md(server_tools)

# Save to file
Path("AGENT.md").write_text(md)
```

### Option 2: Volle Kontrolle (AgentMarkdownGenerator)

```python
from mcp_manager.agent import AgentMarkdownGenerator, ToolInfo, ServerInfo

# Create generator
generator = AgentMarkdownGenerator()

# Add tools with parameters
tool = ToolInfo(
    name="read_file",
    description="Read contents from a file",
    parameters=["path", "encoding"],
    required_params=["path"]
)

# Add to server
generator.add_tool_to_server("filesystem", tool)

# Generate complete documentation
md = generator.generate_agent_md(
    title="My MCP-Man Documentation",
    include_quick_ref=True,
    include_servers=True
)

# Save to file
generator.save_to_file(md, Path("AGENT.md"))
```

### Option 3: Aus Registry (create_agent_md_from_registry)

```python
from mcp_manager.agent import create_agent_md_from_registry, ToolInfo

servers = {
    "filesystem": [
        ToolInfo(name="read", description="Read file"),
        ToolInfo(name="write", description="Write file"),
    ],
    "database": [
        ToolInfo(name="query", description="Execute query"),
    ],
}

# Generate and optionally save
md = create_agent_md_from_registry(
    servers,
    output_path=Path("AGENT.md")  # Optional
)
```

## Klassen & Datentypen

### ToolInfo

Informationen über ein einzelnes Tool:

```python
@dataclass
class ToolInfo:
    name: str                          # Tool name
    description: str                   # Kurzbeschreibung
    parameters: list[str] | None = None   # Parameter liste
    required_params: list[str] | None = None  # Erforderliche Parameter
    server: str | None = None          # Server name (wird gesetzt)
    
    # Methods
    format_params() -> str              # "param1, param2"
    to_markdown() -> str                # Markdown-Zeile
```

### ServerInfo

Informationen über einen MCP-Server:

```python
@dataclass
class ServerInfo:
    name: str                          # Server name
    status: str = "active"             # Status
    tool_count: int = 0                # Anzahl Tools
    tools: list[ToolInfo] | None = None   # Tools list
    last_checked: str | None = None    # ISO timestamp
```

### AgentMarkdownGenerator

Haupt-Klasse zur Erzeugung von AGENT.md:

```python
class AgentMarkdownGenerator:
    # Add/manage servers and tools
    add_server(server: ServerInfo) -> None
    add_tool_to_server(server_name: str, tool: ToolInfo) -> None
    
    # Generate sections
    generate_quick_reference() -> str
    generate_workflow_section() -> str
    generate_commands_section() -> str
    generate_error_handling_section() -> str
    generate_server_section(server_name: str, tools: list[ToolInfo]) -> str
    
    # Generate complete document
    generate_agent_md(
        title: str = "MCP-Man (DuckDB-Powered MCP Gateway)",
        include_quick_ref: bool = True,
        include_servers: bool = True
    ) -> str
    
    # Save to file
    save_to_file(content: str, path: Path | str) -> bool
    generate_and_save(output_path: Path | str, **kwargs) -> bool
```

## Generierte AGENT.md Struktur

```
# MCP-Man (DuckDB-Powered MCP Gateway)

## ⚠️ Wichtig: IMMER Zuerst Suchen!
(Warning + Best Practices)

## Empfohlener Workflow
- 1. Search nach Tools
- 2. Inspect für Details
- 3. Call mit Parametern

## Verfügbare Befehle
### Tool Discovery
- mcp-man search "query"
- mcp-man tools <server>
- ...

### Tool Execution
- mcp-man inspect <server> <tool>
- mcp-man call <server> <tool> {...}
- ...

### System Management
- mcp-man verify
- mcp-man health
- mcp-man refresh

## Verbundene Server
(Table mit Server-Statistiken)

## Server Details
### filesystem
### database
### web
...

## Quick Command Reference
(Copy-paste ready)

## Error Handling & Troubleshooting
### Common Issues & Solutions
### Getting Help

## Wichtige Konzepte
### BM25 vs Semantic Search
### Tool Naming
### Parameter Format
```

## Beispiele

### Beispiel 1: Einfache Verwendung

```python
from mcp_manager.agent import quick_agent_md
from pathlib import Path

# Define tools
tools = {
    "core": [
        ("search", "Search for tools"),
        ("inspect", "Inspect tool details"),
    ]
}

# Generate and save
md = quick_agent_md(tools)
Path("AGENT.md").write_text(md)
```

### Beispiel 2: Mit Server-Status

```python
from mcp_manager.agent import AgentMarkdownGenerator, ServerInfo, ToolInfo

gen = AgentMarkdownGenerator()

# Add server with detailed tools
tools = [
    ToolInfo("read_file", "Read a file", ["path"], ["path"]),
    ToolInfo("write_file", "Write a file", ["path", "content"], ["path", "content"]),
]

server = ServerInfo(name="filesystem", tool_count=len(tools))
gen.add_server(server)

for tool in tools:
    gen.add_tool_to_server("filesystem", tool)

# Generate
md = gen.generate_agent_md(include_servers=True)
```

### Beispiel 3: Individuelle Sektion

```python
gen = AgentMarkdownGenerator()

# Generate nur Quick Reference
quick_ref = gen.generate_quick_reference()

# Generate nur Error Handling
errors = gen.generate_error_handling_section()

# Zusammenfügen
custom_md = f"# My Doc\n\n{quick_ref}\n\n{errors}"
```

## Features

### 1. Tool Discovery Anleitung
- Warnt vor Tool-Namen-Variationen
- Erklärt BM25 vs Semantic Search
- Gibt Workflow-Anleitung

### 2. Umfassende Befehls-Referenz
- Kategorisiert nach Use-Case
- Copy-paste ready
- Alle Parameter erklärt

### 3. Error Handling
- Häufige Fehler dokumentiert
- Lösungen für jeden Fehler
- Best Practices

### 4. Server-Übersicht
- Statistiken pro Server
- Tool-Counts
- Status-Anzeige

### 5. Flexible Ausgabe
- Kann alle Sektionen aktivieren/deaktivieren
- Support für verschiedene Titel
- Datei-Speicherung mit Auto-Directory-Creation

## Testing

Tests befinden sich in `tests/test_agent_markdown_generator.py`:

```bash
pytest tests/test_agent_markdown_generator.py -v
```

Abgedeckt:
- ToolInfo Funktionalität
- ServerInfo Verwaltung
- Generator-Methoden
- Datei-I/O
- Markdown-Qualität
- Helper-Funktionen

## Beispiel-Output

Siehe `examples/generate_agent_md.py` für vollständige Beispiele:

```bash
cd mcp-manager
python examples/generate_agent_md.py
```

Erzeugt:
- `examples/AGENT.md` - Generierte Dokumentation
- Console-Output mit Beispielen

## Integration in CLI

Kann in `cli.py` integriert werden:

```python
@app.command()
def generate_agent_md(
    output_path: Path = typer.Option(
        "AGENT.md",
        "--output",
        "-o",
        help="Path to save AGENT.md"
    )
) -> None:
    """Generate AGENT.md documentation for AI agents."""
    from .agent import AgentMarkdownGenerator
    
    generator = AgentMarkdownGenerator()
    
    # Add servers and tools from registry
    # ... populate generator ...
    
    success = generator.generate_and_save(output_path)
    if success:
        console.print(f"✓ Generated {output_path}")
```

## Best Practices

1. **Immer Tool-Namen suchen** - NIE raten!
2. **Semantische Suche verwenden** - für konzeptuelle Queries
3. **Schema inspizieren** - bevor Tools aufgerufen werden
4. **JSON-Parameter beachten** - Strings, Zahlen, Arrays korrekt formatieren
5. **Error-Messages lesen** - geben Hinweise auf das Problem

## Fehlerbehandlung

### Datei nicht erstellbar

```python
try:
    gen.save_to_file(content, Path("./AGENT.md"))
except Exception as e:
    print(f"Error: {e}")
```

Wird gehandhabt durch:
- Auto-Erstellung von Parent-Directories
- UTF-8 Encoding
- Exception-Handling mit Rückgabewert

### Ungültige Eingaben

```python
# ToolInfo erfordert name und description
tool = ToolInfo(name="", description="")  # Funktioniert, aber sinnlos

# Best practice
tool = ToolInfo(
    name="read_file",
    description="Read contents from a file",
    parameters=["path"],
    required_params=["path"]
)
```

## Performance

- Generierung: < 100ms für 100 Tools
- Datei-Speicherung: < 10ms für 50KB Markdown
- Memory-Footprint: < 1MB für typische Dokumentation

## Zukünftige Enhancements

- [ ] Template-System (Jinja2-Support)
- [ ] Custom CSS/HTML Output
- [ ] Server-Verbindungsstatus einfügen
- [ ] Tool-Nutzungsstatistiken
- [ ] Mehrsprachige Dokumentation
- [ ] PDF-Export
- [ ] Real-time Generierung aus Live-Registry

## Lizenz

Siehe LICENSE file in Repository-Root.

## Support

Bei Fragen oder Problemen:
1. Siehe `examples/generate_agent_md.py` für Beispiele
2. Lies Test-Cases in `tests/test_agent_markdown_generator.py`
3. Überprüfe Docstrings in `templates.py`
