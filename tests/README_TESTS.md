# MCP Gateway Integration Tests

## Quick Start

### 1. Smoke Test (30 seconds)

Schnellster Test - prüft nur ob die Extension existiert:

```bash
cd tests
python test_smoke.py
```

**Erwartetes Ergebnis:**
- ✅ Extension installiert und geladen
- Liste der verfügbaren MCP-Funktionen

**Falls fehlschlägt:**
- Extension existiert möglicherweise noch nicht
- Name der Extension ist anders
- DuckDB Version prüfen

### 2. Full Integration Test (2-3 Minuten)

Umfassender Test aller Funktionen:

```bash
python test_mcp_integration.py
```

**Was wird getestet:**
1. ✓ DuckDB Basis-Verbindung
2. ✓ MCP Extension Installation
3. ✓ Verfügbare MCP-Funktionen
4. ✓ Security Settings (allowed_commands, allowed_urls)
5. ✓ ATTACH Syntax für MCP-Server
6. ✓ Server-Funktionen (start, stop, status)
7. ✓ Unsere Implementation (Database, Security, Registry)

### 3. CLI Test

Nach erfolgreichen Integration Tests:

```bash
# Installation check
uv sync

# Einfachster CLI Test
mcp-man --help

# Datenbank initialisieren
mcp-man init testdb

# Security konfigurieren
mcp-man security add-command /usr/bin/python3
mcp-man security list

# Gateway Status
mcp-man gateway status
```

## Erwartete Probleme und Lösungen

### Problem 1: Extension nicht gefunden

```
✗ Extension installation failed: Extension 'duckdb_mcp' not found
```

**Ursache:** Extension noch nicht verfügbar oder anderer Name

**Lösung:**
1. DuckDB Version prüfen: `python -c "import duckdb; print(duckdb.__version__)"`
2. Community Extensions auflisten: 
   ```python
   import duckdb
   conn = duckdb.connect()
   conn.execute("SELECT * FROM duckdb_extensions() WHERE installed = false").fetchall()
   ```
3. Dokumentation prüfen: https://duckdb.org/community_extensions/extensions/duckdb_mcp

### Problem 2: ATTACH Syntax falsch

```
✗ SYNTAX ERROR - ATTACH format may be wrong!
```

**Ursache:** ATTACH Syntax für MCP ist anders als erwartet

**Lösung:**
- Extension-Dokumentation lesen für korrekte ATTACH-Syntax
- Unsere `gateway.py` und `client.py` anpassen

### Problem 3: Funktionen haben andere Namen

```
⚠ mcp_list_resources NOT found
```

**Ursache:** Funktionen haben andere Namen als in Dokumentation

**Lösung:**
1. Alle verfügbaren Funktionen listen:
   ```python
   conn.execute("SELECT function_name FROM duckdb_functions() WHERE function_name LIKE 'mcp_%'")
   ```
2. Mapping in unserer Implementation korrigieren

### Problem 4: Settings nicht verfügbar

```
⚠ allowed_mcp_commands failed
```

**Ursache:** Settings haben andere Namen oder andere Syntax

**Lösung:**
1. Alle MCP-Settings listen:
   ```python
   conn.execute("SELECT * FROM duckdb_settings() WHERE name LIKE '%mcp%'")
   ```
2. `security.py` entsprechend anpassen

## Test-Strategie

```
test_smoke.py               → Schneller Check (30s)
         ↓ PASS
test_mcp_integration.py     → Detaillierte Tests (2-3min)
         ↓ PASS
mcp-man init testdb         → CLI Test
         ↓ PASS
Echten MCP Server testen    → Production Test
```

## Debugging

### DuckDB Version prüfen

```python
import duckdb
print(f"DuckDB Version: {duckdb.__version__}")
```

Mindestens Version **0.10.0** empfohlen.

### Extension-Details anzeigen

```python
import duckdb
conn = duckdb.connect()
conn.execute("INSTALL duckdb_mcp FROM community")

# Extension-Info
result = conn.execute("""
    SELECT * FROM duckdb_extensions() 
    WHERE extension_name = 'duckdb_mcp'
""").fetchone()
print(f"Extension: {result}")

# Alle MCP-Funktionen
funcs = conn.execute("""
    SELECT function_name, function_type, return_type
    FROM duckdb_functions() 
    WHERE function_name LIKE 'mcp_%'
""").fetchall()

for func in funcs:
    print(f"{func[0]}: {func[1]} -> {func[2]}")
```

### Logging aktivieren

Falls Tests fehlschlagen, mehr Details ausgeben:

```python
import duckdb
conn = duckdb.connect()
conn.execute("SET enable_progress_bar = true")
conn.execute("SET enable_profiling = true")
```

## Nach erfolgreichen Tests

1. **Implementation korrigieren** basierend auf tatsächlicher API
2. **README.md aktualisieren** mit korrekten Beispielen
3. **CLI Commands testen** mit echtem MCP Server
4. **Dokumentation schreiben** für gefundene Edge Cases

## Hilfe

Falls Tests fehlschlagen:
1. Output der Tests kopieren
2. Zeigen welcher Test fehlschlägt
3. Wir analysieren zusammen und fixen die Implementation

