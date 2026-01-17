"""Test Windows path handling in DuckDB MCP extension."""

import duckdb

# Test different path formats
test_cases = [
    ("bunx.exe", "Just filename"),
    ("C:/Users/bjoern/.bun/bin/bunx.exe", "Forward slashes"),
    ("C:\\Users\\bjoern\\.bun\\bin\\bunx.exe", "Backslashes (escaped)"),
]

conn = duckdb.connect(':memory:')
conn.execute("INSTALL duckdb_mcp FROM community")
conn.execute("LOAD duckdb_mcp")

print("Testing different path formats for allowed_mcp_commands:\n")

for path, description in test_cases:
    try:
        conn.execute(f"SET allowed_mcp_commands = '{path}'")
        result = conn.execute("SELECT current_setting('allowed_mcp_commands')").fetchone()
        print(f"✓ {description:30} -> {result[0]}")
    except Exception as e:
        print(f"✗ {description:30} -> ERROR: {e}")

# Test multiple paths
print("\n\nTesting multiple paths with different delimiters:\n")

multi_tests = [
    ("bunx.exe:python.exe", "Colon delimiter"),
    ("bunx.exe;python.exe", "Semicolon delimiter"),
    ("bunx.exe,python.exe", "Comma delimiter"),
]

conn2 = duckdb.connect(':memory:')
conn2.execute("INSTALL duckdb_mcp FROM community")
conn2.execute("LOAD duckdb_mcp")

for paths, description in multi_tests:
    try:
        conn2.execute(f"SET allowed_mcp_commands = '{paths}'")
        result = conn2.execute("SELECT current_setting('allowed_mcp_commands')").fetchone()
        print(f"✓ {description:25} -> {result[0]}")
    except Exception as e:
        print(f"✗ {description:25} -> ERROR: {e}")

