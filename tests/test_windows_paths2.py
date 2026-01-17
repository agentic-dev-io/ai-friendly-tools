"""Test Windows path handling - single connection."""

import duckdb
import os

# Create a fresh connection
conn = duckdb.connect(':memory:')
conn.execute("INSTALL duckdb_mcp FROM community")
conn.execute("LOAD duckdb_mcp")

print("Testing Windows paths with colon delimiter:\n")

# Test with real paths
paths = [
    "C:/Users/bjoern/.bun/bin/bunx.exe",
    "C:/Program Files/Docker/Docker/resources/bin/docker.exe",
]

paths_str = ":".join(paths)
print(f"Setting: {paths_str}\n")

try:
    conn.execute(f"SET allowed_mcp_commands = '{paths_str}'")
    result = conn.execute("SELECT current_setting('allowed_mcp_commands')").fetchone()
    print(f"✓ Set successfully!")
    print(f"  Stored as: {result[0]}")
    print()
    
    # Check how it's parsed
    stored_commands = result[0].split(":")
    print(f"Number of commands parsed: {len(stored_commands)}")
    for i, cmd in enumerate(stored_commands, 1):
        print(f"  {i}. '{cmd}'")
        
except Exception as e:
    print(f"✗ ERROR: {e}")

# Test ATTACH with these settings
print("\n\nTesting ATTACH statement:")
try:
    attach_sql = """
        ATTACH 'C:/Users/bjoern/.bun/bin/bunx.exe' AS test_server (
            TYPE mcp,
            TRANSPORT 'stdio',
            ARGS '["bunx.exe", "-y", "@upstash/context7-mcp"]'
        )
    """
    conn.execute(attach_sql)
    print("✓ ATTACH worked!")
except Exception as e:
    print(f"ATTACH error: {e}")

