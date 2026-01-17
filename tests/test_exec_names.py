"""Test using just executable names in PATH."""

import duckdb

conn = duckdb.connect(':memory:')
conn.execute("INSTALL duckdb_mcp FROM community")
conn.execute("LOAD duckdb_mcp")

print("Testing with executable names only:\n")

# Use just names, not full paths
commands = "bunx.exe:docker.exe:serena.exe:opencode-mcp.exe"
print(f"Setting: {commands}\n")

conn.execute(f"SET allowed_mcp_commands = '{commands}'")
result = conn.execute("SELECT current_setting('allowed_mcp_commands')").fetchone()
print(f"✓ Stored as: {result[0]}")
print()

# Parse
stored = result[0].split(":")
print(f"Parsed commands: {len(stored)}")
for i, cmd in enumerate(stored, 1):
    print(f"  {i}. '{cmd}'")

# Test ATTACH
print("\n\nTesting ATTACH with bunx.exe:")
try:
    attach_sql = """
        ATTACH 'bunx.exe' AS test_server (
            TYPE mcp,
            TRANSPORT 'stdio',
            ARGS '["bunx.exe", "-y", "@upstash/context7-mcp"]'
        )
    """
    conn.execute(attach_sql)
    print("✓ ATTACH syntax accepted!")
except Exception as e:
    error_msg = str(e)
    if "not allowed" in error_msg.lower():
        print(f"✗ Command not allowed: {error_msg}")
    elif "syntax" in error_msg.lower():
        print(f"✗ SYNTAX ERROR: {error_msg}")
    else:
        print(f"Connection error (expected without real server): {error_msg[:200]}...")

