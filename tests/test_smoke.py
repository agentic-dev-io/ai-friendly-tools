"""Quick smoke test for MCP Gateway."""

import duckdb


def main():
    """Quick smoke test."""
    print("🔥 SMOKE TEST - DuckDB MCP Extension\n")
    
    # Test 1: DuckDB works
    print("1. Testing DuckDB connection...")
    conn = duckdb.connect(':memory:')
    result = conn.execute("SELECT 42 as answer").fetchone()
    assert result[0] == 42
    print("   ✓ DuckDB works\n")
    
    # Test 2: Extension exists
    print("2. Checking if MCP extension exists in community...")
    try:
        conn.execute("INSTALL duckdb_mcp FROM community")
        print("   ✓ Extension installed")
    except Exception as e:
        print(f"   ✗ Extension installation failed: {e}")
        print("\n❌ CRITICAL: duckdb_mcp extension not available")
        print("\nPossible reasons:")
        print("  1. Extension doesn't exist yet (it's very new)")
        print("  2. Extension has different name")
        print("  3. Need to use different installation method")
        print("\nCheck: https://duckdb.org/community_extensions/extensions/duckdb_mcp")
        return False
    
    # Test 3: Load extension
    print("3. Loading MCP extension...")
    try:
        conn.execute("LOAD duckdb_mcp")
        print("   ✓ Extension loaded\n")
    except Exception as e:
        print(f"   ✗ Extension load failed: {e}\n")
        return False
    
    # Test 4: Check functions
    print("4. Checking available MCP functions...")
    funcs = conn.execute("""
        SELECT function_name 
        FROM duckdb_functions() 
        WHERE function_name LIKE 'mcp_%'
    """).fetchall()
    
    if funcs:
        print(f"   ✓ Found {len(funcs)} MCP functions:")
        for func in funcs[:5]:  # Show first 5
            print(f"     - {func[0]}")
        if len(funcs) > 5:
            print(f"     ... and {len(funcs) - 5} more")
    else:
        print("   ⚠ No MCP functions found")
    
    print("\n" + "="*60)
    print("✅ SMOKE TEST PASSED - Extension is available!")
    print("="*60)
    print("\nYou can now run: python tests/test_mcp_integration.py")
    
    conn.close()
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

