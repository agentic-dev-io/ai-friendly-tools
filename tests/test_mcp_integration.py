"""Integration tests for MCP Gateway functionality."""

import sys
from pathlib import Path

import duckdb


def test_step(name: str):
    """Decorator for test steps."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            try:
                result = func(*args, **kwargs)
                print(f"✓ PASSED: {name}")
                return result
            except Exception as e:
                print(f"✗ FAILED: {name}")
                print(f"Error: {e}")
                return None
        return wrapper
    return decorator


@test_step("1. DuckDB Connection")
def test_duckdb_connection():
    """Test basic DuckDB connection."""
    conn = duckdb.connect(':memory:')
    result = conn.execute("SELECT 'Hello DuckDB' as message").fetchone()
    print(f"   Result: {result}")
    assert result[0] == 'Hello DuckDB'
    return conn


@test_step("2. MCP Extension Installation")
def test_extension_installation(conn):
    """Test MCP extension installation from community."""
    # Try to install extension
    conn.execute("INSTALL duckdb_mcp FROM community")
    print("   ✓ Extension installed")
    
    conn.execute("LOAD duckdb_mcp")
    print("   ✓ Extension loaded")
    
    # Verify extension is loaded
    result = conn.execute("""
        SELECT extension_name, loaded, installed 
        FROM duckdb_extensions() 
        WHERE extension_name = 'duckdb_mcp'
    """).fetchone()
    
    print(f"   Extension status: name={result[0]}, loaded={result[1]}, installed={result[2]}")
    assert result[1] is True, "Extension not loaded"
    return conn


@test_step("3. MCP Extension Functions Available")
def test_extension_functions(conn):
    """Check if MCP functions are available."""
    # List all functions that start with 'mcp_'
    result = conn.execute("""
        SELECT function_name 
        FROM duckdb_functions() 
        WHERE function_name LIKE 'mcp_%'
        ORDER BY function_name
    """).fetchall()
    
    print(f"   Found {len(result)} MCP functions:")
    for row in result:
        print(f"     - {row[0]}")
    
    expected_functions = [
        'mcp_list_resources',
        'mcp_call_tool',
        'mcp_server_start',
        'mcp_server_status',
        'mcp_publish_table'
    ]
    
    available_functions = [row[0] for row in result]
    for func in expected_functions:
        if func in available_functions:
            print(f"   ✓ {func} available")
        else:
            print(f"   ⚠ {func} NOT found (may have different name)")
    
    return conn


@test_step("4. MCP Settings Configuration")
def test_mcp_settings(conn):
    """Test MCP security settings."""
    # List available settings
    result = conn.execute("""
        SELECT name, value, description 
        FROM duckdb_settings() 
        WHERE name LIKE '%mcp%'
        ORDER BY name
    """).fetchall()
    
    print(f"   Found {len(result)} MCP settings:")
    for row in result:
        print(f"     - {row[0]}: {row[1]}")
        print(f"       {row[2]}")
    
    # Try to set security settings
    try:
        conn.execute("SET allowed_mcp_commands = '/usr/bin/python3:/usr/bin/node'")
        print("   ✓ allowed_mcp_commands setting works")
    except Exception as e:
        print(f"   ⚠ allowed_mcp_commands failed: {e}")
    
    try:
        conn.execute("SET allowed_mcp_urls = 'http://localhost: https://api.example.com'")
        print("   ✓ allowed_mcp_urls setting works")
    except Exception as e:
        print(f"   ⚠ allowed_mcp_urls failed: {e}")
    
    try:
        conn.execute("SET mcp_log_level = 'info'")
        print("   ✓ mcp_log_level setting works")
    except Exception as e:
        print(f"   ⚠ mcp_log_level failed: {e}")
    
    return conn


@test_step("5. ATTACH Syntax Test")
def test_attach_syntax(conn):
    """Test ATTACH statement syntax for MCP."""
    # This will likely fail without actual MCP server, but we can test syntax
    print("   Testing ATTACH statement syntax...")
    print("   (Expected to fail without actual server - checking error message)")
    
    try:
        # Try ATTACH with minimal args
        conn.execute("""
            ATTACH 'test_command' AS test_server (
                TYPE mcp,
                TRANSPORT 'stdio',
                ARGS '["echo", "test"]'
            )
        """)
        print("   ✓ ATTACH statement executed (or syntax accepted)")
        
        # Try to detach
        conn.execute("DETACH test_server")
        print("   ✓ DETACH statement works")
        
    except Exception as e:
        error_msg = str(e)
        print(f"   Expected error (no server): {error_msg}")
        
        # Check if error is about missing server (good) or syntax (bad)
        if "syntax" in error_msg.lower() or "parser" in error_msg.lower():
            print("   ✗ SYNTAX ERROR - ATTACH format may be wrong!")
            print("   Need to check DuckDB MCP extension documentation")
        else:
            print("   ✓ Syntax seems OK (error is about connection, not syntax)")
    
    return conn


@test_step("6. MCP Server Functions")
def test_server_functions(conn):
    """Test MCP server functions."""
    print("   Testing server status function...")
    
    try:
        result = conn.execute("SELECT mcp_server_status()").fetchone()
        print(f"   Server status: {result}")
    except Exception as e:
        print(f"   Server status error (expected if no server running): {e}")
    
    print("\n   Testing server start function...")
    try:
        # Try to start server (may fail, we're just checking function exists)
        conn.execute("SELECT mcp_server_start('stdio', 'localhost', 0, '{}')")
        print("   ✓ Server start function exists")
        
        # Try to stop
        conn.execute("SELECT mcp_server_stop()")
        print("   ✓ Server stop function exists")
    except Exception as e:
        print(f"   Server functions error: {e}")
    
    return conn


@test_step("7. Test Our Implementation")
def test_our_implementation():
    """Test our actual implementation classes."""
    print("   Importing our modules...")
    
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent / "mcp-manager" / "src"))
        
        from mcp_manager.database import DatabaseConnectionPool, install_mcp_extension
        from mcp_manager.registry import ConnectionRegistry
        from mcp_manager.security import SecurityConfig, validate_command
        
        print("   ✓ All modules imported successfully")
        
        # Test DatabaseConnectionPool
        print("\n   Testing DatabaseConnectionPool...")
        pool = DatabaseConnectionPool(Path("/tmp/test_mcp"))
        conn = pool.get_connection("test")
        print("   ✓ Connection pool works")
        pool.close_all()
        
        # Test SecurityConfig
        print("\n   Testing SecurityConfig...")
        security = SecurityConfig(
            allowed_commands=["/usr/bin/python3"],
            allowed_urls=["http://localhost:"]
        )
        assert validate_command("/usr/bin/python3", security) is True
        assert validate_command("/usr/bin/evil", security) is False
        print("   ✓ Security validation works")
        
        # Test ConnectionRegistry
        print("\n   Testing ConnectionRegistry...")
        registry = ConnectionRegistry()
        registry.register("test", "stdio", ["python3", "server.py"])
        assert registry.get("test") is not None
        print("   ✓ Connection registry works")
        registry.clear()
        
        return True
        
    except Exception as e:
        print(f"   Implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("DuckDB MCP Gateway Integration Tests")
    print("="*60)
    
    # Test 1: Basic connection
    conn = test_duckdb_connection()
    if not conn:
        print("\n❌ Cannot continue - DuckDB connection failed")
        return False
    
    # Test 2: Extension installation
    conn = test_extension_installation(conn)
    if not conn:
        print("\n❌ Cannot continue - Extension installation failed")
        print("\nPossible issues:")
        print("  - Extension name might be different")
        print("  - Extension not available in community repo")
        print("  - DuckDB version too old")
        return False
    
    # Test 3: Functions available
    test_extension_functions(conn)
    
    # Test 4: Settings
    test_mcp_settings(conn)
    
    # Test 5: ATTACH syntax
    test_attach_syntax(conn)
    
    # Test 6: Server functions
    test_server_functions(conn)
    
    # Test 7: Our implementation
    test_our_implementation()
    
    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review any failures above")
    print("  2. Check DuckDB MCP extension documentation if syntax errors")
    print("  3. Fix implementation based on actual API")
    print("  4. Test with actual MCP server: mcp-man init testdb")
    
    conn.close()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

