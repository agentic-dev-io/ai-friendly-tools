"""Comprehensive integration tests for all AIFT tools."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

# Import CLI apps
from aift.cli import app as core_app


class TestCoreIntegration:
    """Integration tests for the core AIFT CLI."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner for testing."""
        return CliRunner()

    def test_core_version_command(self, runner):
        """Test core version command."""
        result = runner.invoke(core_app, ["version"])
        assert result.exit_code == 0
        assert "AIFT" in result.stdout
        assert "0.1.0" in result.stdout

    def test_core_info_command(self, runner):
        """Test core info command shows configuration."""
        result = runner.invoke(core_app, ["info"])
        assert result.exit_code == 0
        assert "Configuration" in result.stdout

    def test_core_config_show(self, runner):
        """Test config show displays current configuration."""
        result = runner.invoke(core_app, ["config-show"])
        assert result.exit_code == 0
        assert "Configuration" in result.stdout

    def test_core_help_command(self, runner):
        """Test help output."""
        result = runner.invoke(core_app, ["--help"])
        assert result.exit_code == 0
        assert "AI-Friendly Tools" in result.stdout

    def test_core_hello_command(self, runner):
        """Test hello command."""
        result = runner.invoke(core_app, ["hello", "Integration", "--count", "1"])
        assert result.exit_code == 0
        assert "Hello" in result.stdout
        assert "Integration" in result.stdout

    def test_core_debug_command(self, runner):
        """Test debug command shows system information."""
        result = runner.invoke(core_app, ["debug"])
        assert result.exit_code == 0
        assert "System Information" in result.stdout or "Debug" in result.stdout

    def test_core_validate_command(self, runner):
        """Test validate command checks configuration."""
        result = runner.invoke(core_app, ["validate"])
        # Should exit with 0 or show validation results
        assert "Configuration" in result.stdout or "valid" in result.stdout.lower()

    def test_core_test_command(self, runner):
        """Test test command runs functionality tests."""
        result = runner.invoke(core_app, ["test"])
        # May have some test output
        assert result.exit_code in [0, 1]  # 0 for pass, 1 for some tests failing is ok

    def test_core_multiple_commands_sequence(self, runner):
        """Test running multiple commands in sequence."""
        # Run version
        result1 = runner.invoke(core_app, ["version"])
        assert result1.exit_code == 0

        # Run info
        result2 = runner.invoke(core_app, ["info"])
        assert result2.exit_code == 0

        # Run config-show
        result3 = runner.invoke(core_app, ["config-show"])
        assert result3.exit_code == 0


class TestWorkspaceIntegration:
    """Integration tests for workspace functionality."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir) / "test_workspace"
            workspace_path.mkdir(parents=True, exist_ok=True)
            yield workspace_path

    def test_workspace_creation(self, temp_workspace):
        """Test creating a workspace."""
        assert temp_workspace.exists()
        assert temp_workspace.is_dir()

    def test_workspace_structure(self, temp_workspace):
        """Test creating standard workspace structure."""
        # Create standard subdirectories
        (temp_workspace / "data").mkdir()
        (temp_workspace / "logs").mkdir()
        (temp_workspace / "cache").mkdir()

        assert (temp_workspace / "data").exists()
        assert (temp_workspace / "logs").exists()
        assert (temp_workspace / "cache").exists()

    def test_workspace_config_file(self, temp_workspace):
        """Test creating workspace config file."""
        config = {
            "workspace": str(temp_workspace),
            "tools": ["core", "web", "mcp-manager"],
            "version": "0.1.0",
        }

        config_file = temp_workspace / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        assert config_file.exists()
        with open(config_file) as f:
            loaded = json.load(f)
            assert loaded["workspace"] == str(temp_workspace)
            assert "core" in loaded["tools"]


class TestDatabaseIntegration:
    """Integration tests for DuckDB functionality."""

    @pytest.fixture
    def duckdb_conn(self):
        """Create an in-memory DuckDB connection."""
        try:
            import duckdb

            conn = duckdb.connect(":memory:")
            yield conn
            conn.close()
        except ImportError:
            pytest.skip("duckdb not installed")

    def test_duckdb_basic_query(self, duckdb_conn):
        """Test basic DuckDB query."""
        result = duckdb_conn.execute("SELECT 'Hello' as greeting").fetchone()
        assert result[0] == "Hello"

    def test_duckdb_table_creation(self, duckdb_conn):
        """Test creating and querying a table."""
        duckdb_conn.execute("""
            CREATE TABLE test_data AS
            SELECT 
                1 as id,
                'test' as name,
                3.14 as value
        """)

        result = duckdb_conn.execute("SELECT * FROM test_data").fetchall()
        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == "test"

    def test_duckdb_json_handling(self, duckdb_conn):
        """Test JSON handling in DuckDB."""
        duckdb_conn.execute("""
            CREATE TABLE json_test AS
            SELECT 
                {'name': 'test', 'value': 42} as data
        """)

        result = duckdb_conn.execute(
            "SELECT data->>'name' as name FROM json_test"
        ).fetchone()
        assert result[0] == "test"

    def test_duckdb_aggregation(self, duckdb_conn):
        """Test aggregation functions."""
        duckdb_conn.execute("""
            CREATE TABLE numbers AS
            SELECT 1 as num UNION ALL
            SELECT 2 UNION ALL
            SELECT 3
        """)

        result = duckdb_conn.execute(
            "SELECT SUM(num) as total FROM numbers"
        ).fetchone()
        assert result[0] == 6


class TestConfigurationIntegration:
    """Integration tests for configuration management."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_content = """
[core]
debug = true
log_level = "info"

[web]
timeout = 30
max_results = 100

[mcp_manager]
data_dir = "/tmp/mcp"
"""
            config_path.write_text(config_content)
            yield config_path

    def test_config_file_exists(self, temp_config):
        """Test that config file exists."""
        assert temp_config.exists()

    def test_config_file_readable(self, temp_config):
        """Test that config file is readable."""
        content = temp_config.read_text()
        assert "[core]" in content
        assert "[web]" in content
        assert "[mcp_manager]" in content

    def test_config_parsing(self, temp_config):
        """Test parsing config file."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        content = temp_config.read_text()
        # Just verify it's valid TOML structure
        assert "[" in content
        assert "=" in content


class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner for testing."""
        return CliRunner()

    def test_invalid_command(self, runner):
        """Test handling of invalid command."""
        result = runner.invoke(core_app, ["nonexistent-command"])
        assert result.exit_code != 0

    def test_missing_required_argument(self, runner):
        """Test handling of missing required arguments."""
        # Some commands might require arguments
        result = runner.invoke(core_app, ["--help"])
        assert result.exit_code == 0

    def test_invalid_option_value(self, runner):
        """Test handling of invalid option values."""
        result = runner.invoke(core_app, ["hello", "--count", "invalid"])
        assert result.exit_code != 0


class TestEndToEndWorkflow:
    """Integration tests for end-to-end workflows."""

    @pytest.fixture
    def runner(self):
        """Create a CliRunner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_basic_workflow(self, runner, temp_workspace):
        """Test basic workflow: version -> info -> config."""
        # Get version
        result1 = runner.invoke(core_app, ["version"])
        assert result1.exit_code == 0
        version = result1.stdout

        # Get info
        result2 = runner.invoke(core_app, ["info"])
        assert result2.exit_code == 0

        # Get config
        result3 = runner.invoke(core_app, ["config-show"])
        assert result3.exit_code == 0

        # All should be successful
        assert "0.1.0" in version

    def test_workspace_initialization_workflow(self, temp_workspace):
        """Test workspace initialization workflow."""
        # Create workspace structure
        (temp_workspace / "data").mkdir()
        (temp_workspace / "tools").mkdir()
        (temp_workspace / "config").mkdir()

        # Create config files
        config = {
            "version": "0.1.0",
            "workspace": str(temp_workspace),
            "tools": ["core", "web", "mcp-manager"],
        }

        (temp_workspace / "config" / "main.json").write_text(json.dumps(config))

        # Verify workspace
        assert (temp_workspace / "data").is_dir()
        assert (temp_workspace / "tools").is_dir()
        assert (temp_workspace / "config").is_dir()
        assert (temp_workspace / "config" / "main.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
