"""Tests for AIFT CLI application."""

import pytest
from typer.testing import CliRunner
from aift.cli import app

runner = CliRunner()


def test_version():
    """Test version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AIFT" in result.stdout
    assert "0.1.0" in result.stdout


def test_hello_default():
    """Test hello command with default argument."""
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello" in result.stdout
    assert "World" in result.stdout


def test_hello_with_name():
    """Test hello command with custom name."""
    result = runner.invoke(app, ["hello", "AI"])
    assert result.exit_code == 0
    assert "Hello" in result.stdout
    assert "AI" in result.stdout


def test_hello_with_count():
    """Test hello command with count option."""
    result = runner.invoke(app, ["hello", "Test", "--count", "2"])
    assert result.exit_code == 0
    assert "Hello" in result.stdout
    assert "Test" in result.stdout


def test_info():
    """Test info command."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "AIFT Configuration" in result.stdout


def test_config_show():
    """Test config-show command."""
    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0
    assert "Current Configuration" in result.stdout


def test_help():
    """Test help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AI-Friendly Tools" in result.stdout
