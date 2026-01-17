"""Tests for AIFT configuration."""

import json
import tempfile
from pathlib import Path
import pytest
from aift.config import AIFTConfig, get_config


def test_config_defaults():
    """Test default configuration values."""
    config = AIFTConfig()
    assert config.log_level == "INFO"
    assert config.config_dir == Path.home() / "aift"


def test_get_config():
    """Test get_config function."""
    config = get_config()
    assert isinstance(config, AIFTConfig)
    assert config.log_level == "INFO"


def test_tool_config_save_and_load():
    """Test saving and loading tool configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AIFTConfig(config_dir=Path(tmpdir))
        
        # Save tool config
        test_config = {"setting1": "value1", "setting2": 42}
        config.save_tool_config("testtool", test_config)
        
        # Load tool config
        loaded_config = config.get_tool_config("testtool")
        assert loaded_config == test_config


def test_tool_config_nonexistent():
    """Test loading config for nonexistent tool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AIFTConfig(config_dir=Path(tmpdir))
        
        # Should return empty dict for nonexistent tool
        loaded_config = config.get_tool_config("nonexistent")
        assert loaded_config == {}
