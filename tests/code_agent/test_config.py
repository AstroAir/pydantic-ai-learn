"""
Configuration Tests

Tests for configuration modules.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from code_agent.config import ConfigManager, MCPConfig, MCPConfigLoader, MCPTransportType


class TestConfigManager:
    """Test ConfigManager."""

    def test_config_manager_creation(self):
        """Test config manager creation."""
        manager = ConfigManager()

        assert manager.config == {}

    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        manager = ConfigManager()
        config_dict = {
            "model": "openai:gpt-4",
            "max_retries": 5,
        }

        result = manager.load_from_dict(config_dict)

        assert result["model"] == "openai:gpt-4"
        assert result["max_retries"] == 5

    def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "openai:gpt-4"}, f)
            temp_path = f.name

        try:
            result = manager.load_from_file(temp_path)
            assert result["model"] == "openai:gpt-4"
        finally:
            Path(temp_path).unlink()

    def test_get_config_value(self):
        """Test getting configuration value."""
        manager = ConfigManager()
        manager.load_from_dict(
            {
                "model": "openai:gpt-4",
                "nested": {"key": "value"},
            }
        )

        assert manager.get("model") == "openai:gpt-4"
        assert manager.get("nested.key") == "value"
        assert manager.get("nonexistent", "default") == "default"

    def test_set_config_value(self):
        """Test setting configuration value."""
        manager = ConfigManager()

        manager.set("model", "openai:gpt-3.5-turbo")

        assert manager.get("model") == "openai:gpt-3.5-turbo"

    def test_resolve_env_vars(self):
        """Test resolving environment variables."""
        import os

        os.environ["TEST_VAR"] = "test_value"

        manager = ConfigManager()
        manager.load_from_dict(
            {
                "key": "${TEST_VAR}",
            }
        )

        assert manager.get("key") == "test_value"


class TestMCPConfig:
    """Test MCPConfig."""

    def test_mcp_config_stdio(self):
        """Test MCP config with stdio transport."""
        config = MCPConfig(
            name="test",
            transport=MCPTransportType.STDIO,
            command="python",
            args=["-m", "module"],
        )

        assert config.name == "test"
        assert config.transport == MCPTransportType.STDIO
        assert config.command == "python"

    def test_mcp_config_sse(self):
        """Test MCP config with SSE transport."""
        config = MCPConfig(
            name="test",
            transport=MCPTransportType.SSE,
            url="http://localhost:8000",
        )

        assert config.transport == MCPTransportType.SSE
        assert config.url == "http://localhost:8000"

    def test_mcp_config_validation(self):
        """Test MCP config validation."""
        with pytest.raises(ValueError):
            MCPConfig(
                name="test",
                transport=MCPTransportType.STDIO,
                # Missing required 'command'
            )

    def test_mcp_config_credentials(self):
        """Test MCP config credentials."""
        config = MCPConfig(
            name="test",
            transport=MCPTransportType.SSE,
            url="http://localhost:8000",
            credentials={
                "api_key": "test_key",
                "token": "test_token",
            },
        )

        creds = config.get_credentials()

        assert creds.api_key == "test_key"
        assert creds.token == "test_token"


class TestMCPConfigLoader:
    """Test MCPConfigLoader."""

    def test_load_from_dict(self):
        """Test loading MCP config from dictionary."""
        loader = MCPConfigLoader()
        config_dict = {
            "mcpServers": {
                "test": {
                    "command": "python",
                    "args": ["-m", "module"],
                }
            }
        }

        configs = loader.load_from_dict(config_dict)

        assert len(configs) == 1
        assert configs[0].name == "test"

    def test_load_from_file(self):
        """Test loading MCP config from file."""
        loader = MCPConfigLoader()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "mcpServers": {
                        "test": {
                            "command": "python",
                            "args": ["-m", "module"],
                        }
                    }
                },
                f,
            )
            temp_path = f.name

        try:
            configs = loader.load_from_file(temp_path)
            assert len(configs) == 1
        finally:
            Path(temp_path).unlink()

    def test_load_disabled_servers(self):
        """Test that disabled servers are filtered out."""
        loader = MCPConfigLoader()
        config_dict = {
            "mcpServers": {
                "enabled": {
                    "command": "python",
                    "enabled": True,
                },
                "disabled": {
                    "command": "python",
                    "enabled": False,
                },
            }
        }

        configs = loader.load_from_dict(config_dict)

        assert len(configs) == 1
        assert configs[0].name == "enabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
