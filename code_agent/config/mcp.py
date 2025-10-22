"""
MCP Configuration System

Comprehensive configuration management for MCP (Model Context Protocol) servers
with support for multiple transport types, validation, and environment variable substitution.

Features:
- Multiple transport types (stdio, SSE, HTTP)
- JSON configuration file loading
- Environment variable substitution
- Schema validation with helpful errors
- Secure credential management

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MCPTransportType(str, Enum):
    """MCP transport protocol types."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class MCPServerCredentials:
    """
    Credentials for MCP server authentication.

    Supports environment variable substitution for secure credential management.
    """

    api_key: str | None = None
    token: str | None = None
    username: str | None = None
    password: str | None = None
    custom_headers: dict[str, str] = field(default_factory=dict)

    def resolve_env_vars(self) -> None:
        """Resolve environment variables in credential values."""
        if self.api_key:
            self.api_key = self._resolve_env_var(self.api_key)
        if self.token:
            self.token = self._resolve_env_var(self.token)
        if self.username:
            self.username = self._resolve_env_var(self.username)
        if self.password:
            self.password = self._resolve_env_var(self.password)

        # Resolve env vars in custom headers
        resolved_headers = {}
        for key, value in self.custom_headers.items():
            resolved_headers[key] = self._resolve_env_var(value)
        self.custom_headers = resolved_headers

    @staticmethod
    def _resolve_env_var(value: str) -> str:
        """Resolve environment variable references in format ${VAR_NAME}."""
        pattern = r"\$\{([^}]+)\}"

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replacer, value)


class MCPConfig(BaseModel):
    """
    Configuration for a single MCP server.

    Supports all transport types with appropriate parameters.
    """

    name: str = Field(description="Unique server name/identifier")
    transport: MCPTransportType = Field(description="Transport protocol type")

    # Stdio transport parameters
    command: str | None = Field(default=None, description="Command to execute (stdio)")
    args: list[str] = Field(default_factory=list, description="Command arguments (stdio)")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables (stdio)")
    cwd: str | None = Field(default=None, description="Working directory (stdio)")
    timeout: float = Field(default=30.0, description="Connection timeout in seconds")

    # HTTP/SSE transport parameters
    url: str | None = Field(default=None, description="Server URL (SSE/HTTP)")

    # Tool configuration
    tool_prefix: str | None = Field(default=None, description="Prefix for tool names")
    enabled: bool = Field(default=True, description="Whether server is enabled")

    # Advanced configuration
    credentials: dict[str, Any] = Field(default_factory=dict, description="Authentication credentials")
    retry_config: dict[str, Any] = Field(default_factory=dict, description="Retry configuration")
    health_check_interval: float = Field(default=60.0, description="Health check interval in seconds")

    model_config = {"extra": "forbid"}

    @field_validator("transport", mode="before")
    @classmethod
    def validate_transport(cls, v: Any) -> MCPTransportType:
        """Validate and convert transport type."""
        if isinstance(v, str):
            v = MCPTransportType(v.lower())
        if not isinstance(v, MCPTransportType):
            raise ValueError(f"Invalid transport type: {v}")
        return v

    @model_validator(mode="after")
    def validate_transport_params(self) -> MCPConfig:
        """Validate transport-specific parameters."""
        if self.transport == MCPTransportType.STDIO and not self.command:
            raise ValueError(f"Server '{self.name}': 'command' is required for stdio transport")
        if self.transport in (MCPTransportType.SSE, MCPTransportType.STREAMABLE_HTTP) and not self.url:
            raise ValueError(f"Server '{self.name}': 'url' is required for {self.transport.value} transport")
        return self

    def get_credentials(self) -> MCPServerCredentials:
        """Get resolved credentials with environment variable substitution."""
        creds = MCPServerCredentials(
            api_key=self.credentials.get("api_key"),
            token=self.credentials.get("token"),
            username=self.credentials.get("username"),
            password=self.credentials.get("password"),
            custom_headers=self.credentials.get("custom_headers", {}),
        )
        creds.resolve_env_vars()
        return creds


class MCPConfigFile(BaseModel):
    """
    Schema for MCP configuration file.

    Follows the standard MCP configuration format with extensions.
    """

    mcpServers: dict[str, dict[str, Any]] = Field(  # noqa: N815
        description="MCP server configurations"
    )

    model_config = {"extra": "allow"}

    def to_mcp_configs(self) -> list[MCPConfig]:
        """Convert file format to MCPConfig objects."""
        configs: list[MCPConfig] = []

        for name, server_config in self.mcpServers.items():
            # Determine transport type
            if "command" in server_config:
                transport = MCPTransportType.STDIO
            elif "url" in server_config:
                transport_str = server_config.get("transport", "sse")
                transport = MCPTransportType(transport_str.lower())
            else:
                raise ValueError(f"Server '{name}': Cannot determine transport type")

            # Build MCPConfig
            config = MCPConfig(
                name=name,
                transport=transport,
                command=server_config.get("command"),
                args=server_config.get("args", []),
                env=server_config.get("env", {}),
                cwd=server_config.get("cwd"),
                timeout=server_config.get("timeout", 30.0),
                url=server_config.get("url"),
                tool_prefix=server_config.get("tool_prefix"),
                enabled=server_config.get("enabled", True),
                credentials=server_config.get("credentials", {}),
                retry_config=server_config.get("retry_config", {}),
                health_check_interval=server_config.get("health_check_interval", 60.0),
            )
            configs.append(config)

        return configs


class MCPConfigLoader:
    """
    Loader for MCP configuration files with validation and error reporting.

    Supports JSON format with environment variable substitution.
    """

    def __init__(self) -> None:
        """Initialize configuration loader."""

    def load_from_file(self, config_path: str | Path) -> list[MCPConfig]:
        """
        Load MCP server configurations from file.

        Args:
            config_path: Path to configuration file

        Returns:
            List of validated MCPConfig objects

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"MCP configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                raw_config = json.load(f)

            # Validate against schema
            config_file = MCPConfigFile(**raw_config)

            # Convert to MCPConfig objects
            configs = config_file.to_mcp_configs()

            # Filter enabled servers and return
            return [c for c in configs if c.enabled]

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load MCP configuration: {e}") from e

    def load_from_dict(self, config_dict: dict[str, Any]) -> list[MCPConfig]:
        """
        Load MCP server configurations from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            List of validated MCPConfig objects
        """
        try:
            config_file = MCPConfigFile(**config_dict)
            configs = config_file.to_mcp_configs()
            return [c for c in configs if c.enabled]
        except Exception as e:
            raise ValueError(f"Failed to load MCP configuration from dict: {e}") from e


__all__ = [
    "MCPTransportType",
    "MCPServerCredentials",
    "MCPConfig",
    "MCPConfigFile",
    "MCPConfigLoader",
]
