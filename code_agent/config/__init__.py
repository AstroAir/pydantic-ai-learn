"""
Configuration Module

Comprehensive configuration management for the code agent.

Exports:
    - ConfigManager: Main configuration manager
    - load_config: Load configuration from file
    - MCPConfigLoader: MCP configuration loader
    - StructuredLogger: Structured logging
    - LogLevel, LogFormat: Logging configuration
    - create_logger: Logger factory function
    - Terminal security configuration
    - Prompt configuration and templates
"""

from __future__ import annotations

from .execution import (
    ExecutionConfig,
    HookConfig,
    OutputConfig,
    ResourceLimits,
    SecurityConfig,
    ValidationConfig,
    VerificationConfig,
    create_full_config,
    create_restricted_config,
    create_safe_config,
)
from .logging import (
    LogFormat,
    LogLevel,
    StructuredLogger,
    create_logger,
)
from .manager import ConfigManager, load_config
from .mcp import MCPConfig, MCPConfigLoader, MCPTransportType
from .prompts import (
    PromptConfig,
    PromptTemplate,
    PromptVariable,
    load_prompt_config,
    save_prompt_config,
)
from .terminal_security import (
    CommandValidationConfig,
    FilesystemAccessConfig,
    ResourceLimitConfig,
    TerminalSecurityConfig,
    create_development_terminal_config,
    create_safe_terminal_config,
)

__all__ = [
    "ConfigManager",
    "load_config",
    "MCPConfigLoader",
    "MCPConfig",
    "MCPTransportType",
    "LogFormat",
    "LogLevel",
    "StructuredLogger",
    "create_logger",
    "ExecutionConfig",
    "ValidationConfig",
    "VerificationConfig",
    "SecurityConfig",
    "ResourceLimits",
    "OutputConfig",
    "HookConfig",
    "create_safe_config",
    "create_restricted_config",
    "create_full_config",
    # Terminal security
    "TerminalSecurityConfig",
    "CommandValidationConfig",
    "ResourceLimitConfig",
    "FilesystemAccessConfig",
    "create_safe_terminal_config",
    "create_development_terminal_config",
    # Prompt configuration
    "PromptConfig",
    "PromptTemplate",
    "PromptVariable",
    "load_prompt_config",
    "save_prompt_config",
]
