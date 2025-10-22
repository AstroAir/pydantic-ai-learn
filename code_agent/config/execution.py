"""
Execution Configuration

Configuration management for code execution with validation, security settings,
and customization options.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..core.types import ExecutionMode


@dataclass
class ValidationConfig:
    """Configuration for code validation."""

    enable_syntax_check: bool = True
    """Enable syntax validation"""

    enable_security_check: bool = True
    """Enable security validation"""

    enable_import_check: bool = True
    """Enable import validation"""

    enable_complexity_check: bool = False
    """Enable complexity validation"""

    max_complexity: int = 20
    """Maximum allowed cyclomatic complexity"""

    max_line_length: int = 120
    """Maximum line length"""

    max_function_length: int = 50
    """Maximum function length in lines"""

    strict_mode: bool = False
    """Enable strict validation mode"""

    custom_validators: list[Callable[[str], list[str]]] = field(default_factory=list)
    """Custom validation functions"""


@dataclass
class VerificationConfig:
    """Configuration for post-execution verification."""

    enable_output_check: bool = True
    """Enable output validation"""

    enable_side_effect_check: bool = True
    """Enable side effect detection"""

    enable_state_check: bool = False
    """Enable state consistency check"""

    max_output_size: int = 1_000_000
    """Maximum output size in bytes"""

    allowed_side_effects: list[str] = field(default_factory=list)
    """Allowed side effects (e.g., 'file_write', 'network_call')"""

    custom_verifiers: list[Callable[[Any], list[str]]] = field(default_factory=list)
    """Custom verification functions"""


@dataclass
class ResourceLimits:
    """Resource limits for code execution."""

    max_execution_time: float = 30.0
    """Maximum execution time in seconds"""

    max_memory_mb: int | None = None
    """Maximum memory in megabytes"""

    max_cpu_percent: int | None = None
    """Maximum CPU usage percentage"""

    max_file_size_mb: int = 10
    """Maximum file size for read/write in megabytes"""

    max_network_calls: int = 0
    """Maximum network calls (0 = disabled)"""

    max_subprocess_calls: int = 0
    """Maximum subprocess calls (0 = disabled)"""


@dataclass
class SecurityConfig:
    """Security configuration for code execution."""

    execution_mode: ExecutionMode = ExecutionMode.SAFE
    """Execution mode"""

    sandbox_enabled: bool = True
    """Enable sandboxed execution"""

    allowed_imports: list[str] | None = None
    """Allowed import modules (None = use defaults)"""

    blocked_imports: list[str] = field(
        default_factory=lambda: [
            "os",
            "subprocess",
            "sys",
            "socket",
            "urllib",
            "requests",
            "http",
            "ftplib",
            "telnetlib",
            "smtplib",
            "poplib",
            "imaplib",
        ]
    )
    """Blocked import modules"""

    allowed_builtins: list[str] | None = None
    """Allowed builtin functions (None = use defaults)"""

    blocked_builtins: list[str] = field(
        default_factory=lambda: ["eval", "exec", "compile", "__import__", "open", "input", "raw_input", "execfile"]
    )
    """Blocked builtin functions"""

    allow_file_access: bool = False
    """Allow file system access"""

    allow_network_access: bool = False
    """Allow network access"""

    allow_subprocess: bool = False
    """Allow subprocess execution"""

    working_directory: Path | None = None
    """Restricted working directory"""


@dataclass
class OutputConfig:
    """Configuration for output formatting."""

    format_type: Literal["json", "text", "markdown", "html"] = "text"
    """Output format"""

    include_metadata: bool = True
    """Include execution metadata"""

    include_timing: bool = True
    """Include timing information"""

    include_resource_usage: bool = False
    """Include resource usage stats"""

    truncate_output: bool = True
    """Truncate long output"""

    max_output_length: int = 10_000
    """Maximum output length in characters"""

    syntax_highlighting: bool = False
    """Enable syntax highlighting (requires rich)"""

    custom_formatters: dict[str, Callable[[Any], str]] = field(default_factory=dict)
    """Custom output formatters"""


@dataclass
class HookConfig:
    """Configuration for execution hooks."""

    enable_pre_validation_hooks: bool = True
    """Enable pre-validation hooks"""

    enable_post_validation_hooks: bool = True
    """Enable post-validation hooks"""

    enable_pre_execution_hooks: bool = True
    """Enable pre-execution hooks"""

    enable_post_execution_hooks: bool = True
    """Enable post-execution hooks"""

    enable_error_hooks: bool = True
    """Enable error hooks"""

    pre_validation_hooks: list[Callable[[str], None]] = field(default_factory=list)
    """Pre-validation hooks"""

    post_validation_hooks: list[Callable[[str, list[str]], None]] = field(default_factory=list)
    """Post-validation hooks"""

    pre_execution_hooks: list[Callable[[str, dict[str, Any]], None]] = field(default_factory=list)
    """Pre-execution hooks"""

    post_execution_hooks: list[Callable[[str, Any], None]] = field(default_factory=list)
    """Post-execution hooks"""

    error_hooks: list[Callable[[Exception, str], None]] = field(default_factory=list)
    """Error hooks"""


@dataclass
class ExecutionConfig:
    """
    Comprehensive configuration for code execution.

    Combines all execution-related configuration options including
    validation, verification, security, resources, output, and hooks.
    """

    validation: ValidationConfig = field(default_factory=ValidationConfig)
    """Validation configuration"""

    verification: VerificationConfig = field(default_factory=VerificationConfig)
    """Verification configuration"""

    security: SecurityConfig = field(default_factory=SecurityConfig)
    """Security configuration"""

    resources: ResourceLimits = field(default_factory=ResourceLimits)
    """Resource limits"""

    output: OutputConfig = field(default_factory=OutputConfig)
    """Output configuration"""

    hooks: HookConfig = field(default_factory=HookConfig)
    """Hook configuration"""

    enable_caching: bool = True
    """Enable result caching"""

    cache_ttl: int = 3600
    """Cache TTL in seconds"""

    enable_logging: bool = True
    """Enable execution logging"""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    """Logging level"""

    dry_run: bool = False
    """Dry run mode (validate only)"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "validation": {
                "enable_syntax_check": self.validation.enable_syntax_check,
                "enable_security_check": self.validation.enable_security_check,
                "enable_import_check": self.validation.enable_import_check,
                "enable_complexity_check": self.validation.enable_complexity_check,
                "max_complexity": self.validation.max_complexity,
                "strict_mode": self.validation.strict_mode,
            },
            "verification": {
                "enable_output_check": self.verification.enable_output_check,
                "enable_side_effect_check": self.verification.enable_side_effect_check,
                "enable_state_check": self.verification.enable_state_check,
                "max_output_size": self.verification.max_output_size,
            },
            "security": {
                "execution_mode": self.security.execution_mode.value,
                "sandbox_enabled": self.security.sandbox_enabled,
                "allow_file_access": self.security.allow_file_access,
                "allow_network_access": self.security.allow_network_access,
                "allow_subprocess": self.security.allow_subprocess,
            },
            "resources": {
                "max_execution_time": self.resources.max_execution_time,
                "max_memory_mb": self.resources.max_memory_mb,
                "max_cpu_percent": self.resources.max_cpu_percent,
            },
            "output": {
                "format_type": self.output.format_type,
                "include_metadata": self.output.include_metadata,
                "include_timing": self.output.include_timing,
                "truncate_output": self.output.truncate_output,
            },
            "enable_caching": self.enable_caching,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "dry_run": self.dry_run,
            **self.metadata,
        }


def create_safe_config() -> ExecutionConfig:
    """Create a safe execution configuration."""
    config = ExecutionConfig()
    config.security.execution_mode = ExecutionMode.SAFE
    config.security.sandbox_enabled = True
    config.security.allow_file_access = False
    config.security.allow_network_access = False
    config.security.allow_subprocess = False
    config.resources.max_execution_time = 10.0
    return config


def create_restricted_config() -> ExecutionConfig:
    """Create a restricted execution configuration."""
    config = ExecutionConfig()
    config.security.execution_mode = ExecutionMode.RESTRICTED
    config.security.sandbox_enabled = True
    config.security.allow_file_access = True
    config.security.allow_network_access = False
    config.security.allow_subprocess = False
    config.resources.max_execution_time = 30.0
    return config


def create_full_config() -> ExecutionConfig:
    """Create a full execution configuration (use with caution)."""
    config = ExecutionConfig()
    config.security.execution_mode = ExecutionMode.FULL
    config.security.sandbox_enabled = False
    config.security.allow_file_access = True
    config.security.allow_network_access = True
    config.security.allow_subprocess = True
    config.resources.max_execution_time = 60.0
    return config
