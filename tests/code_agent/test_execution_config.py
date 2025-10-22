"""
Execution Configuration Tests

Tests for code execution configuration modules.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from typing import Any

from code_agent.config.execution import (
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
from code_agent.core.types import ExecutionMode


class TestValidationConfig:
    """Test ValidationConfig."""

    def test_default_validation_config(self):
        """Test default validation configuration."""
        config = ValidationConfig()

        assert config.enable_syntax_check is True
        assert config.enable_security_check is True
        assert config.enable_import_check is True
        assert config.enable_complexity_check is False
        assert config.max_complexity == 20
        assert config.strict_mode is False
        assert isinstance(config.custom_validators, list)
        assert len(config.custom_validators) == 0

    def test_custom_validation_config(self):
        """Test custom validation configuration."""
        config = ValidationConfig(
            enable_syntax_check=False,
            enable_security_check=True,
            enable_complexity_check=True,
            max_complexity=15,
            strict_mode=True,
        )

        assert config.enable_syntax_check is False
        assert config.enable_security_check is True
        assert config.enable_complexity_check is True
        assert config.max_complexity == 15
        assert config.strict_mode is True

    def test_add_custom_validator(self):
        """Test adding custom validators."""

        def my_validator(code: str) -> list[str]:
            return []

        config = ValidationConfig()
        config.custom_validators.append(my_validator)

        assert len(config.custom_validators) == 1
        assert config.custom_validators[0] == my_validator


class TestVerificationConfig:
    """Test VerificationConfig."""

    def test_default_verification_config(self):
        """Test default verification configuration."""
        config = VerificationConfig()

        assert config.enable_output_check is True
        assert config.enable_side_effect_check is True
        assert config.max_output_size == 1_000_000
        assert isinstance(config.custom_verifiers, list)
        assert len(config.custom_verifiers) == 0

    def test_custom_verification_config(self):
        """Test custom verification configuration."""
        config = VerificationConfig(
            enable_output_check=False,
            max_output_size=500_000,
        )

        assert config.enable_output_check is False
        assert config.max_output_size == 500_000


class TestSecurityConfig:
    """Test SecurityConfig."""

    def test_default_security_config(self):
        """Test default security configuration."""
        config = SecurityConfig()

        assert config.execution_mode == ExecutionMode.SAFE
        assert config.sandbox_enabled is True
        assert "os" in config.blocked_imports
        assert "subprocess" in config.blocked_imports
        assert "eval" in config.blocked_builtins
        assert "exec" in config.blocked_builtins
        assert config.allow_file_access is False
        assert config.allow_network_access is False

    def test_custom_security_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            execution_mode=ExecutionMode.RESTRICTED,
            sandbox_enabled=False,
            blocked_imports=["custom_module"],
            allow_file_access=True,
        )

        assert config.execution_mode == ExecutionMode.RESTRICTED
        assert config.sandbox_enabled is False
        assert config.blocked_imports == ["custom_module"]
        assert config.allow_file_access is True


class TestResourceLimits:
    """Test ResourceLimits."""

    def test_default_resource_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()

        assert limits.max_execution_time == 30.0
        assert limits.max_memory_mb is None
        assert limits.max_cpu_percent is None
        assert limits.max_file_size_mb == 10
        assert limits.max_network_calls == 0
        assert limits.max_subprocess_calls == 0

    def test_custom_resource_limits(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            max_execution_time=60.0,
            max_memory_mb=1024,
            max_cpu_percent=50,
        )

        assert limits.max_execution_time == 60.0
        assert limits.max_memory_mb == 1024
        assert limits.max_cpu_percent == 50


class TestOutputConfig:
    """Test OutputConfig."""

    def test_default_output_config(self):
        """Test default output configuration."""
        config = OutputConfig()

        assert config.format_type == "text"
        assert config.include_metadata is True
        assert config.include_timing is True
        assert config.include_resource_usage is False
        assert config.truncate_output is True
        assert config.max_output_length == 10_000

    def test_custom_output_config(self):
        """Test custom output configuration."""
        config = OutputConfig(
            format_type="json",
            include_metadata=False,
            truncate_output=False,
        )

        assert config.format_type == "json"
        assert config.include_metadata is False
        assert config.truncate_output is False


class TestHookConfig:
    """Test HookConfig."""

    def test_default_hook_config(self):
        """Test default hook configuration."""
        config = HookConfig()

        assert isinstance(config.pre_validation_hooks, list)
        assert isinstance(config.post_validation_hooks, list)
        assert isinstance(config.pre_execution_hooks, list)
        assert isinstance(config.post_execution_hooks, list)
        assert isinstance(config.error_hooks, list)
        assert len(config.pre_validation_hooks) == 0

    def test_add_hooks(self):
        """Test adding hooks."""

        def pre_hook(code: str) -> None:
            pass

        def post_hook(code: str, result: Any) -> None:
            pass

        config = HookConfig()
        config.pre_execution_hooks.append(pre_hook)
        config.post_execution_hooks.append(post_hook)

        assert len(config.pre_execution_hooks) == 1
        assert len(config.post_execution_hooks) == 1


class TestExecutionConfig:
    """Test ExecutionConfig."""

    def test_default_execution_config(self):
        """Test default execution configuration."""
        config = ExecutionConfig()

        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.verification, VerificationConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.resources, ResourceLimits)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.hooks, HookConfig)
        assert config.dry_run is False
        assert config.enable_caching is True
        assert config.cache_ttl == 3600

    def test_custom_execution_config(self):
        """Test custom execution configuration."""
        validation = ValidationConfig(enable_syntax_check=False)
        security = SecurityConfig(execution_mode=ExecutionMode.FULL)

        config = ExecutionConfig(
            validation=validation,
            security=security,
            dry_run=True,
            enable_caching=False,
        )

        assert config.validation.enable_syntax_check is False
        assert config.security.execution_mode == ExecutionMode.FULL
        assert config.dry_run is True
        assert config.enable_caching is False


class TestFactoryFunctions:
    """Test factory functions for creating configurations."""

    def test_create_safe_config(self):
        """Test creating safe configuration."""
        config = create_safe_config()

        assert isinstance(config, ExecutionConfig)
        assert config.security.execution_mode == ExecutionMode.SAFE
        assert config.security.sandbox_enabled is True
        assert config.security.allow_file_access is False
        assert config.security.allow_network_access is False
        assert config.validation.enable_syntax_check is True
        assert config.validation.enable_security_check is True

    def test_create_restricted_config(self):
        """Test creating restricted configuration."""
        config = create_restricted_config()

        assert isinstance(config, ExecutionConfig)
        assert config.security.execution_mode == ExecutionMode.RESTRICTED
        assert config.security.sandbox_enabled is True
        assert config.security.allow_file_access is True
        assert config.security.allow_network_access is False

    def test_create_full_config(self):
        """Test creating full configuration."""
        config = create_full_config()

        assert isinstance(config, ExecutionConfig)
        assert config.security.execution_mode == ExecutionMode.FULL
        assert config.security.sandbox_enabled is False
        assert config.security.allow_file_access is True
        assert config.security.allow_network_access is True

    def test_factory_configs_are_independent(self):
        """Test that factory-created configs are independent instances."""
        config1 = create_safe_config()
        config2 = create_safe_config()

        # Modify config1
        config1.validation.enable_syntax_check = False

        # config2 should not be affected
        assert config2.validation.enable_syntax_check is True
