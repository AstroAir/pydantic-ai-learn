"""
Terminal Sandbox

Secure terminal command execution with validation, resource limits,
and filesystem access restrictions.

Features:
- Command validation and filtering
- Resource limit enforcement
- Filesystem access restrictions
- Audit logging
- Rate limiting
- Circuit breaker integration

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import re
import shlex
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.terminal_security import TerminalSecurityConfig

from ..utils.errors import CircuitBreaker
from ..utils.terminal_exec import CommandResult, run_command, run_command_async

# ============================================================================
# Validation Types
# ============================================================================


@dataclass
class ValidationResult:
    """Result of command validation."""

    is_valid: bool
    """Whether the command is valid"""

    errors: list[str] = field(default_factory=list)
    """Validation errors"""

    warnings: list[str] = field(default_factory=list)
    """Validation warnings"""

    sanitized_command: str = ""
    """Sanitized command (if applicable)"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""


# ============================================================================
# Command Validator
# ============================================================================


class CommandValidator:
    """
    Validates commands against security policies.

    Performs pattern matching, whitelist/blacklist checking,
    and path validation to prevent dangerous operations.
    """

    def __init__(self, config: TerminalSecurityConfig):
        """
        Initialize validator.

        Args:
            config: Terminal security configuration
        """
        self.config = config
        self._compiled_patterns = [re.compile(pattern) for pattern in config.command_validation.dangerous_patterns]

    def validate(self, command: str) -> ValidationResult:
        """
        Validate command against security policies.

        Args:
            command: Command to validate

        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True, sanitized_command=command)

        if not self.config.command_validation.enable_validation:
            return result

        # Check command length
        if len(command) > self.config.command_validation.max_command_length:
            result.is_valid = False
            result.errors.append(
                f"Command exceeds maximum length ({len(command)} > {self.config.command_validation.max_command_length})"
            )
            return result

        # Check for dangerous patterns
        for pattern in self._compiled_patterns:
            if pattern.search(command):
                result.is_valid = False
                result.errors.append(f"Command contains dangerous pattern: {pattern.pattern}")
                if self.config.strict_mode:
                    return result

        # Check shell operators
        if not self.config.command_validation.allow_shell_operators:
            shell_operators = ["&&", "||", "|", ">", "<", ">>", "<<"]
            for op in shell_operators:
                if op in command:
                    result.is_valid = False
                    result.errors.append(f"Shell operator not allowed: {op}")
                    if self.config.strict_mode:
                        return result

        # Check command substitution
        if not self.config.command_validation.allow_command_substitution and ("$(" in command or "`" in command):
            result.is_valid = False
            result.errors.append("Command substitution not allowed")
            if self.config.strict_mode:
                return result

        # Parse command to get base command and arguments
        try:
            parts = shlex.split(command)
        except ValueError as e:
            result.is_valid = False
            result.errors.append(f"Failed to parse command: {e}")
            return result

        if not parts:
            result.is_valid = False
            result.errors.append("Empty command")
            return result

        base_command = Path(parts[0]).name  # Get just the command name, not full path
        arguments = parts[1:]

        # Check argument count
        if len(arguments) > self.config.command_validation.max_arguments:
            result.is_valid = False
            result.errors.append(
                f"Too many arguments ({len(arguments)} > {self.config.command_validation.max_arguments})"
            )
            return result

        # Validate based on mode
        validation_mode = self.config.command_validation.validation_mode

        if (
            validation_mode in ("whitelist", "hybrid")
            and base_command not in self.config.command_validation.allowed_commands
        ):
            result.is_valid = False
            result.errors.append(f"Command not in whitelist: {base_command}")
            if self.config.strict_mode:
                return result

        if (
            validation_mode in ("blacklist", "hybrid")
            and base_command in self.config.command_validation.blocked_commands
        ):
            result.is_valid = False
            result.errors.append(f"Command is blocked: {base_command}")
            return result

        # Validate paths in arguments
        path_validation = self._validate_paths(arguments)
        if not path_validation.is_valid:
            result.is_valid = False
            result.errors.extend(path_validation.errors)
            result.warnings.extend(path_validation.warnings)

        result.metadata["base_command"] = base_command
        result.metadata["argument_count"] = len(arguments)

        return result

    def _validate_paths(self, arguments: list[str]) -> ValidationResult:
        """Validate paths in command arguments."""
        result = ValidationResult(is_valid=True)

        if not self.config.filesystem_access.enable_restrictions:
            return result

        for arg in arguments:
            # Check for blocked paths
            for blocked_path in self.config.filesystem_access.blocked_paths:
                if blocked_path in arg:
                    result.is_valid = False
                    result.errors.append(f"Blocked path in argument: {blocked_path}")
                    if self.config.strict_mode:
                        return result

            # Check for parent directory traversal
            if not self.config.filesystem_access.allow_parent_directory and ".." in arg:
                result.is_valid = False
                result.errors.append("Parent directory traversal not allowed")
                if self.config.strict_mode:
                    return result

            # Check for absolute paths
            if not self.config.filesystem_access.allow_absolute_paths:
                try:
                    path = Path(arg)
                    if path.is_absolute():
                        result.is_valid = False
                        result.errors.append(f"Absolute path not allowed: {arg}")
                        if self.config.strict_mode:
                            return result
                except Exception:
                    # Not a valid path, skip
                    pass

            # Check for blocked directories
            try:
                path = Path(arg).resolve()
                for blocked_dir in self.config.filesystem_access.blocked_directories:
                    try:
                        if path.is_relative_to(blocked_dir):
                            result.is_valid = False
                            result.errors.append(f"Path in blocked directory: {blocked_dir}")
                            if self.config.strict_mode:
                                return result
                    except (ValueError, AttributeError):
                        # is_relative_to not available or path comparison failed
                        if str(path).startswith(str(blocked_dir)):
                            result.is_valid = False
                            result.errors.append(f"Path in blocked directory: {blocked_dir}")
                            if self.config.strict_mode:
                                return result
            except Exception:
                # Path resolution failed, add warning
                result.warnings.append(f"Could not resolve path: {arg}")

        return result


# ============================================================================
# Rate Limiter
# ============================================================================


class RateLimiter:
    """Simple rate limiter for command execution."""

    def __init__(self, max_commands_per_minute: int):
        """Initialize rate limiter."""
        self.max_commands_per_minute = max_commands_per_minute
        self.command_times: deque[float] = deque(maxlen=max_commands_per_minute)

    def check_rate_limit(self) -> tuple[bool, str]:
        """
        Check if rate limit is exceeded.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        current_time = time.time()

        # Remove commands older than 1 minute
        while self.command_times and current_time - self.command_times[0] > 60:
            self.command_times.popleft()

        # Check if limit exceeded
        if len(self.command_times) >= self.max_commands_per_minute:
            return False, f"Rate limit exceeded: {self.max_commands_per_minute} commands per minute"

        # Record this command
        self.command_times.append(current_time)
        return True, ""


# ============================================================================
# Terminal Sandbox
# ============================================================================


class TerminalSandbox:
    """
    Secure terminal sandbox for command execution.

    Provides command validation, resource limits, filesystem restrictions,
    audit logging, and rate limiting for safe terminal operations.
    """

    def __init__(self, config: TerminalSecurityConfig | None = None):
        """
        Initialize terminal sandbox.

        Args:
            config: Terminal security configuration
        """
        if config is None:
            from ..config.terminal_security import create_safe_terminal_config

            config = create_safe_terminal_config()

        self.config = config
        self.validator = CommandValidator(config)
        self.circuit_breaker = CircuitBreaker(name="terminal_sandbox", failure_threshold=5, recovery_timeout=60.0)
        self.rate_limiter = RateLimiter(config.max_commands_per_minute) if config.enable_rate_limiting else None

        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.audit_log: list[dict[str, Any]] = []

    def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        shell: bool = False,
    ) -> CommandResult:
        """
        Execute command with security validation and resource limits.

        Args:
            command: Command to execute
            timeout: Execution timeout (uses config default if None)
            cwd: Working directory
            env: Environment variables
            shell: Use shell execution (not recommended for security)

        Returns:
            Command execution result
        """
        self.execution_count += 1
        start_time = time.time()

        # Check rate limit
        if self.rate_limiter:
            is_allowed, error_msg = self.rate_limiter.check_rate_limit()
            if not is_allowed:
                self.failure_count += 1
                return CommandResult(
                    stdout="",
                    stderr=error_msg,
                    exit_code=429,  # Too Many Requests
                    duration_s=0.0,
                    error=error_msg,
                )

        # Validate command
        validation_result = self.validator.validate(command)
        if not validation_result.is_valid:
            self.failure_count += 1
            error_msg = "; ".join(validation_result.errors)
            self._log_execution(command, success=False, error=error_msg, duration=0.0)
            return CommandResult(
                stdout="",
                stderr=f"Command validation failed: {error_msg}",
                exit_code=403,  # Forbidden
                duration_s=0.0,
                error=error_msg,
            )

        # Use configured timeout if not specified
        if timeout is None:
            timeout = self.config.resource_limits.max_execution_time

        # Set working directory from config if not specified
        if cwd is None and self.config.filesystem_access.working_directory:
            cwd = self.config.filesystem_access.working_directory

        # Execute command
        try:
            result = run_command(
                command,
                timeout=timeout,
                cwd=cwd,
                env=env,
                shell=shell,
            )

            # Check if successful
            if result.exit_code == 0 and not result.error:
                self.success_count += 1
                self._log_execution(command, success=True, duration=result.duration_s)
            else:
                self.failure_count += 1
                self._log_execution(
                    command, success=False, error=result.error or result.stderr, duration=result.duration_s
                )

            return result

        except Exception as e:
            self.failure_count += 1
            error_msg = str(e)
            duration = time.time() - start_time
            self._log_execution(command, success=False, error=error_msg, duration=duration)
            return CommandResult(
                stdout="",
                stderr=error_msg,
                exit_code=-1,
                duration_s=duration,
                error=error_msg,
            )

    async def execute_async(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        shell: bool = False,
        stream: bool = False,
        on_stdout: Any = None,
        on_stderr: Any = None,
    ) -> CommandResult:
        """
        Execute command asynchronously with security validation.

        Args:
            command: Command to execute
            timeout: Execution timeout
            cwd: Working directory
            env: Environment variables
            shell: Use shell execution
            stream: Enable streaming output
            on_stdout: Callback for stdout chunks
            on_stderr: Callback for stderr chunks

        Returns:
            Command execution result
        """
        self.execution_count += 1
        start_time = time.time()

        # Check rate limit
        if self.rate_limiter:
            is_allowed, error_msg = self.rate_limiter.check_rate_limit()
            if not is_allowed:
                self.failure_count += 1
                return CommandResult(
                    stdout="",
                    stderr=error_msg,
                    exit_code=429,
                    duration_s=0.0,
                    error=error_msg,
                )

        # Validate command
        validation_result = self.validator.validate(command)
        if not validation_result.is_valid:
            self.failure_count += 1
            error_msg = "; ".join(validation_result.errors)
            self._log_execution(command, success=False, error=error_msg, duration=0.0)
            return CommandResult(
                stdout="",
                stderr=f"Command validation failed: {error_msg}",
                exit_code=403,
                duration_s=0.0,
                error=error_msg,
            )

        # Use configured timeout if not specified
        if timeout is None:
            timeout = self.config.resource_limits.max_execution_time

        # Set working directory from config if not specified
        if cwd is None and self.config.filesystem_access.working_directory:
            cwd = self.config.filesystem_access.working_directory

        # Execute command
        try:
            result = await run_command_async(
                command,
                timeout=timeout,
                cwd=cwd,
                env=env,
                shell=shell,
                stream=stream,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )

            # Check if successful
            if result.exit_code == 0 and not result.error:
                self.success_count += 1
                self._log_execution(command, success=True, duration=result.duration_s)
            else:
                self.failure_count += 1
                self._log_execution(
                    command, success=False, error=result.error or result.stderr, duration=result.duration_s
                )

            return result

        except Exception as e:
            self.failure_count += 1
            error_msg = str(e)
            duration = time.time() - start_time
            self._log_execution(command, success=False, error=error_msg, duration=duration)
            return CommandResult(
                stdout="",
                stderr=error_msg,
                exit_code=-1,
                duration_s=duration,
                error=error_msg,
            )

    def validate_command(self, command: str) -> ValidationResult:
        """
        Validate command without executing it.

        Args:
            command: Command to validate

        Returns:
            Validation result
        """
        return self.validator.validate(command)

    def _log_execution(self, command: str, success: bool, error: str = "", duration: float = 0.0) -> None:
        """Log command execution for audit trail."""
        if not self.config.enable_audit_logging:
            return

        log_entry = {
            "timestamp": time.time(),
            "command": command,
            "success": success,
            "error": error,
            "duration": duration,
        }

        self.audit_log.append(log_entry)

        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": self.execution_count,
            "successful": self.success_count,
            "failed": self.failure_count,
            "success_rate": self.success_count / self.execution_count if self.execution_count > 0 else 0,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "audit_log_size": len(self.audit_log),
        }

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get recent audit log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of audit log entries
        """
        return self.audit_log[-limit:]

    def clear_audit_log(self) -> None:
        """Clear audit log."""
        self.audit_log.clear()


__all__ = [
    "ValidationResult",
    "CommandValidator",
    "RateLimiter",
    "TerminalSandbox",
]
