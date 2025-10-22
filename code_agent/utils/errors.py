"""
Error Handling & Recovery

Comprehensive error handling system with automatic recovery, circuit breakers,
error categorization, and intelligent retry strategies.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    """Error category classification."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    RECOVERABLE = "recoverable"
    FATAL = "fatal"


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and recovery."""

    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float = field(default_factory=time.time)
    stack_trace: str | None = None
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    input_parameters: dict[str, Any] = field(default_factory=dict)
    recovery_suggestions: list[str] = field(default_factory=list)
    retry_count: int = 0

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        category: ErrorCategory = ErrorCategory.TRANSIENT,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        state_snapshot: dict[str, Any] | None = None,
        input_parameters: dict[str, Any] | None = None,
    ) -> ErrorContext:
        """Create error context from exception."""
        return cls(
            error_type=type(exc).__name__,
            error_message=str(exc),
            category=category,
            severity=severity,
            stack_trace=traceback.format_exc(),
            state_snapshot=state_snapshot or {},
            input_parameters=input_parameters or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace,
            "state_snapshot": self.state_snapshot,
            "input_parameters": self.input_parameters,
            "recovery_suggestions": self.recovery_suggestions,
            "retry_count": self.retry_count,
        }


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    success_count: int = field(default=0, init=False)
    last_failure_time: float | None = field(default=None, init=False)

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. Wait {self.recovery_timeout}s before retry."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


@dataclass
class RetryStrategy:
    """Enhanced retry strategy with exponential backoff."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(self.base_delay * (self.exponential_base**attempt), self.max_delay)

        if self.jitter:
            import random

            delay *= 0.5 + random.random()

        return delay

    def should_retry(self, attempt: int, error: Exception, error_context: ErrorContext) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.max_attempts:
            return False

        return error_context.category not in (ErrorCategory.PERMANENT, ErrorCategory.FATAL)


class ErrorDiagnosisEngine:
    """Automatic error diagnosis and recovery suggestion engine."""

    @staticmethod
    def diagnose(error_context: ErrorContext) -> list[str]:
        """Diagnose error and provide recovery suggestions."""
        suggestions: list[str] = []
        error_type = error_context.error_type
        error_msg = error_context.error_message.lower()

        if "FileNotFoundError" in error_type or "no such file" in error_msg:
            suggestions.extend(
                [
                    "Verify the file path is correct",
                    "Check if the file exists in the expected location",
                    "Ensure you have read permissions for the file",
                ]
            )
            error_context.category = ErrorCategory.PERMANENT

        elif "SyntaxError" in error_type or "syntax" in error_msg:
            suggestions.extend(
                [
                    "Check Python syntax in the target file",
                    "Verify the file is valid Python code",
                    "Look for unclosed brackets, quotes, or parentheses",
                ]
            )
            error_context.category = ErrorCategory.PERMANENT

        elif "PermissionError" in error_type or "permission denied" in error_msg:
            suggestions.extend(
                [
                    "Check file permissions",
                    "Run with appropriate user privileges",
                    "Verify the file is not locked by another process",
                ]
            )
            error_context.category = ErrorCategory.RECOVERABLE

        elif any(x in error_msg for x in ["timeout", "connection", "network"]):
            suggestions.extend(
                [
                    "Check network connectivity",
                    "Verify API endpoint is accessible",
                    "Retry the operation after a delay",
                ]
            )
            error_context.category = ErrorCategory.TRANSIENT

        elif "rate limit" in error_msg or "429" in error_msg:
            suggestions.extend(
                [
                    "Wait before retrying",
                    "Implement exponential backoff",
                    "Check API rate limits and quotas",
                ]
            )
            error_context.category = ErrorCategory.TRANSIENT

        else:
            suggestions.extend(
                [
                    "Review the error message and stack trace",
                    "Check input parameters for validity",
                    "Verify system resources are available",
                ]
            )

        error_context.recovery_suggestions = suggestions
        return suggestions


__all__ = [
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorContext",
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    "RetryStrategy",
    "ErrorDiagnosisEngine",
]
