"""
Utilities Module

Common utilities and helpers for the code agent.

Exports:
    - StructuredLogger: Structured logging system
    - create_logger: Logger factory function
    - ErrorContext: Error context information
    - CircuitBreaker: Circuit breaker pattern
    - RetryStrategy: Retry strategy configuration
"""

from __future__ import annotations

from .errors import (
    CircuitBreaker,
    CircuitBreakerError,
    ErrorCategory,
    ErrorContext,
    ErrorDiagnosisEngine,
    ErrorSeverity,
    RetryStrategy,
)
from .logging import (
    LogFormat,
    LogLevel,
    PerformanceMetrics,
    StructuredLogger,
    create_logger,
)

__all__ = [
    # Logging
    "StructuredLogger",
    "LogLevel",
    "LogFormat",
    "create_logger",
    "PerformanceMetrics",
    # Error handling
    "ErrorContext",
    "ErrorCategory",
    "ErrorSeverity",
    "CircuitBreaker",
    "CircuitBreakerError",
    "RetryStrategy",
    "ErrorDiagnosisEngine",
]
