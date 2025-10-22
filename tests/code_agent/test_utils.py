"""
Utilities Tests

Tests for utility modules.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import pytest

from code_agent.utils import (
    CircuitBreaker,
    CircuitBreakerError,
    ErrorCategory,
    ErrorContext,
    ErrorDiagnosisEngine,
    ErrorSeverity,
    LogFormat,
    LogLevel,
    RetryStrategy,
    StructuredLogger,
)


class TestStructuredLogger:
    """Test StructuredLogger."""

    def test_logger_creation(self) -> None:
        """Test logger creation."""
        logger = StructuredLogger(
            name="test",
            level=LogLevel.INFO,
            format_type=LogFormat.HUMAN,
        )

        assert logger.name == "test"
        assert logger.sanitize_logs is True

    def test_logger_sanitization(self) -> None:
        """Test log sanitization."""
        logger = StructuredLogger(
            name="test",
            sanitize_logs=True,
        )

        message = "api_key=secret123"
        sanitized = logger._sanitize(message)

        assert "secret123" not in sanitized
        assert "***" in sanitized

    def test_logger_metrics(self) -> None:
        """Test logger metrics tracking."""
        logger = StructuredLogger(name="test")

        metrics = logger.start_operation("test_op")
        metrics.complete(success=True)

        summary = logger.get_metrics_summary()

        assert summary["total_operations"] == 1
        assert summary["successful"] == 1


class TestErrorContext:
    """Test ErrorContext."""

    def test_error_context_creation(self) -> None:
        """Test error context creation."""
        context = ErrorContext(
            error_type="ValueError",
            error_message="Invalid value",
            category=ErrorCategory.PERMANENT,
            severity=ErrorSeverity.HIGH,
        )

        assert context.error_type == "ValueError"
        assert context.category == ErrorCategory.PERMANENT

    def test_error_context_from_exception(self) -> None:
        """Test creating error context from exception."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = ErrorContext.from_exception(e)

            assert context.error_type == "ValueError"
            assert "Test error" in context.error_message
            assert context.stack_trace is not None

    def test_error_context_to_dict(self) -> None:
        """Test converting error context to dictionary."""
        context = ErrorContext(
            error_type="ValueError",
            error_message="Invalid value",
            category=ErrorCategory.PERMANENT,
            severity=ErrorSeverity.HIGH,
        )

        context_dict = context.to_dict()

        assert context_dict["error_type"] == "ValueError"
        assert context_dict["category"] == "permanent"


class TestCircuitBreaker:
    """Test CircuitBreaker."""

    def test_circuit_breaker_creation(self) -> None:
        """Test circuit breaker creation."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=3,
        )

        assert breaker.name == "test"
        assert breaker.failure_threshold == 3

    def test_circuit_breaker_success(self) -> None:
        """Test circuit breaker with successful call."""
        breaker = CircuitBreaker(name="test")

        def success_func():
            return "success"

        result = breaker.call(success_func)

        assert result == "success"

    def test_circuit_breaker_failure(self) -> None:
        """Test circuit breaker with failures."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        def fail_func():
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Second failure
        with pytest.raises(ValueError):
            breaker.call(fail_func)

        # Circuit should be open now
        with pytest.raises(CircuitBreakerError):
            breaker.call(fail_func)

    def test_circuit_breaker_reset(self) -> None:
        """Test circuit breaker reset."""
        breaker = CircuitBreaker(name="test")
        breaker.failure_count = 5

        breaker.reset()

        assert breaker.failure_count == 0


class TestRetryStrategy:
    """Test RetryStrategy."""

    def test_retry_strategy_creation(self) -> None:
        """Test retry strategy creation."""
        strategy = RetryStrategy(
            max_attempts=5,
            base_delay=2.0,
        )

        assert strategy.max_attempts == 5
        assert strategy.base_delay == 2.0

    def test_calculate_delay(self) -> None:
        """Test delay calculation."""
        strategy = RetryStrategy(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )

        delay_0 = strategy.calculate_delay(0)
        delay_1 = strategy.calculate_delay(1)
        delay_2 = strategy.calculate_delay(2)

        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 4.0

    def test_should_retry(self) -> None:
        """Test retry decision."""
        strategy = RetryStrategy(max_attempts=3)
        context = ErrorContext(
            error_type="ValueError",
            error_message="Test",
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.MEDIUM,
        )

        # Should retry for transient errors
        assert strategy.should_retry(0, ValueError(), context) is True

        # Should not retry after max attempts
        assert strategy.should_retry(3, ValueError(), context) is False


class TestErrorDiagnosisEngine:
    """Test ErrorDiagnosisEngine."""

    def test_diagnose_file_not_found(self) -> None:
        """Test diagnosing FileNotFoundError."""
        context = ErrorContext(
            error_type="FileNotFoundError",
            error_message="File not found",
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.MEDIUM,
        )

        suggestions = ErrorDiagnosisEngine.diagnose(context)

        assert len(suggestions) > 0
        assert context.category == ErrorCategory.PERMANENT

    def test_diagnose_syntax_error(self) -> None:
        """Test diagnosing SyntaxError."""
        context = ErrorContext(
            error_type="SyntaxError",
            error_message="Invalid syntax",
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.MEDIUM,
        )

        suggestions = ErrorDiagnosisEngine.diagnose(context)

        assert len(suggestions) > 0
        assert context.category == ErrorCategory.PERMANENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
