"""
Structured Logging System

Comprehensive logging with JSON formatting, performance metrics, and sanitization.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format enumeration."""

    JSON = "json"
    HUMAN = "human"


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class HumanFormatter(logging.Formatter):
    """Human-readable log formatter."""

    def __init__(self) -> None:
        """Initialize formatter with custom format."""
        super().__init__(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""

    operation_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(self, success: bool = True, error: str | None = None) -> None:
        """Mark operation as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "operation": self.operation_name,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


class StructuredLogger:
    """Structured logger with JSON formatting and performance tracking."""

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        format_type: LogFormat = LogFormat.HUMAN,
        log_file: Path | None = None,
        sanitize_logs: bool = True,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            level: Log level
            format_type: Output format (JSON or human-readable)
            log_file: Optional file path for logging
            sanitize_logs: Whether to sanitize sensitive data
        """
        self.name = name
        self.sanitize_logs = sanitize_logs
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        self.logger.handlers.clear()

        formatter: logging.Formatter
        formatter = JSONFormatter() if format_type == LogFormat.JSON else HumanFormatter()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.metrics: list[PerformanceMetrics] = []

    def _sanitize(self, message: str) -> str:
        """Sanitize sensitive data from log messages."""
        if not self.sanitize_logs:
            return message

        import re

        patterns = [
            (r'(api[_-]?key|token|password|secret)["\s:=]+["\']?[\w-]+', r"\1=***"),
            (r"Bearer\s+[\w-]+", "Bearer ***"),
            (r"sk-[\w-]+", "sk-***"),
        ]

        sanitized = message
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized

    def debug(self, message: str, **extra: Any) -> None:
        """Log debug message."""
        self.logger.debug(self._sanitize(message), extra={"extra_fields": extra})

    def info(self, message: str, **extra: Any) -> None:
        """Log info message."""
        self.logger.info(self._sanitize(message), extra={"extra_fields": extra})

    def warning(self, message: str, **extra: Any) -> None:
        """Log warning message."""
        self.logger.warning(self._sanitize(message), extra={"extra_fields": extra})

    def error(self, message: str, **extra: Any) -> None:
        """Log error message."""
        self.logger.error(self._sanitize(message), extra={"extra_fields": extra})

    def critical(self, message: str, **extra: Any) -> None:
        """Log critical message."""
        self.logger.critical(self._sanitize(message), extra={"extra_fields": extra})

    def start_operation(self, operation_name: str, **metadata: Any) -> PerformanceMetrics:
        """Start tracking an operation."""
        metrics = PerformanceMetrics(operation_name=operation_name, metadata=metadata)
        self.metrics.append(metrics)
        self.debug(f"Started operation: {operation_name}", **metadata)
        return metrics

    def complete_operation(self, metrics: PerformanceMetrics, success: bool = True, error: str | None = None) -> None:
        """Complete operation tracking."""
        metrics.complete(success=success, error=error)

        if success:
            self.info(
                f"Completed operation: {metrics.operation_name}", duration_ms=metrics.duration_ms, **metrics.metadata
            )
        else:
            self.error(
                f"Failed operation: {metrics.operation_name}",
                duration_ms=metrics.duration_ms,
                error=error,
                **metrics.metadata,
            )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all tracked metrics."""
        if not self.metrics:
            return {"total_operations": 0}

        successful = [m for m in self.metrics if m.success]
        failed = [m for m in self.metrics if not m.success]

        return {
            "total_operations": len(self.metrics),
            "successful": len(successful),
            "failed": len(failed),
            "avg_duration_ms": sum(m.duration_ms or 0 for m in self.metrics) / len(self.metrics),
            "total_input_tokens": sum(m.input_tokens for m in self.metrics),
            "total_output_tokens": sum(m.output_tokens for m in self.metrics),
        }


def create_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.HUMAN,
    log_file: Path | None = None,
    sanitize_logs: bool = True,
) -> StructuredLogger:
    """
    Create a structured logger instance.

    Args:
        name: Logger name
        level: Log level
        format_type: Output format
        log_file: Optional log file path
        sanitize_logs: Whether to sanitize sensitive data

    Returns:
        Configured structured logger
    """
    return StructuredLogger(
        name=name, level=level, format_type=format_type, log_file=log_file, sanitize_logs=sanitize_logs
    )


__all__ = [
    "LogLevel",
    "LogFormat",
    "StructuredLogger",
    "PerformanceMetrics",
    "create_logger",
]
