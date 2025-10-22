"""
Core Agent Configuration

Configuration management for the code agent with validation and defaults.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import UsageLimits

if TYPE_CHECKING:
    from .routing_config import RoutingConfig


@dataclass
class AgentConfig:
    """
    Configuration for CodeAgent.

    Attributes:
        model: Model identifier (e.g., "openai:gpt-4")
        enable_streaming: Enable streaming responses
        enable_logging: Enable structured logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json or human)
        usage_limits: Token usage limits
        max_retries: Maximum retry attempts
        retry_backoff_factor: Exponential backoff factor
        enable_error_recovery: Enable automatic error recovery
        enable_context_management: Enable context window management
        max_context_tokens: Maximum context tokens
        enable_graph_integration: Enable graph workflow integration
        enable_circuit_breaker: Enable circuit breaker pattern
        circuit_breaker_threshold: Circuit breaker failure threshold
    """

    model: str = "openai:gpt-4"
    """Model identifier"""

    enable_streaming: bool = False
    """Enable streaming responses"""

    enable_logging: bool = True
    """Enable structured logging"""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """Logging level"""

    log_format: Literal["json", "human"] = "human"
    """Log format"""

    usage_limits: UsageLimits | None = None
    """Token usage limits"""

    max_retries: int = 3
    """Maximum retry attempts"""

    retry_backoff_factor: float = 2.0
    """Exponential backoff factor"""

    enable_error_recovery: bool = True
    """Enable automatic error recovery"""

    enable_context_management: bool = True
    """Enable context window management"""

    max_context_tokens: int = 100_000
    """Maximum context tokens"""

    enable_graph_integration: bool = True
    """Enable graph workflow integration"""

    enable_circuit_breaker: bool = True
    """Enable circuit breaker pattern"""

    circuit_breaker_threshold: int = 5
    """Circuit breaker failure threshold"""

    routing_config: RoutingConfig | None = None
    """Intelligent routing configuration (optional)"""

    additional_config: dict[str, Any] = field(default_factory=dict)
    """Additional configuration options"""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            "model": self.model,
            "enable_streaming": self.enable_streaming,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "usage_limits": self.usage_limits,
            "max_retries": self.max_retries,
            "retry_backoff_factor": self.retry_backoff_factor,
            "enable_error_recovery": self.enable_error_recovery,
            "enable_context_management": self.enable_context_management,
            "max_context_tokens": self.max_context_tokens,
            "enable_graph_integration": self.enable_graph_integration,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            **self.additional_config,
        }
        if self.routing_config is not None:
            result["routing_config"] = self.routing_config.to_dict()
        return result
