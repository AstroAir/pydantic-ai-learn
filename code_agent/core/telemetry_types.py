"""
Telemetry Data Models

Defines data structures for telemetry events, metrics, and traces
in the routing system.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ============================================================================
# Enumerations
# ============================================================================


class TelemetryEventType(str, Enum):
    """Types of telemetry events."""

    ROUTING_DECISION = "routing_decision"
    PROMPT_ENHANCEMENT = "prompt_enhancement"
    REQUEST_CLASSIFICATION = "request_classification"
    MODEL_SELECTION = "model_selection"
    COST_CALCULATION = "cost_calculation"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR = "error"


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ExportFormat(str, Enum):
    """Telemetry export formats."""

    OPENTELEMETRY = "opentelemetry"
    PROMETHEUS = "prometheus"
    JSON = "json"
    CSV = "csv"


# ============================================================================
# Core Telemetry Data Models
# ============================================================================


@dataclass
class TelemetryEvent:
    """
    Base telemetry event.

    Attributes:
        event_type: Type of event
        timestamp: Event timestamp
        event_id: Unique event identifier
        attributes: Event attributes
        resource_attributes: Resource-level attributes
    """

    event_type: TelemetryEventType
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time() * 1000000)}")
    attributes: dict[str, Any] = field(default_factory=dict)
    resource_attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "event_id": self.event_id,
            "attributes": self.attributes,
            "resource_attributes": self.resource_attributes,
        }


@dataclass
class RoutingMetric:
    """
    Routing decision metric.

    Tracks metrics for routing decisions including model selection,
    confidence scores, and decision reasoning.
    """

    selected_model: str
    confidence: float
    difficulty_level: str
    request_mode: str
    timestamp: float = field(default_factory=time.time)
    alternatives: list[str] = field(default_factory=list)
    reasoning: str = ""
    fallback_used: bool = False
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected_model": self.selected_model,
            "confidence": self.confidence,
            "difficulty_level": self.difficulty_level,
            "request_mode": self.request_mode,
            "timestamp": self.timestamp,
            "alternatives": self.alternatives,
            "reasoning": self.reasoning,
            "fallback_used": self.fallback_used,
            "dry_run": self.dry_run,
        }


@dataclass
class EnhancementMetric:
    """
    Prompt enhancement metric.

    Tracks effectiveness of prompt enhancements.
    """

    original_length: int
    enhanced_length: int
    confidence: float
    improvements_count: int
    strategy: str
    timestamp: float = field(default_factory=time.time)
    improvements: list[str] = field(default_factory=list)

    @property
    def length_change_percent(self) -> float:
        """Calculate percentage change in length."""
        if self.original_length == 0:
            return 0.0
        return ((self.enhanced_length - self.original_length) / self.original_length) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_length": self.original_length,
            "enhanced_length": self.enhanced_length,
            "length_change_percent": self.length_change_percent,
            "confidence": self.confidence,
            "improvements_count": self.improvements_count,
            "strategy": self.strategy,
            "timestamp": self.timestamp,
            "improvements": self.improvements,
        }


@dataclass
class ClassificationMetric:
    """
    Request classification metric.

    Tracks classification accuracy and performance.
    """

    difficulty_level: str
    request_mode: str
    confidence: float
    requires_tools: bool
    estimated_tokens: int
    timestamp: float = field(default_factory=time.time)
    keywords: list[str] = field(default_factory=list)
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "difficulty_level": self.difficulty_level,
            "request_mode": self.request_mode,
            "confidence": self.confidence,
            "requires_tools": self.requires_tools,
            "estimated_tokens": self.estimated_tokens,
            "timestamp": self.timestamp,
            "keywords": self.keywords,
            "cache_hit": self.cache_hit,
        }


@dataclass
class PerformanceMetric:
    """
    Performance metric for routing components.

    Tracks latency and throughput.
    """

    component: str
    operation: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class CostMetric:
    """
    Cost tracking metric.

    Tracks costs and savings from intelligent routing.
    """

    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    timestamp: float = field(default_factory=time.time)
    currency: str = "USD"
    alternative_model: str | None = None
    alternative_cost: float | None = None

    @property
    def total_cost(self) -> float:
        """Calculate total cost."""
        return self.input_cost + self.output_cost

    @property
    def cost_savings(self) -> float:
        """Calculate cost savings compared to alternative."""
        if self.alternative_cost is None:
            return 0.0
        return self.alternative_cost - self.total_cost

    @property
    def savings_percent(self) -> float:
        """Calculate savings percentage."""
        if self.alternative_cost is None or self.alternative_cost == 0:
            return 0.0
        return (self.cost_savings / self.alternative_cost) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "currency": self.currency,
            "alternative_model": self.alternative_model,
            "alternative_cost": self.alternative_cost,
            "cost_savings": self.cost_savings,
            "savings_percent": self.savings_percent,
            "timestamp": self.timestamp,
        }


@dataclass
class MetricAggregation:
    """
    Aggregated metrics over a time period.

    Provides statistical summaries of metrics.
    """

    metric_name: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    p95_value: float
    p99_value: float
    start_time: float
    end_time: float

    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean_value,
            "median": self.median_value,
            "p95": self.p95_value,
            "p99": self.p99_value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
        }


__all__ = [
    "TelemetryEventType",
    "MetricType",
    "ExportFormat",
    "TelemetryEvent",
    "RoutingMetric",
    "EnhancementMetric",
    "ClassificationMetric",
    "PerformanceMetric",
    "CostMetric",
    "MetricAggregation",
]
