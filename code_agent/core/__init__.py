"""
Code Agent Core Module

Core agent implementation with enhanced PydanticAI features.

Exports:
    - CodeAgent: Main agent class
    - create_code_agent: Factory function for agent creation
    - AgentConfig: Agent configuration
    - HierarchicalAgent: Hierarchical agent with sub-agent support
    - SubAgent: Base sub-agent class
    - AgentRegistry: Agent registration and discovery
    - TaskDelegator: Task delegation system
    - A2A integration components
    - Routing system: Intelligent request routing and model selection
"""

from __future__ import annotations

from .a2a_integration import (
    A2AClient,
    A2AConfig,
    A2AServer,
)
from .agent import CodeAgent, create_code_agent
from .agent_registry import AgentRegistry
from .config import AgentConfig
from .hierarchical_agent import (
    CodeSubAgent,
    HierarchicalAgent,
    HierarchicalAgentConfig,
)
from .model_router import ModelRouter, RoutingDecision
from .prompt_enhancer import PromptEnhancer
from .request_classifier import RequestClassifier
from .routing_config import (
    ClassificationConfig,
    EnhancementConfig,
    RoutingConfig,
    RoutingPolicy,
    create_default_routing_config,
    create_example_routing_config,
)
from .routing_types import (
    DifficultyLevel,
    EnhancementStrategy,
    ModelCapabilities,
    ModelConfig,
    ModelCostProfile,
    PromptEnhancement,
    RequestClassification,
    RequestMode,
    RoutingStrategy,
)
from .sub_agent import (
    DelegatedTask,
    SubAgent,
    SubAgentInfo,
    SubAgentResult,
    SubAgentStatus,
    TaskStatus,
)
from .task_delegator import TaskDelegator
from .telemetry_collector import TelemetryCollector, TelemetryCollectorStats
from .telemetry_config import (
    ConfigValidationError,
    FileExportConfig,
    HTTPExportConfig,
    OpenTelemetryExportConfig,
    PrometheusExportConfig,
    TelemetryConfig,
    TelemetryPrivacyConfig,
    TelemetrySamplingConfig,
    create_default_telemetry_config,
    create_file_telemetry_config,
    create_opentelemetry_config,
    create_prometheus_telemetry_config,
)
from .telemetry_types import (
    ClassificationMetric,
    CostMetric,
    EnhancementMetric,
    ExportFormat,
    MetricAggregation,
    MetricType,
    PerformanceMetric,
    RoutingMetric,
    TelemetryEvent,
    TelemetryEventType,
)
from .types import (
    AgentState,
    AnalysisResult,
    CodeGenerationResult,
    RefactoringResult,
)

__all__ = [
    # Main classes
    "CodeAgent",
    "AgentConfig",
    "HierarchicalAgent",
    "HierarchicalAgentConfig",
    # Factory functions
    "create_code_agent",
    # Types
    "AgentState",
    "AnalysisResult",
    "RefactoringResult",
    "CodeGenerationResult",
    # Sub-agent system
    "SubAgent",
    "CodeSubAgent",
    "SubAgentInfo",
    "SubAgentStatus",
    "DelegatedTask",
    "SubAgentResult",
    "TaskStatus",
    "AgentRegistry",
    "TaskDelegator",
    # A2A integration
    "A2AServer",
    "A2AClient",
    "A2AConfig",
    # Routing system
    "PromptEnhancer",
    "RequestClassifier",
    "ModelRouter",
    "RoutingDecision",
    # Routing configuration
    "RoutingConfig",
    "EnhancementConfig",
    "ClassificationConfig",
    "RoutingPolicy",
    "create_default_routing_config",
    "create_example_routing_config",
    # Routing types
    "DifficultyLevel",
    "RequestMode",
    "RoutingStrategy",
    "EnhancementStrategy",
    "RequestClassification",
    "PromptEnhancement",
    "ModelConfig",
    "ModelCapabilities",
    "ModelCostProfile",
    # Telemetry configuration
    "TelemetryConfig",
    "TelemetryPrivacyConfig",
    "TelemetrySamplingConfig",
    "FileExportConfig",
    "PrometheusExportConfig",
    "OpenTelemetryExportConfig",
    "HTTPExportConfig",
    "ConfigValidationError",
    "create_default_telemetry_config",
    "create_file_telemetry_config",
    "create_prometheus_telemetry_config",
    "create_opentelemetry_config",
    # Telemetry collector
    "TelemetryCollector",
    "TelemetryCollectorStats",
    # Telemetry types
    "TelemetryEvent",
    "TelemetryEventType",
    "MetricType",
    "ExportFormat",
    "RoutingMetric",
    "EnhancementMetric",
    "ClassificationMetric",
    "PerformanceMetric",
    "CostMetric",
    "MetricAggregation",
]
