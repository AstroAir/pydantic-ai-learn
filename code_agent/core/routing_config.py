"""
Routing Configuration

Configuration management for intelligent request routing system.

Features:
- Multi-model configuration
- Routing policies and strategies
- Prompt enhancement settings
- Classification thresholds
- Feature flags

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .routing_types import (
    DifficultyLevel,
    EnhancementStrategy,
    ModelConfig,
    RequestMode,
    RoutingStrategy,
)

if TYPE_CHECKING:
    from .telemetry_config import TelemetryConfig


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class EnhancementConfig:
    """
    Prompt enhancement configuration.

    Attributes:
        enabled: Enable prompt enhancement
        strategy: Enhancement strategy to use
        min_confidence: Minimum confidence to apply enhancement
        max_length_increase: Maximum allowed prompt length increase (%)
        preserve_original: Keep original prompt in metadata
        enhancement_model: Model to use for LLM-based enhancement
        custom_rules: Custom enhancement rules
    """

    enabled: bool = False
    """Enable enhancement"""

    strategy: EnhancementStrategy = EnhancementStrategy.RULE_BASED
    """Enhancement strategy"""

    min_confidence: float = 0.7
    """Minimum confidence threshold"""

    max_length_increase: float = 0.5
    """Max length increase (50%)"""

    preserve_original: bool = True
    """Preserve original prompt"""

    enhancement_model: str | None = None
    """Model for LLM-based enhancement"""

    custom_rules: dict[str, Any] = field(default_factory=dict)
    """Custom enhancement rules"""


@dataclass
class ClassificationConfig:
    """
    Request classification configuration.

    Attributes:
        enabled: Enable auto-classification
        min_confidence: Minimum confidence threshold
        use_llm: Use LLM for classification (vs heuristics)
        classification_model: Model to use for LLM-based classification
        difficulty_keywords: Keywords for difficulty detection
        mode_keywords: Keywords for mode detection
        cache_ttl: Classification cache TTL in seconds
    """

    enabled: bool = False
    """Enable classification"""

    min_confidence: float = 0.6
    """Minimum confidence threshold"""

    use_llm: bool = False
    """Use LLM for classification"""

    classification_model: str | None = None
    """Classification model"""

    difficulty_keywords: dict[str, list[str]] = field(
        default_factory=lambda: {
            "simple": ["what", "explain", "tell me", "hello", "hi", "thanks"],
            "moderate": ["analyze", "review", "check", "find", "show"],
            "complex": ["refactor", "generate", "create", "implement", "optimize", "debug"],
        }
    )
    """Difficulty keywords"""

    mode_keywords: dict[str, list[str]] = field(
        default_factory=lambda: {
            "chat": ["what", "why", "how", "explain", "tell me", "describe"],
            "agent": ["create", "generate", "refactor", "fix", "implement", "modify", "delete"],
        }
    )
    """Mode keywords"""

    cache_ttl: int = 300
    """Cache TTL in seconds"""


@dataclass
class RoutingPolicy:
    """
    Model routing policy.

    Attributes:
        strategy: Routing strategy
        fallback_model: Fallback model if routing fails
        enable_fallback: Enable fallback on errors
        difficulty_model_map: Map difficulty to preferred models
        mode_model_map: Map mode to preferred models
        cost_threshold: Maximum cost threshold
        prefer_streaming: Prefer streaming-capable models
        prefer_tools: Prefer tool-capable models
    """

    strategy: RoutingStrategy = RoutingStrategy.DIFFICULTY_BASED
    """Routing strategy"""

    fallback_model: str | None = None
    """Fallback model"""

    enable_fallback: bool = True
    """Enable fallback"""

    difficulty_model_map: dict[str, str] = field(
        default_factory=lambda: {
            "simple": "openai:gpt-4o-mini",
            "moderate": "openai:gpt-4o",
            "complex": "openai:gpt-4",
        }
    )
    """Difficulty to model mapping"""

    mode_model_map: dict[str, str] = field(
        default_factory=lambda: {
            "chat": "openai:gpt-4o-mini",
            "agent": "openai:gpt-4",
        }
    )
    """Mode to model mapping"""

    cost_threshold: float | None = None
    """Max cost threshold"""

    prefer_streaming: bool = False
    """Prefer streaming models"""

    prefer_tools: bool = False
    """Prefer tool-capable models"""


@dataclass
class RoutingConfig:
    """
    Complete routing system configuration.

    Attributes:
        enabled: Enable routing system
        models: Available model configurations
        enhancement: Prompt enhancement config
        classification: Request classification config
        policy: Routing policy
        enable_metrics: Enable routing metrics
        enable_logging: Enable routing logging
        dry_run: Dry run mode (log decisions without routing)
    """

    enabled: bool = False
    """Enable routing"""

    models: list[ModelConfig] = field(default_factory=list)
    """Model configurations"""

    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    """Enhancement config"""

    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    """Classification config"""

    policy: RoutingPolicy = field(default_factory=RoutingPolicy)
    """Routing policy"""

    enable_metrics: bool = True
    """Enable metrics"""

    enable_logging: bool = True
    """Enable logging"""

    dry_run: bool = False
    """Dry run mode"""

    telemetry_config: TelemetryConfig | None = None
    """Telemetry configuration"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "models": [m.to_dict() for m in self.models],
            "enhancement": {
                "enabled": self.enhancement.enabled,
                "strategy": self.enhancement.strategy.value,
                "min_confidence": self.enhancement.min_confidence,
                "max_length_increase": self.enhancement.max_length_increase,
                "preserve_original": self.enhancement.preserve_original,
                "enhancement_model": self.enhancement.enhancement_model,
                "custom_rules": self.enhancement.custom_rules,
            },
            "classification": {
                "enabled": self.classification.enabled,
                "min_confidence": self.classification.min_confidence,
                "use_llm": self.classification.use_llm,
                "classification_model": self.classification.classification_model,
                "difficulty_keywords": self.classification.difficulty_keywords,
                "mode_keywords": self.classification.mode_keywords,
                "cache_ttl": self.classification.cache_ttl,
            },
            "policy": {
                "strategy": self.policy.strategy.value,
                "fallback_model": self.policy.fallback_model,
                "enable_fallback": self.policy.enable_fallback,
                "difficulty_model_map": self.policy.difficulty_model_map,
                "mode_model_map": self.policy.mode_model_map,
                "cost_threshold": self.policy.cost_threshold,
                "prefer_streaming": self.policy.prefer_streaming,
                "prefer_tools": self.policy.prefer_tools,
            },
            "enable_metrics": self.enable_metrics,
            "enable_logging": self.enable_logging,
            "dry_run": self.dry_run,
        }


# ============================================================================
# Configuration Helpers
# ============================================================================


def create_default_routing_config() -> RoutingConfig:
    """
    Create default routing configuration with sensible defaults.

    Returns:
        Default routing configuration (disabled but with default models)
    """
    from .routing_types import ModelCapabilities, ModelCostProfile

    # Include default models even when disabled
    default_models = [
        ModelConfig(
            name="openai:gpt-4o-mini",
            display_name="GPT-4o Mini",
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                max_tokens=128_000,
                supports_code=True,
            ),
            cost_profile=ModelCostProfile(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
            difficulty_levels=[DifficultyLevel.SIMPLE, DifficultyLevel.MODERATE],
            modes=[RequestMode.CHAT, RequestMode.AGENT],
            priority=1,
        ),
    ]

    return RoutingConfig(
        enabled=False,
        models=default_models,
        enhancement=EnhancementConfig(enabled=True),  # Enable enhancement by default
        classification=ClassificationConfig(enabled=True),  # Enable classification by default
        policy=RoutingPolicy(),
    )


def create_example_routing_config() -> RoutingConfig:
    """
    Create example routing configuration with multiple models.

    Returns:
        Example routing configuration
    """
    from .routing_types import ModelCapabilities, ModelCostProfile

    models = [
        ModelConfig(
            name="openai:gpt-4o-mini",
            display_name="GPT-4o Mini",
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                max_tokens=128_000,
                supports_code=True,
            ),
            cost_profile=ModelCostProfile(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
            difficulty_levels=[DifficultyLevel.SIMPLE, DifficultyLevel.MODERATE],
            modes=[RequestMode.CHAT, RequestMode.AGENT],
            priority=1,
        ),
        ModelConfig(
            name="openai:gpt-4o",
            display_name="GPT-4o",
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                max_tokens=128_000,
                supports_code=True,
            ),
            cost_profile=ModelCostProfile(input_cost_per_1k=0.0025, output_cost_per_1k=0.01),
            difficulty_levels=[DifficultyLevel.MODERATE, DifficultyLevel.COMPLEX],
            modes=[RequestMode.CHAT, RequestMode.AGENT],
            priority=2,
        ),
        ModelConfig(
            name="openai:gpt-4",
            display_name="GPT-4",
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                max_tokens=128_000,
                supports_code=True,
            ),
            cost_profile=ModelCostProfile(input_cost_per_1k=0.03, output_cost_per_1k=0.06),
            difficulty_levels=[DifficultyLevel.COMPLEX],
            modes=[RequestMode.AGENT],
            priority=3,
        ),
    ]

    return RoutingConfig(
        enabled=True,
        models=models,
        enhancement=EnhancementConfig(enabled=True, strategy=EnhancementStrategy.RULE_BASED),
        classification=ClassificationConfig(enabled=True, use_llm=False),
        policy=RoutingPolicy(
            strategy=RoutingStrategy.HYBRID,
            fallback_model="openai:gpt-4o",
            enable_fallback=True,
        ),
    )
