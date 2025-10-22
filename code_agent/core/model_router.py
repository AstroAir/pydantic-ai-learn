"""
Model Router System

Intelligent model selection and routing based on request classification.

Features:
- Multi-model registry and management
- Intelligent routing based on difficulty and mode
- Cost optimization
- Capability matching
- Fallback strategies
- Routing metrics and logging

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .routing_types import (
    ModelConfig,
    RequestClassification,
    RoutingStrategy,
)

if TYPE_CHECKING:
    from ..config.logging import StructuredLogger
    from .routing_config import RoutingConfig


@dataclass
class RoutingDecision:
    """
    Result of model routing decision.

    Attributes:
        selected_model: Selected model identifier
        classification: Request classification
        strategy: Routing strategy used
        reasoning: Explanation of routing decision
        alternatives: Alternative models considered
        estimated_cost: Estimated cost for this routing
        confidence: Confidence in routing decision (0.0 to 1.0)
        metadata: Additional routing metadata
    """

    selected_model: str
    """Selected model"""

    classification: RequestClassification
    """Request classification"""

    strategy: RoutingStrategy
    """Strategy used"""

    reasoning: str = ""
    """Routing reasoning"""

    alternatives: list[str] = field(default_factory=list)
    """Alternative models"""

    estimated_cost: float = 0.0
    """Estimated cost"""

    confidence: float = 0.8
    """Confidence in routing decision"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    timestamp: datetime = field(default_factory=datetime.now)
    """Decision timestamp"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected_model": self.selected_model,
            "classification": self.classification.to_dict(),
            "strategy": self.strategy.value,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "estimated_cost": self.estimated_cost,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelRouter:
    """
    Model routing system for intelligent model selection.

    Routes requests to appropriate models based on:
    - Request difficulty
    - Request mode (chat vs agent)
    - Model capabilities
    - Cost optimization
    - Custom routing policies
    """

    def __init__(
        self,
        config: RoutingConfig,
        logger: StructuredLogger | None = None,
    ) -> None:
        """
        Initialize model router.

        Args:
            config: Routing configuration
            logger: Optional logger
        """
        self.config = config
        self.logger = logger
        self._model_registry: dict[str, ModelConfig] = {}
        self._routing_metrics: dict[str, int] = {}

        # Build model registry
        self._build_registry()

    def _build_registry(self) -> None:
        """Build model registry from configuration."""
        for model_config in self.config.models:
            if model_config.enabled:
                self._model_registry[model_config.name] = model_config

        if self.logger:
            self.logger.info(
                "Model registry built",
                extra={"model_count": len(self._model_registry), "models": list(self._model_registry.keys())},
            )

    def route(
        self,
        classification: RequestClassification,
        default_model: str | None = None,
    ) -> RoutingDecision:
        """
        Route request to appropriate model.

        Args:
            classification: Request classification
            default_model: Default model to use if routing fails

        Returns:
            Routing decision
        """
        if not self.config.enabled or not self._model_registry:
            # Routing disabled or no models configured
            fallback = default_model or self.config.policy.fallback_model or "openai:gpt-4"
            return RoutingDecision(
                selected_model=fallback,
                classification=classification,
                strategy=RoutingStrategy.DIFFICULTY_BASED,
                reasoning="Routing disabled or no models configured",
            )

        # Select routing strategy
        strategy = self.config.policy.strategy

        # Route based on strategy
        if strategy == RoutingStrategy.DIFFICULTY_BASED:
            decision = self._route_by_difficulty(classification)
        elif strategy == RoutingStrategy.MODE_BASED:
            decision = self._route_by_mode(classification)
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            decision = self._route_by_cost(classification)
        elif strategy == RoutingStrategy.CAPABILITY_MATCHED:
            decision = self._route_by_capability(classification)
        elif strategy == RoutingStrategy.HYBRID:
            decision = self._route_hybrid(classification)
        else:
            decision = self._route_by_difficulty(classification)

        # Apply fallback if needed
        if decision.selected_model not in self._model_registry and self.config.policy.enable_fallback:
            fallback = self.config.policy.fallback_model or default_model or "openai:gpt-4"
            decision.selected_model = fallback
            decision.reasoning += f" (fallback to {fallback})"

        # Update metrics
        self._update_metrics(decision)

        # Log decision
        if self.logger and self.config.enable_logging:
            self.logger.info(
                "Model routed",
                extra={
                    "model": decision.selected_model,
                    "difficulty": classification.difficulty.value,
                    "mode": classification.mode.value,
                    "strategy": strategy.value,
                    "confidence": classification.confidence,
                },
            )

        return decision

    def _route_by_difficulty(self, classification: RequestClassification) -> RoutingDecision:
        """Route based on difficulty level."""
        difficulty_key = classification.difficulty.value
        preferred_model = self.config.policy.difficulty_model_map.get(difficulty_key)

        if preferred_model and preferred_model in self._model_registry:
            return RoutingDecision(
                selected_model=preferred_model,
                classification=classification,
                strategy=RoutingStrategy.DIFFICULTY_BASED,
                reasoning=f"Selected based on {difficulty_key} difficulty",
            )

        # Find suitable model
        candidates = [
            model for model in self._model_registry.values() if classification.difficulty in model.difficulty_levels
        ]

        if candidates:
            # Select highest priority
            selected = max(candidates, key=lambda m: m.priority)
            return RoutingDecision(
                selected_model=selected.name,
                classification=classification,
                strategy=RoutingStrategy.DIFFICULTY_BASED,
                reasoning=f"Selected {selected.name} for {difficulty_key} difficulty",
                alternatives=[m.name for m in candidates if m.name != selected.name],
            )

        # No suitable model found
        return RoutingDecision(
            selected_model=self.config.policy.fallback_model or "openai:gpt-4",
            classification=classification,
            strategy=RoutingStrategy.DIFFICULTY_BASED,
            reasoning="No suitable model found for difficulty level",
        )

    def _route_by_mode(self, classification: RequestClassification) -> RoutingDecision:
        """Route based on request mode."""
        mode_key = classification.mode.value
        preferred_model = self.config.policy.mode_model_map.get(mode_key)

        if preferred_model and preferred_model in self._model_registry:
            return RoutingDecision(
                selected_model=preferred_model,
                classification=classification,
                strategy=RoutingStrategy.MODE_BASED,
                reasoning=f"Selected based on {mode_key} mode",
            )

        # Find suitable model
        candidates = [model for model in self._model_registry.values() if classification.mode in model.modes]

        if candidates:
            selected = max(candidates, key=lambda m: m.priority)
            return RoutingDecision(
                selected_model=selected.name,
                classification=classification,
                strategy=RoutingStrategy.MODE_BASED,
                reasoning=f"Selected {selected.name} for {mode_key} mode",
                alternatives=[m.name for m in candidates if m.name != selected.name],
            )

        return RoutingDecision(
            selected_model=self.config.policy.fallback_model or "openai:gpt-4",
            classification=classification,
            strategy=RoutingStrategy.MODE_BASED,
            reasoning="No suitable model found for mode",
        )

    def _route_by_cost(self, classification: RequestClassification) -> RoutingDecision:
        """Route based on cost optimization."""
        # Find suitable models
        candidates = [
            model
            for model in self._model_registry.values()
            if classification.difficulty in model.difficulty_levels and classification.mode in model.modes
        ]

        if not candidates:
            return self._route_by_difficulty(classification)

        # Calculate estimated costs
        estimated_tokens = classification.estimated_tokens or 1000
        costs = []
        for model in candidates:
            input_cost = (estimated_tokens / 1000) * model.cost_profile.input_cost_per_1k
            output_cost = (estimated_tokens / 1000) * model.cost_profile.output_cost_per_1k
            total_cost = input_cost + output_cost
            costs.append((model, total_cost))

        # Select cheapest
        selected, cost = min(costs, key=lambda x: x[1])

        return RoutingDecision(
            selected_model=selected.name,
            classification=classification,
            strategy=RoutingStrategy.COST_OPTIMIZED,
            reasoning=f"Selected {selected.name} for cost optimization (est. ${cost:.4f})",
            estimated_cost=cost,
            alternatives=[m.name for m, _ in costs if m.name != selected.name],
        )

    def _route_by_capability(self, classification: RequestClassification) -> RoutingDecision:
        """Route based on capability matching."""
        # Filter by required capabilities
        candidates = list(self._model_registry.values())

        # Filter by tool support if needed
        if classification.requires_tools:
            candidates = [m for m in candidates if m.capabilities.supports_tools]

        # Filter by streaming preference
        if self.config.policy.prefer_streaming:
            streaming_models = [m for m in candidates if m.capabilities.supports_streaming]
            if streaming_models:
                candidates = streaming_models

        # Filter by difficulty and mode
        candidates = [
            m for m in candidates if classification.difficulty in m.difficulty_levels and classification.mode in m.modes
        ]

        if candidates:
            selected = max(candidates, key=lambda m: m.priority)
            return RoutingDecision(
                selected_model=selected.name,
                classification=classification,
                strategy=RoutingStrategy.CAPABILITY_MATCHED,
                reasoning=f"Selected {selected.name} based on capability matching",
                alternatives=[m.name for m in candidates if m.name != selected.name],
            )

        return self._route_by_difficulty(classification)

    def _route_hybrid(self, classification: RequestClassification) -> RoutingDecision:
        """Route using hybrid strategy combining multiple factors."""
        # Start with difficulty-based routing
        difficulty_decision = self._route_by_difficulty(classification)

        # If confidence is high, use it
        if classification.confidence >= 0.8:
            return difficulty_decision

        # Otherwise, try cost optimization
        cost_decision = self._route_by_cost(classification)

        # Combine reasoning
        return RoutingDecision(
            selected_model=cost_decision.selected_model,
            classification=classification,
            strategy=RoutingStrategy.HYBRID,
            reasoning=f"Hybrid: {cost_decision.reasoning}; fallback from {difficulty_decision.selected_model}",
            alternatives=list(set(difficulty_decision.alternatives + cost_decision.alternatives)),
            estimated_cost=cost_decision.estimated_cost,
        )

    def _update_metrics(self, decision: RoutingDecision) -> None:
        """Update routing metrics."""
        if not self.config.enable_metrics:
            return

        model_key = decision.selected_model
        self._routing_metrics[model_key] = self._routing_metrics.get(model_key, 0) + 1

    def get_metrics(self) -> dict[str, int]:
        """Get routing metrics."""
        return self._routing_metrics.copy()

    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        return list(self._model_registry.keys())
