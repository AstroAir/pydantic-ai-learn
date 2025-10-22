"""
Routing System Type Definitions

Data models for intelligent request routing, classification, and prompt enhancement.

Features:
- Request difficulty classification
- Chat vs Agent mode detection
- Multi-model configuration
- Routing policies and strategies
- Prompt enhancement metadata

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ============================================================================
# Enumerations
# ============================================================================


class DifficultyLevel(str, Enum):
    """Request difficulty classification levels."""

    SIMPLE = "simple"
    """Simple queries: greetings, basic questions, straightforward requests"""

    MODERATE = "moderate"
    """Moderate complexity: multi-step reasoning, code analysis, explanations"""

    COMPLEX = "complex"
    """Complex tasks: code generation, refactoring, multi-file analysis"""


class RequestMode(str, Enum):
    """Request processing mode."""

    CHAT = "chat"
    """Chat mode: conversational, Q&A, explanations"""

    AGENT = "agent"
    """Agent mode: tool usage, code manipulation, multi-step operations"""


class RoutingStrategy(str, Enum):
    """Model selection strategy."""

    DIFFICULTY_BASED = "difficulty_based"
    """Route based on difficulty level"""

    MODE_BASED = "mode_based"
    """Route based on request mode (chat vs agent)"""

    COST_OPTIMIZED = "cost_optimized"
    """Optimize for cost while meeting requirements"""

    CAPABILITY_MATCHED = "capability_matched"
    """Match model capabilities to request requirements"""

    HYBRID = "hybrid"
    """Combine multiple strategies"""


class EnhancementStrategy(str, Enum):
    """Prompt enhancement strategy."""

    NONE = "none"
    """No enhancement"""

    RULE_BASED = "rule_based"
    """Rule-based enhancement using heuristics"""

    LLM_BASED = "llm_based"
    """LLM-based enhancement using lightweight model"""

    HYBRID = "hybrid"
    """Combine rule-based and LLM-based"""


# ============================================================================
# Request Classification
# ============================================================================


@dataclass
class RequestClassification:
    """
    Classification result for a user request.

    Attributes:
        difficulty: Assessed difficulty level
        mode: Detected request mode (chat vs agent)
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Explanation of classification
        requires_tools: Whether request requires tool usage
        estimated_tokens: Estimated token usage
        keywords: Extracted keywords
        metadata: Additional classification metadata
    """

    difficulty: DifficultyLevel
    """Difficulty level"""

    mode: RequestMode
    """Request mode"""

    confidence: float = 0.0
    """Confidence score (0.0 to 1.0)"""

    reasoning: str = ""
    """Classification reasoning"""

    requires_tools: bool = False
    """Whether tools are required"""

    estimated_tokens: int = 0
    """Estimated token usage"""

    keywords: list[str] = field(default_factory=list)
    """Extracted keywords"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    timestamp: datetime = field(default_factory=datetime.now)
    """Classification timestamp"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "difficulty": self.difficulty.value,
            "mode": self.mode.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "requires_tools": self.requires_tools,
            "estimated_tokens": self.estimated_tokens,
            "keywords": self.keywords,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Prompt Enhancement
# ============================================================================


@dataclass
class PromptEnhancement:
    """
    Result of prompt enhancement.

    Attributes:
        original_prompt: Original user prompt
        enhanced_prompt: Enhanced prompt
        strategy: Enhancement strategy used
        improvements: List of improvements made
        confidence: Enhancement confidence (0.0 to 1.0)
        metadata: Additional enhancement metadata
    """

    original_prompt: str
    """Original prompt"""

    enhanced_prompt: str
    """Enhanced prompt"""

    strategy: EnhancementStrategy
    """Strategy used"""

    improvements: list[str] = field(default_factory=list)
    """Improvements made"""

    confidence: float = 0.0
    """Enhancement confidence"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    timestamp: datetime = field(default_factory=datetime.now)
    """Enhancement timestamp"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_prompt": self.original_prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "strategy": self.strategy.value,
            "improvements": self.improvements,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Model Configuration
# ============================================================================


@dataclass
class ModelCapabilities:
    """
    Model capability metadata.

    Attributes:
        supports_tools: Whether model supports tool calling
        supports_streaming: Whether model supports streaming
        max_tokens: Maximum context tokens
        supports_vision: Whether model supports vision
        supports_code: Whether model is optimized for code
        custom_capabilities: Custom capability flags
    """

    supports_tools: bool = True
    supports_streaming: bool = True
    max_tokens: int = 100_000
    supports_vision: bool = False
    supports_code: bool = True
    custom_capabilities: dict[str, bool] = field(default_factory=dict)


@dataclass
class ModelCostProfile:
    """
    Model cost profile.

    Attributes:
        input_cost_per_1k: Cost per 1K input tokens
        output_cost_per_1k: Cost per 1K output tokens
        currency: Currency code (e.g., "USD")
    """

    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    currency: str = "USD"


@dataclass
class ModelConfig:
    """
    Configuration for a single model.

    Attributes:
        name: Model identifier (e.g., "openai:gpt-4")
        display_name: Human-readable name
        capabilities: Model capabilities
        cost_profile: Cost information
        difficulty_levels: Supported difficulty levels
        modes: Supported request modes
        priority: Selection priority (higher = preferred)
        enabled: Whether model is enabled
        metadata: Additional model metadata
    """

    name: str
    """Model identifier"""

    display_name: str = ""
    """Display name"""

    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    """Capabilities"""

    cost_profile: ModelCostProfile = field(default_factory=ModelCostProfile)
    """Cost profile"""

    difficulty_levels: list[DifficultyLevel] = field(
        default_factory=lambda: [DifficultyLevel.SIMPLE, DifficultyLevel.MODERATE, DifficultyLevel.COMPLEX]
    )
    """Supported difficulty levels"""

    modes: list[RequestMode] = field(default_factory=lambda: [RequestMode.CHAT, RequestMode.AGENT])
    """Supported modes"""

    priority: int = 0
    """Selection priority"""

    enabled: bool = True
    """Whether enabled"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "capabilities": {
                "supports_tools": self.capabilities.supports_tools,
                "supports_streaming": self.capabilities.supports_streaming,
                "max_tokens": self.capabilities.max_tokens,
                "supports_vision": self.capabilities.supports_vision,
                "supports_code": self.capabilities.supports_code,
                "custom_capabilities": self.capabilities.custom_capabilities,
            },
            "cost_profile": {
                "input_cost_per_1k": self.cost_profile.input_cost_per_1k,
                "output_cost_per_1k": self.cost_profile.output_cost_per_1k,
                "currency": self.cost_profile.currency,
            },
            "difficulty_levels": [d.value for d in self.difficulty_levels],
            "modes": [m.value for m in self.modes],
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }
