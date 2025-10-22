"""
Example: Using the Routing System

This example demonstrates how to use the intelligent routing system
with prompt enhancement, request classification, and model selection.

Features demonstrated:
- Prompt enhancement for clarity
- Request classification (difficulty and mode)
- Intelligent model routing
- Configuration management
- Metrics and logging

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from code_agent.core import (
    DifficultyLevel,
    EnhancementConfig,
    EnhancementStrategy,
    ModelCapabilities,
    ModelConfig,
    ModelCostProfile,
    RequestMode,
    RoutingConfig,
    RoutingPolicy,
    RoutingStrategy,
    create_code_agent,
    create_default_routing_config,
)


def example_basic_routing() -> None:
    """Example 1: Basic routing with default configuration."""
    print("=" * 80)
    print("Example 1: Basic Routing with Default Configuration")
    print("=" * 80)

    # Create default routing config
    routing_config = create_default_routing_config()
    routing_config.enabled = True

    # Create agent with routing
    agent = create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    # Run a simple query
    result = agent.run_sync("What is the purpose of this codebase?")
    print(f"\nResult: {result.data}")

    # Check routing metrics
    if agent.model_router:
        print(f"\nRouting Metrics: {agent.model_router.get_metrics()}")


def example_custom_routing() -> None:
    """Example 2: Custom routing configuration with multiple models."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Routing with Multiple Models")
    print("=" * 80)

    # Define custom models
    models = [
        ModelConfig(
            name="openai:gpt-4o-mini",
            enabled=True,
            difficulty_levels=[DifficultyLevel.SIMPLE, DifficultyLevel.MODERATE],
            modes=[RequestMode.CHAT, RequestMode.AGENT],
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                max_tokens=128000,
                supports_vision=False,
            ),
            cost_profile=ModelCostProfile(
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
                currency="USD",
            ),
            priority=1,
        ),
        ModelConfig(
            name="openai:gpt-4o",
            enabled=True,
            difficulty_levels=[DifficultyLevel.MODERATE, DifficultyLevel.COMPLEX],
            modes=[RequestMode.CHAT, RequestMode.AGENT],
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                max_tokens=128000,
                supports_vision=True,
            ),
            cost_profile=ModelCostProfile(
                input_cost_per_1k=0.0025,
                output_cost_per_1k=0.01,
                currency="USD",
            ),
            priority=2,
        ),
        ModelConfig(
            name="openai:gpt-4",
            enabled=True,
            difficulty_levels=[DifficultyLevel.COMPLEX],
            modes=[RequestMode.AGENT],
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_streaming=True,
                max_tokens=128000,
                supports_vision=False,
            ),
            cost_profile=ModelCostProfile(
                input_cost_per_1k=0.03,
                output_cost_per_1k=0.06,
                currency="USD",
            ),
            priority=3,
        ),
    ]

    # Create routing config
    routing_config = RoutingConfig(
        enabled=True,
        models=models,
        enhancement=EnhancementConfig(
            enabled=True,
            strategy=EnhancementStrategy.RULE_BASED,
            min_confidence=0.5,
        ),
        policy=RoutingPolicy(
            strategy=RoutingStrategy.COST_OPTIMIZED,
            enable_fallback=True,
            fallback_model="openai:gpt-4o-mini",
        ),
    )

    # Create agent
    agent = create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    # Test different types of requests
    test_prompts = [
        "What is Python?",  # Simple chat
        "Analyze the code in main.py and suggest improvements",  # Moderate agent
        # Complex agent task
        "Refactor the entire codebase to use async/await patterns and implement comprehensive error handling",
    ]

    for prompt in test_prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")
        print(f"{'=' * 60}")

        result = agent.run_sync(prompt)
        print(f"Result: {result.data}")

    # Show metrics
    if agent.model_router:
        print(f"\n{'=' * 60}")
        print("Routing Metrics:")
        print(f"{'=' * 60}")
        for model, count in agent.model_router.get_metrics().items():
            print(f"  {model}: {count} requests")


def example_dry_run_mode() -> None:
    """Example 3: Dry run mode to test routing without actually switching models."""
    print("\n" + "=" * 80)
    print("Example 3: Dry Run Mode")
    print("=" * 80)

    # Create routing config with dry run enabled
    routing_config = create_default_routing_config()
    routing_config.enabled = True
    routing_config.dry_run = True  # Enable dry run

    # Create agent
    agent = create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    # Run a complex query
    result = agent.run_sync(
        "Implement a complete microservices architecture with service discovery, load balancing, and circuit breakers"
    )

    print(f"\nResult: {result.data}")
    print("\nNote: In dry run mode, routing decisions are logged but not applied.")


def example_enhancement_only() -> None:
    """Example 4: Use only prompt enhancement without routing."""
    print("\n" + "=" * 80)
    print("Example 4: Prompt Enhancement Only")
    print("=" * 80)

    # Create routing config with only enhancement enabled
    routing_config = RoutingConfig(
        enabled=True,
        enhancement=EnhancementConfig(
            enabled=True,
            strategy=EnhancementStrategy.RULE_BASED,
            min_confidence=0.3,
        ),
        models=[],  # No models configured, so no routing
    )

    # Create agent
    agent = create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    # Test with vague prompt
    vague_prompt = "Fix it"
    print(f"\nOriginal prompt: {vague_prompt}")

    result = agent.run_sync(vague_prompt)
    print(f"Result: {result.data}")


def example_classification_only() -> None:
    """Example 5: Use only request classification for analysis."""
    print("\n" + "=" * 80)
    print("Example 5: Request Classification Only")
    print("=" * 80)

    # Create routing config with only classification enabled
    routing_config = RoutingConfig(
        enabled=True,
        enhancement=EnhancementConfig(enabled=False),
        models=[],  # No models configured
    )

    # Create agent
    agent = create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    # Test different prompts
    test_prompts = [
        "What is Python?",
        "Analyze the code in main.py",
        "Refactor the entire codebase with comprehensive error handling and async patterns",
    ]

    for prompt in test_prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")

        if agent.request_classifier:
            classification = agent.request_classifier.classify(prompt)
            print(f"Difficulty: {classification.difficulty.value}")
            print(f"Mode: {classification.mode.value}")
            print(f"Confidence: {classification.confidence:.2f}")
            print(f"Requires Tools: {classification.requires_tools}")


if __name__ == "__main__":
    # Run all examples
    example_basic_routing()
    example_custom_routing()
    example_dry_run_mode()
    example_enhancement_only()
    example_classification_only()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
