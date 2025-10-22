"""
Context Management Examples

Demonstrates the context management system for intelligent conversation
pruning, summarization, and relevance scoring.

Author: The Augster
Python Version: 3.12+
"""

from code_agent import (
    ContextConfig,
    ContextManager,
    ImportanceLevel,
    LogLevel,
    PruningStrategy,
    create_context_manager,
)


def example_1_basic_context_management() -> None:
    """Example 1: Basic context management with automatic pruning."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Context Management")
    print("=" * 70)

    # Create context manager with default settings
    ctx_mgr = create_context_manager(
        max_tokens=1000,  # Small limit for demo
        model_name="gpt-4",
        strategy=PruningStrategy.RECENCY,
    )

    # Add messages
    for i in range(20):
        ctx_mgr.add_message(
            content=f"This is message {i} with some content to analyze.",
            message_type="user" if i % 2 == 0 else "assistant",
            importance=ImportanceLevel.MEDIUM,
        )

    # Get statistics
    stats = ctx_mgr.get_statistics()
    print("\nContext Statistics:")
    print(f"  Total Tokens: {stats['total_tokens']}")
    print(f"  Segments Count: {stats['segments_count']}")
    print(f"  Pruning Count: {stats['pruning_count']}")
    print(f"  Token Usage: {stats['token_usage_percent']:.1f}%")
    print(f"  Health Status: {ctx_mgr.get_health_status()}")


def example_2_importance_based_pruning() -> None:
    """Example 2: Importance-based pruning preserves critical messages."""
    print("\n" + "=" * 70)
    print("Example 2: Importance-Based Pruning")
    print("=" * 70)

    ctx_mgr = create_context_manager(max_tokens=500, strategy=PruningStrategy.IMPORTANCE)

    # Add messages with different importance levels
    ctx_mgr.add_message("Starting analysis...", message_type="user", importance=ImportanceLevel.LOW)

    ctx_mgr.add_message(
        "Error: File not found - critical issue!", message_type="assistant", importance=ImportanceLevel.CRITICAL
    )

    ctx_mgr.add_message(
        "Here's the code fix:\n```python\ndef fix():\n    pass\n```",
        message_type="assistant",
        importance=ImportanceLevel.HIGH,
    )

    for i in range(10):
        ctx_mgr.add_message(f"Regular message {i}", message_type="user", importance=ImportanceLevel.MEDIUM)

    # Check what was preserved
    stats = ctx_mgr.get_statistics()
    print("\nImportance Distribution:")
    for level, count in stats["importance_distribution"].items():
        print(f"  {level}: {count}")

    print(f"\nMessages with errors: {stats['has_error_count']}")
    print(f"Messages with code: {stats['has_code_count']}")


def example_3_relevance_based_pruning() -> None:
    """Example 3: Relevance-based pruning keeps related messages."""
    print("\n" + "=" * 70)
    print("Example 3: Relevance-Based Pruning")
    print("=" * 70)

    ctx_mgr = create_context_manager(max_tokens=800, strategy=PruningStrategy.RELEVANCE)

    # Add messages about different topics
    topics = [
        ("authentication", "How do I implement user authentication?"),
        ("authentication", "Use JWT tokens for secure authentication."),
        ("database", "What database should I use?"),
        ("database", "PostgreSQL is a good choice for relational data."),
        ("testing", "How do I write unit tests?"),
        ("testing", "Use pytest for Python testing."),
        ("authentication", "Can you show me a JWT example?"),
        ("authentication", "Here's a JWT implementation example..."),
    ]

    for _topic, content in topics:
        ctx_mgr.add_message(content, message_type="user" if "?" in content else "assistant")

    # Add more messages to trigger pruning
    for i in range(10):
        ctx_mgr.add_message(f"Additional message {i} about authentication and JWT", message_type="user")

    stats = ctx_mgr.get_statistics()
    print("\nAfter relevance-based pruning:")
    print(f"  Segments remaining: {stats['segments_count']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Pruning operations: {stats['pruning_count']}")


def example_4_context_checkpointing() -> None:
    """Example 4: Create and restore context checkpoints."""
    print("\n" + "=" * 70)
    print("Example 4: Context Checkpointing")
    print("=" * 70)

    ctx_mgr = create_context_manager(max_tokens=2000)

    # Add initial messages
    for i in range(5):
        ctx_mgr.add_message(f"Initial message {i}", message_type="user")

    # Create checkpoint
    checkpoint = ctx_mgr.create_checkpoint()
    print(f"\nCheckpoint created: {checkpoint.checkpoint_id}")
    print(f"  Segments: {len(checkpoint.segments)}")
    print(f"  Tokens: {checkpoint.total_tokens}")

    # Add more messages
    for i in range(10):
        ctx_mgr.add_message(f"New message {i}", message_type="user")

    print("\nAfter adding more messages:")
    print(f"  Segments: {len(ctx_mgr.get_segments())}")
    print(f"  Tokens: {ctx_mgr.total_tokens}")

    # Restore checkpoint
    ctx_mgr.restore_checkpoint(checkpoint.checkpoint_id)
    print("\nAfter restoring checkpoint:")
    print(f"  Segments: {len(ctx_mgr.get_segments())}")
    print(f"  Tokens: {ctx_mgr.total_tokens}")


def example_5_manual_pruning() -> None:
    """Example 5: Manual pruning with different strategies."""
    print("\n" + "=" * 70)
    print("Example 5: Manual Pruning Strategies")
    print("=" * 70)

    ctx_mgr = create_context_manager(max_tokens=5000)

    # Add many messages
    for i in range(30):
        ctx_mgr.add_message(f"Message {i} with content", message_type="user" if i % 2 == 0 else "assistant")

    print("\nBefore pruning:")
    print(f"  Segments: {len(ctx_mgr.get_segments())}")
    print(f"  Tokens: {ctx_mgr.total_tokens}")

    # Try different pruning strategies
    strategies = [
        PruningStrategy.RECENCY,
        PruningStrategy.SLIDING_WINDOW,
        PruningStrategy.IMPORTANCE,
    ]

    for strategy in strategies:
        # Create fresh context manager
        test_mgr = create_context_manager(max_tokens=5000)
        for i in range(30):
            test_mgr.add_message(f"Message {i}", message_type="user" if i % 2 == 0 else "assistant")

        # Manual prune
        test_mgr.manual_prune(strategy, target_tokens=500)

        print(f"\nAfter {strategy.value} pruning:")
        print(f"  Segments: {len(test_mgr.get_segments())}")
        print(f"  Tokens: {test_mgr.total_tokens}")


def example_6_context_summarization() -> None:
    """Example 6: Context summarization for long conversations."""
    print("\n" + "=" * 70)
    print("Example 6: Context Summarization")
    print("=" * 70)

    config = ContextConfig(
        max_tokens=2000,
        enable_summarization=True,
        summarization_threshold=0.5,  # Trigger at 50%
        pruning_threshold=0.9,
    )

    ctx_mgr = ContextManager(config)

    # Add many messages to trigger summarization
    for i in range(40):
        content = f"Message {i}: "
        if i % 5 == 0:
            content += "Error occurred in processing"
        elif i % 3 == 0:
            content += "Code example: def func(): pass"
        else:
            content += "Regular conversation content"

        ctx_mgr.add_message(content, message_type="user" if i % 2 == 0 else "assistant")

    stats = ctx_mgr.get_statistics()
    print("\nAfter adding messages:")
    print(f"  Total segments: {stats['segments_count']}")
    print(f"  Summarized segments: {stats['summarized_count']}")
    print(f"  Summarization count: {stats['summarization_count']}")
    print(f"  Total tokens: {stats['total_tokens']}")


def example_7_integrated_with_agent() -> None:
    """Example 7: Context management integrated with CodeAgent."""
    print("\n" + "=" * 70)
    print("Example 7: Integrated with CodeAgent")
    print("=" * 70)

    try:
        from code_agent import CodeAgent

        # Create agent with context management enabled
        agent = CodeAgent(
            model="openai:gpt-4", enable_context_management=True, max_context_tokens=50_000, log_level=LogLevel.INFO
        )

        print("\nCodeAgent created with context management")

        # Check context manager
        if agent.state.context_manager:
            print(f"  Max tokens: {agent.state.context_manager.config.max_tokens}")
            print(f"  Strategy: {agent.state.context_manager.config.default_strategy.value}")
            print(f"  Summarization: {agent.state.context_manager.config.enable_summarization}")

            # Get initial statistics
            stats = agent.state.get_context_statistics()
            print("\nInitial context statistics:")
            print(f"  Segments: {stats['segments_count']}")
            print(f"  Tokens: {stats['total_tokens']}")
            print(f"  Health: {agent.state.get_context_health()}")
        else:
            print("  Context management not enabled")

    except Exception as e:
        print(f"\nNote: Full integration requires API key: {e}")
        print("Context management system is ready for use!")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CONTEXT MANAGEMENT EXAMPLES")
    print("=" * 70)

    example_1_basic_context_management()
    example_2_importance_based_pruning()
    example_3_relevance_based_pruning()
    example_4_context_checkpointing()
    example_5_manual_pruning()
    example_6_context_summarization()
    example_7_integrated_with_agent()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
