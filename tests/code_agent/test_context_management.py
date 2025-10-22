"""
Unit Tests for Context Management System

Tests for context pruning, summarization, relevance scoring, and orchestration.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import sys

from code_agent.adapters.context import (
    TIKTOKEN_AVAILABLE,
    ContextConfig,
    ContextManager,
    ImportanceLevel,
    PruningStrategy,
    RecencyBasedPruner,
    RelevanceScorer,
    TokenCounter,
    create_context_manager,
)


def test_token_counter() -> None:
    """Test token counting functionality."""
    print("\n" + "=" * 70)
    print("Test: Token Counter")
    print("=" * 70)

    counter = TokenCounter(model_name="gpt-4")

    # Test simple text
    text = "Hello, world!"
    tokens = counter.count_tokens(text)
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    assert tokens > 0, "Token count should be positive"

    # Test message counting
    message = {"role": "user", "content": "This is a test message"}
    msg_tokens = counter.count_message_tokens(message)
    print(f"\nMessage tokens: {msg_tokens}")
    assert msg_tokens > 0, "Message token count should be positive"

    print("✓ Token counter tests passed")


def test_relevance_scorer() -> None:
    """Test relevance scoring functionality."""
    print("\n" + "=" * 70)
    print("Test: Relevance Scorer")
    print("=" * 70)

    scorer = RelevanceScorer(cache_size=100)

    # Test keyword extraction
    text = "Python programming language with machine learning capabilities"
    keywords = scorer.extract_keywords(text, top_n=5)
    print(f"Text: '{text}'")
    print(f"Keywords: {keywords}")
    assert len(keywords) > 0, "Should extract keywords"
    assert "python" in keywords or "programming" in keywords, "Should extract relevant keywords"

    # Test similarity calculation
    text1 = "Python programming and machine learning"
    text2 = "Machine learning with Python code"
    similarity = scorer.calculate_similarity(text1, text2)
    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Similarity: {similarity:.3f}")
    assert 0 <= similarity <= 1, "Similarity should be between 0 and 1"
    assert similarity > 0, "Related texts should have positive similarity"

    # Test cache
    similarity2 = scorer.calculate_similarity(text1, text2)
    assert similarity == similarity2, "Cached result should match"

    print("✓ Relevance scorer tests passed")


def test_recency_based_pruner() -> None:
    """Test recency-based pruning."""
    print("\n" + "=" * 70)
    print("Test: Recency-Based Pruner")
    print("=" * 70)

    config = ContextConfig(max_tokens=1000)
    counter = TokenCounter()
    pruner = RecencyBasedPruner(config, counter)

    # Create context manager and add messages with longer content
    ctx_mgr = create_context_manager(max_tokens=500, strategy=PruningStrategy.RECENCY)

    for i in range(30):
        # Add longer messages to ensure token count is significant
        ctx_mgr.add_message(
            f"Message {i} with some longer content to ensure we have enough tokens for pruning to work properly",
            message_type="user" if i % 2 == 0 else "assistant",
        )

    segments = ctx_mgr.get_segments()
    total_tokens = sum(seg.metadata.token_count for seg in segments)
    print(f"Total segments before pruning: {len(segments)}")
    print(f"Total tokens: {total_tokens}")

    # Prune to target (much smaller)
    target = 100
    pruned = pruner.prune(segments, target_tokens=target)
    pruned_tokens = sum(seg.metadata.token_count for seg in pruned)
    print(f"Segments after pruning: {len(pruned)}")
    print(f"Tokens after pruning: {pruned_tokens}")

    assert len(pruned) <= len(segments), "Should not add segments"
    assert len(pruned) > 0, "Should keep some segments"

    # Check that most recent messages are kept
    if len(pruned) > 0:
        last_segment = pruned[-1]
        print(f"Last segment content: {str(last_segment.content)[:50]}...")
        # Should keep recent messages (high numbers)
        has_recent = any(str(i) in str(last_segment.content) for i in range(25, 30))
        assert has_recent, "Should keep most recent messages"

    print("✓ Recency-based pruner tests passed")


def test_importance_based_pruner() -> None:
    """Test importance-based pruning."""
    print("\n" + "=" * 70)
    print("Test: Importance-Based Pruner")
    print("=" * 70)

    ctx_mgr = create_context_manager(max_tokens=1000, strategy=PruningStrategy.IMPORTANCE)

    # Add messages with different importance
    ctx_mgr.add_message("Low priority message", importance=ImportanceLevel.LOW)
    ctx_mgr.add_message("Medium priority message", importance=ImportanceLevel.MEDIUM)
    ctx_mgr.add_message("High priority message", importance=ImportanceLevel.HIGH)
    ctx_mgr.add_message("Critical error occurred!", importance=ImportanceLevel.CRITICAL)

    # Add many low priority messages
    for i in range(15):
        ctx_mgr.add_message(f"Filler message {i}", importance=ImportanceLevel.LOW)

    # Trigger pruning
    ctx_mgr.manual_prune(PruningStrategy.IMPORTANCE, target_tokens=200)

    segments = ctx_mgr.get_segments()
    print(f"Segments after pruning: {len(segments)}")

    # Check that critical message is preserved
    has_critical = any("Critical" in str(seg.content) for seg in segments)
    print(f"Critical message preserved: {has_critical}")
    assert has_critical, "Critical messages should be preserved"

    print("✓ Importance-based pruner tests passed")


def test_sliding_window_pruner() -> None:
    """Test sliding window pruning."""
    print("\n" + "=" * 70)
    print("Test: Sliding Window Pruner")
    print("=" * 70)

    ctx_mgr = create_context_manager(max_tokens=2000, strategy=PruningStrategy.SLIDING_WINDOW)

    # Add messages with longer content
    for i in range(40):
        ctx_mgr.add_message(
            f"Message {i} with longer content to ensure sufficient tokens for testing pruning behavior",
            message_type="user",
        )

    initial_count = len(ctx_mgr.get_segments())
    initial_tokens = ctx_mgr.total_tokens
    print(f"Initial segments: {initial_count}")
    print(f"Initial tokens: {initial_tokens}")

    # Trigger pruning with small target
    target = 200
    ctx_mgr.manual_prune(PruningStrategy.SLIDING_WINDOW, target_tokens=target)

    final_count = len(ctx_mgr.get_segments())
    final_tokens = ctx_mgr.total_tokens
    print(f"Final segments: {final_count}")
    print(f"Final tokens: {final_tokens}")

    assert final_count <= initial_count, "Should not add segments"
    assert final_count > 0, "Should keep some segments"
    assert final_tokens <= initial_tokens, "Should not increase tokens"

    print("✓ Sliding window pruner tests passed")


def test_context_manager_auto_pruning() -> None:
    """Test automatic pruning triggers."""
    print("\n" + "=" * 70)
    print("Test: Context Manager Auto-Pruning")
    print("=" * 70)

    config = ContextConfig(
        max_tokens=500,
        pruning_threshold=0.8,  # Trigger at 80%
        min_tokens_to_keep=200,
        enable_summarization=False,  # Disable summarization to test pruning
    )

    ctx_mgr = ContextManager(config)

    # Add messages with longer content until pruning triggers
    for i in range(60):
        ctx_mgr.add_message(
            f"Message {i} with longer content to fill tokens and trigger automatic pruning behavior",
            message_type="user",
        )

    stats = ctx_mgr.get_statistics()
    print(f"Final token count: {stats['total_tokens']}")
    print(f"Pruning count: {stats['pruning_count']}")
    print(f"Summarization count: {stats['summarization_count']}")
    print(f"Token usage: {stats['token_usage_percent']:.1f}%")

    # Should have triggered either pruning or summarization
    assert stats["pruning_count"] > 0 or stats["summarization_count"] > 0, (
        "Should have triggered pruning or summarization"
    )
    assert stats["total_tokens"] <= config.max_tokens, "Should not exceed max tokens"

    print("✓ Auto-pruning tests passed")


def test_context_checkpointing() -> None:
    """Test context checkpointing and restoration."""
    print("\n" + "=" * 70)
    print("Test: Context Checkpointing")
    print("=" * 70)

    ctx_mgr = create_context_manager(max_tokens=2000)

    # Add initial messages
    for i in range(5):
        ctx_mgr.add_message(f"Initial message {i}")

    initial_count = len(ctx_mgr.get_segments())
    print(f"Initial segments: {initial_count}")

    # Create checkpoint
    checkpoint = ctx_mgr.create_checkpoint()
    print(f"Checkpoint created: {checkpoint.checkpoint_id}")

    # Add more messages
    for i in range(10):
        ctx_mgr.add_message(f"New message {i}")

    modified_count = len(ctx_mgr.get_segments())
    print(f"Segments after modification: {modified_count}")
    assert modified_count > initial_count, "Should have more segments"

    # Restore checkpoint
    restored = ctx_mgr.restore_checkpoint(checkpoint.checkpoint_id)
    assert restored, "Should successfully restore checkpoint"

    restored_count = len(ctx_mgr.get_segments())
    print(f"Segments after restoration: {restored_count}")
    assert restored_count == initial_count, "Should restore to original count"

    print("✓ Checkpointing tests passed")


def test_context_statistics() -> None:
    """Test context statistics and health metrics."""
    print("\n" + "=" * 70)
    print("Test: Context Statistics")
    print("=" * 70)

    ctx_mgr = create_context_manager(max_tokens=1000)

    # Add various messages
    ctx_mgr.add_message("Regular message", importance=ImportanceLevel.MEDIUM)
    ctx_mgr.add_message("Error occurred!", importance=ImportanceLevel.CRITICAL)
    ctx_mgr.add_message("Code: def func(): pass", importance=ImportanceLevel.HIGH)

    stats = ctx_mgr.get_statistics()

    print("Statistics:")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Segments: {stats['segments_count']}")
    print(f"  Has code: {stats['has_code_count']}")
    print(f"  Has error: {stats['has_error_count']}")
    print(f"  Importance distribution: {stats['importance_distribution']}")

    assert stats["segments_count"] == 3, "Should have 3 segments"
    assert stats["has_error_count"] > 0, "Should detect error"
    assert stats["has_code_count"] > 0, "Should detect code"

    # Test health status
    health = ctx_mgr.get_health_status()
    print(f"  Health status: {health}")
    assert health in ["healthy", "warning", "critical"], "Should return valid health status"

    print("✓ Statistics tests passed")


def main() -> int:
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CONTEXT MANAGEMENT UNIT TESTS")
    print("=" * 70)
    print(f"Tiktoken available: {TIKTOKEN_AVAILABLE}")

    try:
        test_token_counter()
        test_relevance_scorer()
        test_recency_based_pruner()
        test_importance_based_pruner()
        test_sliding_window_pruner()
        test_context_manager_auto_pruning()
        test_context_checkpointing()
        test_context_statistics()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
