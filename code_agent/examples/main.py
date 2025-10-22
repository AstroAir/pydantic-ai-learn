"""
Code Agent Enhanced Examples

Demonstrates all advanced features of the enhanced CodeAgent:
1. Basic code analysis
2. Streaming analysis with real-time feedback
3. Async iteration over execution nodes
4. Usage limits and tracking
5. Retry logic for robust execution
6. Multi-turn conversations with history
7. Quick convenience functions
8. Advanced workflows

Author: The Augster
Python Version: 3.12+
"""

import asyncio
from typing import Any

from code_agent import (
    UsageLimitExceeded,
    UsageLimits,
    create_code_agent,
    quick_analyze,
    quick_refactor,
)

# ============================================================================
# Example 1: Basic Code Analysis (Same as before)
# ============================================================================


def example_basic_analysis() -> None:
    """Example: Analyze code structure and metrics."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Code Analysis")
    print("=" * 60)

    # Create agent
    agent = create_code_agent()

    # Analyze a file
    result = agent.run_sync("Analyze the code structure in tools/task_planning_toolkit.py")

    print("\nAnalysis Result:")
    print(result.output)


# ============================================================================
# Example 2: Streaming Analysis (NEW!)
# ============================================================================


async def example_streaming_analysis() -> None:
    """Example: Stream analysis with real-time feedback."""
    print("\n" + "=" * 60)
    print("Example 2: Streaming Analysis")
    print("=" * 60)

    agent = create_code_agent(enable_streaming=True)

    print("\nStreaming analysis of tools/code_agent_toolkit.py:")
    print("-" * 60)

    # Stream analysis
    async for text in agent.run_stream("Analyze tools/code_agent_toolkit.py and provide a summary"):
        print(text, end="", flush=True)

    print("\n" + "-" * 60)


# ============================================================================
# Example 3: Async Iteration Over Nodes (NEW!)
# ============================================================================


async def example_async_iteration() -> None:
    """Example: Iterate over agent execution nodes."""
    print("\n" + "=" * 60)
    print("Example 3: Async Iteration Over Execution Nodes")
    print("=" * 60)

    agent = create_code_agent()

    print("\nExecution nodes:")
    nodes = []

    async for node in agent.iter_nodes("What is the complexity of tools/filesystem_tools.py?"):
        node_type = type(node).__name__
        nodes.append(node_type)
        print(f"  - {node_type}")

    print(f"\nTotal nodes: {len(nodes)}")


# ============================================================================
# Example 4: Usage Limits and Tracking (NEW!)
# ============================================================================


def example_usage_limits() -> None:
    """Example: Use usage limits and track token usage."""
    print("\n" + "=" * 60)
    print("Example 4: Usage Limits and Tracking")
    print("=" * 60)

    # Create agent with usage limits
    agent = create_code_agent(usage_limits=UsageLimits(response_tokens_limit=100))

    try:
        # This should work (short response)
        result = agent.run_sync("What is the main purpose of tools/bash_tool.py? Answer in one sentence.")
        print("\nShort analysis succeeded:")
        print(result.output)
        print("\nUsage:")
        print(agent.get_usage_summary())

    except UsageLimitExceeded as e:
        print(f"\nUsage limit exceeded: {e}")

    try:
        # This might exceed limits (long response)
        result = agent.run_sync("Provide a comprehensive analysis of tools/task_planning_toolkit.py")
        print("\nLong analysis result:")
        print(result.output)

    except UsageLimitExceeded as e:
        print(f"\nUsage limit exceeded: {e}")
        print("\nCurrent usage:")
        print(agent.get_usage_summary())


# ============================================================================
# Example 5: Retry Logic (NEW!)
# ============================================================================


def example_retry_logic() -> None:
    """Example: Automatic retry on failures."""
    print("\n" + "=" * 60)
    print("Example 5: Retry Logic")
    print("=" * 60)

    agent = create_code_agent()

    # Try to analyze a file (will retry automatically if it fails)
    print("\nAnalyzing with automatic retry...")
    result = agent.run_sync("Analyze tools/code_agent_toolkit.py")

    print("Analysis completed successfully!")
    print(f"Result length: {len(result.output)} characters")


# ============================================================================
# Example 6: Multi-Turn Conversations (NEW!)
# ============================================================================


def example_conversation_history() -> None:
    """Example: Multi-turn conversation with history."""
    print("\n" + "=" * 60)
    print("Example 6: Multi-Turn Conversations")
    print("=" * 60)

    agent = create_code_agent()

    # First turn
    print("\nTurn 1: Initial analysis")
    result1 = agent.run_sync("Analyze tools/filesystem_tools.py and identify the main functions")
    print(result1.output[:200] + "...")

    # Second turn with history
    print("\nTurn 2: Follow-up question")
    result2 = agent.run_sync(
        "What are the top 3 improvements you would suggest?", message_history=result1.new_messages()
    )
    print(result2.output)

    # Third turn with accumulated history
    print("\nTurn 3: Another follow-up")
    result3 = agent.run_sync(
        "Can you explain the first suggestion in more detail?", message_history=result2.all_messages()
    )
    print(result3.output)

    # Show message history
    print(f"\nTotal messages in history: {len(agent.get_message_history())}")


# ============================================================================
# Example 7: Quick Convenience Functions (NEW!)
# ============================================================================


def example_quick_functions() -> None:
    """Example: Use quick convenience functions."""
    print("\n" + "=" * 60)
    print("Example 7: Quick Convenience Functions")
    print("=" * 60)

    # Quick analyze
    print("\nQuick analyze:")
    result = quick_analyze("tools/bash_tool.py")
    print(result[:300] + "...")

    # Quick refactor
    print("\nQuick refactor suggestions:")
    suggestions = quick_refactor("tools/filesystem_tools.py")
    print(suggestions[:300] + "...")


# ============================================================================
# Example 8: Streaming with Event Handler (NEW!)
# ============================================================================


async def example_streaming_with_events() -> None:
    """Example: Stream with custom event handler."""
    print("\n" + "=" * 60)
    print("Example 8: Streaming with Event Handler")
    print("=" * 60)

    agent = create_code_agent()

    events_received = []

    async def event_handler(event: Any) -> None:
        """Custom event handler."""
        event_type = type(event).__name__
        events_received.append(event_type)
        print(f"[Event] {event_type}")

    print("\nStreaming with event tracking:")
    async for text in agent.run_stream("Analyze tools/task_planning_toolkit.py", event_handler=event_handler):
        print(text, end="", flush=True)

    print(f"\n\nTotal events received: {len(events_received)}")
    print(f"Event types: {set(events_received)}")


# ============================================================================
# Example 9: Complete Workflow (NEW!)
# ============================================================================


async def example_complete_workflow() -> None:
    """Example: Complete code improvement workflow."""
    print("\n" + "=" * 60)
    print("Example 9: Complete Code Improvement Workflow")
    print("=" * 60)

    # Create agent with limits
    agent = create_code_agent(usage_limits=UsageLimits(response_tokens_limit=2000))

    file_path = "tools/bash_tool.py"

    # Step 1: Analyze
    print(f"\nStep 1: Analyzing {file_path}...")
    result1 = agent.run_sync(f"Analyze {file_path} and identify issues")
    print(result1.output[:200] + "...")

    # Step 2: Get suggestions (with history)
    print("\nStep 2: Getting refactoring suggestions...")
    result2 = agent.run_sync(
        "Based on the analysis, suggest the top 3 refactoring opportunities", message_history=result1.new_messages()
    )
    print(result2.output[:200] + "...")

    # Step 3: Generate improved code (with history)
    print("\nStep 3: Generating improved code example...")
    result3 = agent.run_sync(
        "Generate an example of how to implement the first suggestion", message_history=result2.all_messages()
    )
    print(result3.output[:200] + "...")

    # Show usage
    print("\n" + "-" * 60)
    print(agent.get_usage_summary())


# ============================================================================
# Example 10: Error Handling and Recovery (NEW!)
# ============================================================================


def example_error_handling() -> None:
    """Example: Handle errors gracefully."""
    print("\n" + "=" * 60)
    print("Example 10: Error Handling and Recovery")
    print("=" * 60)

    agent = create_code_agent()

    # Try to analyze non-existent file
    print("\nAttempting to analyze non-existent file...")
    try:
        result = agent.run_sync("Analyze nonexistent_file.py")
        print(result.output)
    except Exception as e:
        print(f"Error caught: {type(e).__name__}: {e}")
        print("Agent recovered gracefully!")

    # Continue with valid analysis
    print("\nContinuing with valid analysis...")
    result = agent.run_sync("Analyze tools/filesystem_tools.py")
    print(f"Success! Result length: {len(result.output)} characters")


# ============================================================================
# Main Execution
# ============================================================================


async def main_async() -> None:
    """Run async examples."""
    print("\n" + "=" * 60)
    print("CODE AGENT ENHANCED EXAMPLES - ASYNC")
    print("=" * 60)

    try:
        await example_streaming_analysis()
    except Exception as e:
        print(f"Error in streaming analysis: {e}")

    try:
        await example_async_iteration()
    except Exception as e:
        print(f"Error in async iteration: {e}")

    try:
        await example_streaming_with_events()
    except Exception as e:
        print(f"Error in streaming with events: {e}")

    try:
        await example_complete_workflow()
    except Exception as e:
        print(f"Error in complete workflow: {e}")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CODE AGENT ENHANCED EXAMPLES")
    print("=" * 60)

    # Run synchronous examples
    try:
        example_basic_analysis()
    except Exception as e:
        print(f"Error in basic analysis: {e}")

    try:
        example_usage_limits()
    except Exception as e:
        print(f"Error in usage limits: {e}")

    try:
        example_retry_logic()
    except Exception as e:
        print(f"Error in retry logic: {e}")

    try:
        example_conversation_history()
    except Exception as e:
        print(f"Error in conversation history: {e}")

    try:
        example_quick_functions()
    except Exception as e:
        print(f"Error in quick functions: {e}")

    try:
        example_error_handling()
    except Exception as e:
        print(f"Error in error handling: {e}")

    # Run async examples
    print("\n" + "=" * 60)
    print("Running async examples...")
    print("=" * 60)
    asyncio.run(main_async())

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
