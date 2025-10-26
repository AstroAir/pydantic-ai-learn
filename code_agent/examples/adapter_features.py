"""
Adapter Features Examples

Demonstrates adapter features of the code agent.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from code_agent.adapters import (
    ContextConfig,
    ContextManager,
    GraphConfig,
    GraphState,
    ImportanceLevel,
    PruningStrategy,
    WorkflowOrchestrator,
    WorkflowState,
)
from code_agent.config import ConfigManager
from code_agent.core import AgentConfig, create_code_agent
from code_agent.utils import ErrorCategory, RetryStrategy


def example_custom_configuration() -> None:
    """Example: Create agent with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 1: Custom Configuration")
    print("=" * 60)

    # Create custom config
    config = AgentConfig(
        model="openai:gpt-3.5-turbo",
        enable_streaming=True,
        enable_logging=True,
        log_level="DEBUG",
        max_retries=5,
    )

    # Create agent with config
    agent = create_code_agent(
        model=config.model,
        enable_streaming=config.enable_streaming,
    )

    print(f"Agent created with model: {agent.config.model}")
    print(f"Streaming enabled: {agent.config.enable_streaming}")


def example_config_management() -> None:
    """Example: Configuration management."""
    print("\n" + "=" * 60)
    print("Example 2: Configuration Management")
    print("=" * 60)

    manager = ConfigManager()

    # Load from dictionary
    config_dict = {
        "model": "openai:gpt-4",
        "max_retries": 5,
        "timeout": 30.0,
        "features": {
            "streaming": True,
            "logging": True,
        },
    }

    manager.load_from_dict(config_dict)

    # Get values
    model = manager.get("model")
    streaming = manager.get("features.streaming")

    print(f"Model: {model}")
    print(f"Streaming: {streaming}")

    # Set values
    manager.set("max_retries", 10)
    print(f"Updated max_retries: {manager.get('max_retries')}")


def example_context_management() -> None:
    """Example: Context window management."""
    print("\n" + "=" * 60)
    print("Example 3: Context Management")
    print("=" * 60)

    config = ContextConfig(max_tokens=100_000)
    manager = ContextManager(config=config)

    # Add context messages
    manager.add_message(
        content="User query: Analyze this code",
        message_type="user",
        importance=ImportanceLevel.HIGH,
    )

    manager.add_message(
        content="Code to analyze: def foo(): pass",
        message_type="assistant",
        importance=ImportanceLevel.HIGH,
    )

    manager.add_message(
        content="Previous analysis results",
        message_type="system",
        importance=ImportanceLevel.MEDIUM,
    )

    # Get statistics
    stats = manager.get_statistics()
    print(f"Context stats: {stats}")

    # Manual prune context
    manager.manual_prune(strategy=PruningStrategy.IMPORTANCE)
    print(f"After pruning: {manager.get_statistics()}")


def example_workflow_orchestration() -> None:
    """Example: Workflow orchestration."""
    print("\n" + "=" * 60)
    print("Example 4: Workflow Orchestration")
    print("=" * 60)

    workflow = WorkflowOrchestrator(operation_name="code_analysis_workflow")

    # Transition through workflow states
    workflow.transition_to(WorkflowState.RUNNING)
    print(f"Workflow state: {workflow.current_state.value}")

    # Create checkpoint
    checkpoint = workflow.create_checkpoint(
        input_data={"code": "def foo(): pass"}, output_data={"analysis": "complete"}
    )
    print(f"Checkpoint created: {checkpoint.checkpoint_id}")

    # Get workflow status
    status = workflow.get_workflow_status()
    print(f"Workflow status: {status}")


def example_graph_orchestration() -> None:
    """Example: Graph-based orchestration."""
    print("\n" + "=" * 60)
    print("Example 5: Graph Orchestration")
    print("=" * 60)

    # Create graph configuration
    config = GraphConfig(enable_persistence=True, enable_streaming=True, max_iterations=1000, enable_monitoring=True)
    print(f"Graph config created: persistence={config.enable_persistence}")

    # Create graph state
    state = GraphState()
    print(f"Graph state created with {len(state.active_graphs)} active graphs")


def example_retry_strategy() -> None:
    """Example: Retry strategy configuration."""
    print("\n" + "=" * 60)
    print("Example 6: Retry Strategy")
    print("=" * 60)

    # Create retry strategy
    strategy = RetryStrategy(
        max_attempts=5,
        base_delay=1.0,
        exponential_base=2.0,
        max_delay=60.0,
        jitter=True,
    )

    # Calculate delays
    print("Retry delays:")
    for attempt in range(5):
        delay = strategy.calculate_delay(attempt)
        print(f"  Attempt {attempt + 1}: {delay:.2f}s")

    # Check if should retry
    from code_agent.utils import ErrorContext, ErrorSeverity

    context = ErrorContext(
        error_type="ConnectionError",
        error_message="Connection timeout",
        category=ErrorCategory.TRANSIENT,
        severity=ErrorSeverity.MEDIUM,
    )

    should_retry = strategy.should_retry(0, ConnectionError(), context)
    print(f"\nShould retry transient error: {should_retry}")


def example_combined_workflow() -> None:
    """Example: Combined workflow with multiple components."""
    print("\n" + "=" * 60)
    print("Example 7: Combined Workflow")
    print("=" * 60)

    # Create agent
    _agent = create_code_agent()

    # Create context manager
    context_config = ContextConfig(max_tokens=50_000)
    context = ContextManager(config=context_config)

    # Create workflow
    workflow = WorkflowOrchestrator(operation_name="full_analysis")

    # Add context
    context.add_message(
        content="Code to analyze",
        message_type="user",
        importance=ImportanceLevel.HIGH,
    )

    # Transition workflow
    workflow.transition_to(WorkflowState.RUNNING)

    # Create checkpoint
    checkpoint = workflow.create_checkpoint(
        input_data={"code": "def foo(): pass"}, output_data={"analysis": "complete"}
    )

    print("Agent created successfully")
    print(f"Context stats: {context.get_statistics()}")
    print(f"Workflow checkpoint: {checkpoint.checkpoint_id}")


def main() -> None:
    """Run all advanced examples."""
    print("\n" + "=" * 60)
    print("Code Agent - Advanced Features Examples")
    print("=" * 60)

    example_custom_configuration()
    example_config_management()
    example_context_management()
    example_workflow_orchestration()
    example_graph_orchestration()
    example_retry_strategy()
    example_combined_workflow()

    print("\n" + "=" * 60)
    print("Advanced examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
