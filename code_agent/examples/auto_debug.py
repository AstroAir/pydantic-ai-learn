"""
Code Agent Auto-Debugging Examples

Comprehensive examples demonstrating the new auto-debugging capabilities:
- Automatic error detection and recovery
- Circuit breaker pattern
- Workflow automation
- Structured logging
- Error diagnosis
- Performance metrics

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from code_agent import (
    CodeAgent,
    LogFormat,
    LogLevel,
    WorkflowState,
)

try:
    from pydantic_ai import UsageLimits

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    from typing import Any

    UsageLimits = Any  # type: ignore[assignment,misc]


# ============================================================================
# Example 1: Basic Auto-Debugging with Logging
# ============================================================================


def example_basic_auto_debug() -> None:
    """
    Demonstrate basic auto-debugging with structured logging.

    Shows:
    - Automatic error recovery
    - Structured logging
    - Performance metrics
    """
    print("\n" + "=" * 80)
    print("Example 1: Basic Auto-Debugging with Logging")
    print("=" * 80 + "\n")

    # Create agent with debug logging
    agent = CodeAgent(model="openai:gpt-4", log_level=LogLevel.DEBUG, log_format=LogFormat.HUMAN, enable_workflow=True)

    # Analyze code with automatic error recovery
    try:
        result = agent.run_sync("Analyze the code in code_agent/agent.py and identify the top 3 improvements")
        print(f"Analysis Result:\n{result.output}\n")

        # Show metrics
        metrics = agent.state.logger.get_metrics_summary()
        print("Performance Metrics:")
        print(f"  Total Operations: {metrics['total_operations']}")
        print(f"  Successful: {metrics['successful']}")
        print(f"  Failed: {metrics['failed']}")
        print(f"  Avg Duration: {metrics.get('avg_duration_ms', 0):.2f}ms")

    except Exception as e:
        print(f"Error: {e}")

        # Show error summary
        error_summary = agent.state.get_error_summary()
        print("\nError Summary:")
        print(f"  Total Errors: {error_summary['total_errors']}")
        if error_summary.get("by_category"):
            print(f"  By Category: {error_summary['by_category']}")


# ============================================================================
# Example 2: Circuit Breaker Pattern
# ============================================================================


def example_circuit_breaker() -> None:
    """
    Demonstrate circuit breaker pattern for failing operations.

    Shows:
    - Circuit breaker preventing cascading failures
    - Automatic recovery after timeout
    - Circuit state monitoring
    """
    print("\n" + "=" * 80)
    print("Example 2: Circuit Breaker Pattern")
    print("=" * 80 + "\n")

    agent = CodeAgent(model="openai:gpt-4", log_level=LogLevel.INFO, enable_workflow=True)

    # Get circuit breaker for analysis operations
    cb = agent.state.get_or_create_circuit_breaker("analyze_code", failure_threshold=3, recovery_timeout=30.0)

    print(f"Circuit Breaker State: {cb.get_state()}")

    # Attempt analysis
    try:
        _ = agent.run_sync("Analyze code_agent/toolkit.py")
        print("Analysis successful!")
        print(f"Circuit State: {cb.state.value}")

    except Exception as e:
        print(f"Analysis failed: {e}")
        print(f"Circuit State: {cb.state.value}")
        print(f"Failure Count: {cb.failure_count}")


# ============================================================================
# Example 3: Workflow Automation
# ============================================================================


def example_workflow_automation() -> None:
    """
    Demonstrate workflow automation for multi-step debugging.

    Shows:
    - Workflow state machine
    - State checkpointing
    - Workflow status tracking
    """
    print("\n" + "=" * 80)
    print("Example 3: Workflow Automation")
    print("=" * 80 + "\n")

    agent = CodeAgent(model="openai:gpt-4", log_level=LogLevel.INFO, enable_workflow=True)

    if agent.state.workflow_orchestrator:
        workflow = agent.state.workflow_orchestrator

        # Create checkpoint before analysis
        checkpoint = workflow.create_checkpoint(input_data={"file": "code_agent/agent.py"}, output_data=None)
        print(f"Created checkpoint: {checkpoint.checkpoint_id}")

        # Transition to running state
        workflow.transition_to(WorkflowState.RUNNING)
        print(f"Workflow state: {workflow.current_state.value}")

        # Perform analysis
        try:
            _ = agent.run_sync("Analyze code_agent/agent.py")

            # Transition to completed
            workflow.transition_to(WorkflowState.COMPLETED)

            # Create completion checkpoint
            workflow.create_checkpoint(input_data={"file": "code_agent/agent.py"}, output_data={"result": "success"})

            # Show workflow status
            status = workflow.get_workflow_status()
            print("\nWorkflow Status:")
            print(f"  State: {status['current_state']}")
            print(f"  Checkpoints: {status['checkpoints_count']}")
            print(f"  Errors: {status['errors_encountered']}")

        except Exception as e:
            workflow.transition_to(WorkflowState.FAILED)
            print(f"Workflow failed: {e}")


# ============================================================================
# Example 4: Error Diagnosis and Recovery
# ============================================================================


def example_error_diagnosis() -> None:
    """
    Demonstrate automatic error diagnosis and recovery suggestions.

    Shows:
    - Error categorization
    - Automatic diagnosis
    - Recovery suggestions
    """
    print("\n" + "=" * 80)
    print("Example 4: Error Diagnosis and Recovery")
    print("=" * 80 + "\n")

    agent = CodeAgent(model="openai:gpt-4", log_level=LogLevel.INFO, enable_workflow=True)

    # Attempt to analyze non-existent file (will trigger error)
    try:
        _ = agent.run_sync("Analyze nonexistent_file.py")

    except Exception as e:
        print(f"Error occurred: {e}")

        # Check error history
        if agent.state.error_history:
            latest_error = agent.state.error_history[-1]
            print("\nError Details:")
            print(f"  Type: {latest_error.error_type}")
            print(f"  Category: {latest_error.category.value}")
            print(f"  Severity: {latest_error.severity.value}")
            print("\nRecovery Suggestions:")
            for i, suggestion in enumerate(latest_error.recovery_suggestions, 1):
                print(f"  {i}. {suggestion}")


# ============================================================================
# Example 5: JSON Logging for Machine Parsing
# ============================================================================


def example_json_logging() -> None:
    """
    Demonstrate JSON-formatted logging for machine parsing.

    Shows:
    - JSON log format
    - Structured log data
    - Log sanitization
    """
    print("\n" + "=" * 80)
    print("Example 5: JSON Logging")
    print("=" * 80 + "\n")

    # Create agent with JSON logging
    agent = CodeAgent(model="openai:gpt-4", log_level=LogLevel.INFO, log_format=LogFormat.JSON, enable_workflow=True)

    print("Logging in JSON format (check console for structured logs)")

    try:
        _ = agent.run_sync("Analyze code_agent/logging_config.py")
        print("\nAnalysis completed - check logs above for JSON output")

    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# Example 6: Comprehensive Usage with All Features
# ============================================================================


def example_comprehensive() -> None:
    """
    Demonstrate all auto-debugging features together.

    Shows:
    - Full configuration
    - Error handling
    - Logging
    - Metrics
    - Workflow
    """
    print("\n" + "=" * 80)
    print("Example 6: Comprehensive Auto-Debugging")
    print("=" * 80 + "\n")

    # Create fully configured agent
    usage_limits = None
    if PYDANTIC_AI_AVAILABLE:
        usage_limits = UsageLimits(request_limit=10)

    agent = CodeAgent(
        model="openai:gpt-4",
        usage_limits=usage_limits,
        enable_streaming=False,
        log_level=LogLevel.INFO,
        log_format=LogFormat.HUMAN,
        enable_workflow=True,
    )

    print("Agent Configuration:")
    print(f"  Logging: {agent.state.logger.name}")
    print(f"  Workflow: {'Enabled' if agent.state.workflow_orchestrator else 'Disabled'}")
    print(f"  Circuit Breakers: {len(agent.state.circuit_breakers)}")
    print()

    # Perform analysis with full error handling
    try:
        result = agent.run_sync("Analyze code_agent/error_handling.py and explain the circuit breaker implementation")
        print(f"Analysis Result:\n{result.output}\n")

        # Show comprehensive summary
        print("=" * 80)
        print("Summary")
        print("=" * 80)

        # Usage summary
        print(f"\n{agent.get_usage_summary()}")

        # Metrics summary
        metrics = agent.state.logger.get_metrics_summary()
        print("\nPerformance Metrics:")
        print(f"  Operations: {metrics['total_operations']}")
        print(f"  Success Rate: {metrics['successful']}/{metrics['total_operations']}")
        if metrics.get("avg_duration_ms"):
            print(f"  Avg Duration: {metrics['avg_duration_ms']:.2f}ms")

        # Error summary
        error_summary = agent.state.get_error_summary()
        print("\nError Summary:")
        print(f"  Total Errors: {error_summary['total_errors']}")

        # Circuit breaker status
        print("\nCircuit Breakers:")
        for name, cb in agent.state.circuit_breakers.items():
            state = cb.get_state()
            print(f"  {name}: {state['state']} (failures: {state['failure_count']})")

    except Exception as e:
        print(f"Error: {e}")
        print(f"\nError Summary: {agent.state.get_error_summary()}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all examples."""
    examples = [
        ("Basic Auto-Debugging", example_basic_auto_debug),
        ("Circuit Breaker", example_circuit_breaker),
        ("Workflow Automation", example_workflow_automation),
        ("Error Diagnosis", example_error_diagnosis),
        ("JSON Logging", example_json_logging),
        ("Comprehensive", example_comprehensive),
    ]

    print("\n" + "=" * 80)
    print("CODE AGENT AUTO-DEBUGGING EXAMPLES")
    print("=" * 80)

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nExample '{name}' failed: {e}")

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
