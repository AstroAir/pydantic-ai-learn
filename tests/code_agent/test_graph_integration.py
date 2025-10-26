"""
Test Graph Integration

Tests for pydantic_graph integration with code_agent module.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from code_agent import CodeAgent, GraphConfig

# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class GraphTestState:
    """State object used for graph execution in tests (not a pytest test class)."""

    counter: int
    agent: CodeAgent | None = None


@dataclass
class SimpleTestState:
    """Simple test state without agent (for persistence testing)."""

    counter: int


@dataclass
class GraphTestNode(BaseNode[GraphTestState, None, int]):
    """Graph node used for execution tests (not a pytest test class)."""

    async def run(self, ctx: GraphRunContext[GraphTestState]) -> GraphTestNode | End[int]:
        """Execute test node."""
        if ctx.state.counter <= 0:
            return End(ctx.state.counter)
        ctx.state.counter -= 1
        return GraphTestNode()


@dataclass
class SimpleTestNode(BaseNode[SimpleTestState, None, int]):
    """Simple test node for persistence testing."""

    async def run(self, ctx: GraphRunContext[SimpleTestState]) -> SimpleTestNode | End[int]:
        """Execute test node."""
        if ctx.state.counter <= 0:
            return End(ctx.state.counter)
        ctx.state.counter -= 1
        return SimpleTestNode()


# ============================================================================
# Tests
# ============================================================================


def test_graph_integration_enabled() -> None:
    """Test that graph integration can be enabled."""
    print("\n" + "=" * 60)
    print("Test 1: Graph Integration Enabled")
    print("=" * 60)

    agent = CodeAgent(model="test", enable_graph=True)

    assert agent.state.graph_state is not None, "Graph state should be initialized"
    assert agent.state.graph_persistence_adapter is not None, "Persistence adapter should be initialized"

    print("✓ Graph integration enabled successfully")


def test_graph_integration_disabled() -> None:
    """Test that graph integration can be disabled."""
    print("\n" + "=" * 60)
    print("Test 2: Graph Integration Disabled")
    print("=" * 60)

    agent = CodeAgent(model="test", enable_graph=False)

    assert agent.state.graph_state is None, "Graph state should be None when disabled"
    assert agent.state.graph_persistence_adapter is None, "Persistence adapter should be None when disabled"

    print("✓ Graph integration disabled successfully")


def test_graph_config() -> None:
    """Test graph configuration."""
    print("\n" + "=" * 60)
    print("Test 3: Graph Configuration")
    print("=" * 60)

    config = GraphConfig(
        enable_persistence=True,
        checkpoint_dir=Path(".test_checkpoints"),
        enable_streaming=True,
        max_iterations=500,
        enable_monitoring=True,
    )

    agent = CodeAgent(model="test", enable_graph=True, graph_config=config)

    assert agent.state.graph_state is not None
    assert agent.state.graph_state.config.max_iterations == 500
    assert agent.state.graph_state.config.checkpoint_dir == Path(".test_checkpoints")

    print("✓ Graph configuration applied successfully")


@pytest.mark.anyio
async def test_simple_graph_execution() -> None:
    """Test simple graph execution."""
    print("\n" + "=" * 60)
    print("Test 4: Simple Graph Execution")
    print("=" * 60)

    agent = CodeAgent(model="test", enable_graph=True)

    # Create graph
    graph = Graph(nodes=[GraphTestNode])
    state = GraphTestState(counter=3, agent=agent)

    # Execute graph
    result = await graph.run(GraphTestNode(), state=state)

    assert result.output == 0, f"Expected output 0, got {result.output}"

    print(f"✓ Graph executed successfully, output: {result.output}")


@pytest.mark.anyio
async def test_graph_iteration() -> None:
    """Test graph iteration."""
    print("\n" + "=" * 60)
    print("Test 5: Graph Iteration")
    print("=" * 60)

    agent = CodeAgent(model="test", enable_graph=True)

    # Create graph
    graph = Graph(nodes=[GraphTestNode])
    state = GraphTestState(counter=5, agent=agent)

    # Iterate through graph
    nodes = []
    async with graph.iter(GraphTestNode(), state=state) as run:
        async for node in run:
            nodes.append(node)

    # Should have nodes (counter iterations + final End)
    # Counter starts at 5, decrements to 0, then End
    assert len(nodes) >= 6, f"Expected at least 6 nodes, got {len(nodes)}"
    assert isinstance(nodes[-1], End), "Last node should be End"

    print(f"✓ Graph iteration successful, {len(nodes)} nodes executed")


@pytest.mark.anyio
async def test_graph_monitoring() -> None:
    """Test graph execution monitoring."""
    print("\n" + "=" * 60)
    print("Test 6: Graph Monitoring")
    print("=" * 60)

    agent = CodeAgent(model="test", enable_graph=True)

    # Execute multiple graphs
    for i in range(3):
        graph = Graph(nodes=[GraphTestNode])
        state = GraphTestState(counter=2, agent=agent)

        if agent.state.graph_state:
            graph_id = f"test_graph_{i}"
            agent.state.graph_state.register_graph(graph_id, graph)  # type: ignore[arg-type]
            agent.state.graph_state.start_execution(graph_id)

            await graph.run(GraphTestNode(), state=state)

            agent.state.graph_state.complete_execution(graph_id, success=True)

    # Check statistics
    stats = agent.state.get_graph_statistics()
    health = agent.state.get_graph_health()

    assert stats["total_executions"] == 3, f"Expected 3 executions, got {stats['total_executions']}"
    assert stats["successful_executions"] == 3, f"Expected 3 successful, got {stats['successful_executions']}"
    assert health == "healthy", f"Expected healthy status, got {health}"

    print("✓ Graph monitoring working correctly")
    print(f"  Total Executions: {stats['total_executions']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Health: {health}")


@pytest.mark.anyio
async def test_graph_persistence() -> None:
    """Test graph persistence."""
    print("\n" + "=" * 60)
    print("Test 7: Graph Persistence")
    print("=" * 60)

    agent = CodeAgent(
        model="test",
        enable_graph=True,
        graph_config=GraphConfig(enable_persistence=True, checkpoint_dir=Path(".test_graph_checkpoints")),
    )

    # Create persistence handler
    if agent.state.graph_persistence_adapter:
        persistence = agent.state.graph_persistence_adapter.create_file_persistence("test_persistence")

        # Create and initialize graph with simple state (no agent field)
        graph = Graph(nodes=[SimpleTestNode])
        state = SimpleTestState(counter=5)

        await graph.initialize(SimpleTestNode(), state=state, persistence=persistence)

        # Check checkpoint was created
        checkpoints = agent.state.graph_persistence_adapter.list_checkpoints()
        assert "test_persistence" in checkpoints, "Checkpoint should be created"

        print("✓ Graph persistence working correctly")
        print(f"  Checkpoints: {checkpoints}")

        # Cleanup
        agent.state.graph_persistence_adapter.delete_checkpoint("test_persistence")


def test_graph_statistics_methods() -> None:
    """Test graph statistics methods."""
    print("\n" + "=" * 60)
    print("Test 8: Graph Statistics Methods")
    print("=" * 60)

    agent = CodeAgent(model="test", enable_graph=True)

    # Test statistics methods
    stats = agent.state.get_graph_statistics()
    health = agent.state.get_graph_health()

    assert isinstance(stats, dict), "Statistics should be a dictionary"
    assert isinstance(health, str), "Health should be a string"
    assert health == "no_activity", f"Expected no_activity, got {health}"

    print("✓ Graph statistics methods working correctly")
    print(f"  Statistics: {stats}")
    print(f"  Health: {health}")


def test_graph_disabled_statistics() -> None:
    """Test statistics when graph is disabled."""
    print("\n" + "=" * 60)
    print("Test 9: Graph Disabled Statistics")
    print("=" * 60)

    agent = CodeAgent(model="test", enable_graph=False)

    stats = agent.state.get_graph_statistics()
    health = agent.state.get_graph_health()

    assert stats == {"graph_integration": "disabled"}, "Should indicate disabled"
    assert health == "disabled", "Health should be disabled"

    print("✓ Disabled graph statistics working correctly")


# ============================================================================
# Test Runner
# ============================================================================


async def run_async_tests() -> None:
    """Run all async tests."""
    await test_simple_graph_execution()
    await test_graph_iteration()
    await test_graph_monitoring()
    await test_graph_persistence()


def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GRAPH INTEGRATION TESTS")
    print("=" * 60)

    # Run sync tests
    test_graph_integration_enabled()
    test_graph_integration_disabled()
    test_graph_config()
    test_graph_statistics_methods()
    test_graph_disabled_statistics()

    # Run async tests
    asyncio.run(run_async_tests())

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
