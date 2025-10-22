"""
Code Agent Graph Integration Examples

Demonstrates how to use pydantic_graph workflows with the code_agent module
for complex multi-step operations, state management, and workflow orchestration.

Examples:
1. Simple countdown graph with code_agent integration
2. Code analysis workflow using graphs
3. Multi-step debugging workflow with persistence
4. Graph execution monitoring and recovery

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext

from code_agent import CodeAgent, GraphConfig

# ============================================================================
# Example 1: Simple Countdown Graph
# ============================================================================


@dataclass
class CountDownState:
    """State for countdown graph."""

    counter: int
    agent: CodeAgent | None = None


@dataclass
class CountDown(BaseNode[CountDownState, None, int]):
    """Countdown node that decrements counter."""

    async def run(self, ctx: GraphRunContext[CountDownState]) -> CountDown | End[int]:
        """Execute countdown step."""
        if ctx.state.counter <= 0:
            return End(ctx.state.counter)

        # Log progress using code_agent logger
        if ctx.state.agent and ctx.state.agent.state.logger:
            ctx.state.agent.state.logger.info(f"Countdown: {ctx.state.counter}", counter=ctx.state.counter)

        ctx.state.counter -= 1
        return CountDown()


async def example_countdown() -> None:
    """
    Example 1: Simple countdown graph integrated with code_agent.

    Demonstrates:
    - Creating a simple graph
    - Integrating with code_agent logging
    - Iterating through graph execution
    """
    print("\n" + "=" * 60)
    print("Example 1: Simple Countdown Graph")
    print("=" * 60)

    # Create code agent with graph support
    agent = CodeAgent(enable_graph=True)

    # Create countdown graph
    count_down_graph = Graph(nodes=[CountDown])

    # Create state with agent reference
    state = CountDownState(counter=5, agent=agent)

    # Execute graph
    print("\nExecuting countdown graph...")
    async with count_down_graph.iter(CountDown(), state=state) as run:
        async for node in run:
            print(f"  Node: {node}")

    if run.result is not None:
        print(f"\nFinal output: {run.result.output}")

    # Show graph statistics
    if agent.state.graph_state:
        stats = agent.state.graph_state.get_statistics()
        print(f"\nGraph Statistics: {stats}")


# ============================================================================
# Example 2: Code Analysis Workflow Graph
# ============================================================================


@dataclass
class AnalysisState:
    """State for code analysis workflow."""

    file_path: str
    analysis_result: str | None = None
    patterns_found: str | None = None
    metrics: str | None = None
    agent: CodeAgent | None = None


@dataclass
class AnalyzeStructure(BaseNode[AnalysisState]):
    """Analyze code structure."""

    async def run(
        self, ctx: GraphRunContext[AnalysisState]
    ) -> Annotated[DetectPatterns, Edge(label="Structure analyzed")]:
        """Analyze code structure."""
        if ctx.state.agent:
            result = ctx.state.agent.run_sync(f"Analyze the structure of {ctx.state.file_path}")
            ctx.state.analysis_result = str(result.output)

        return DetectPatterns()


@dataclass
class DetectPatterns(BaseNode[AnalysisState]):
    """Detect code patterns."""

    async def run(
        self, ctx: GraphRunContext[AnalysisState]
    ) -> Annotated[CalculateMetrics, Edge(label="Patterns detected")]:
        """Detect code patterns."""
        if ctx.state.agent:
            result = ctx.state.agent.run_sync(f"Detect patterns in {ctx.state.file_path}")
            ctx.state.patterns_found = str(result.output)

        return CalculateMetrics()


@dataclass
class CalculateMetrics(BaseNode[AnalysisState, None, dict[str, str | None]]):
    """Calculate code metrics."""

    async def run(
        self, ctx: GraphRunContext[AnalysisState]
    ) -> Annotated[End[dict[str, str | None]], Edge(label="Metrics calculated")]:
        """Calculate metrics."""
        if ctx.state.agent:
            result = ctx.state.agent.run_sync(f"Calculate metrics for {ctx.state.file_path}")
            ctx.state.metrics = str(result.output)

        return End(
            {
                "file_path": ctx.state.file_path,
                "analysis": ctx.state.analysis_result,
                "patterns": ctx.state.patterns_found,
                "metrics": ctx.state.metrics,
            }
        )


async def example_analysis_workflow() -> None:
    """
    Example 2: Code analysis workflow using graphs.

    Demonstrates:
    - Multi-step analysis workflow
    - Integration with code_agent tools
    - Graph edges with labels
    - Collecting results from graph execution
    """
    print("\n" + "=" * 60)
    print("Example 2: Code Analysis Workflow Graph")
    print("=" * 60)

    # Create code agent
    agent = CodeAgent(enable_graph=True)

    # Create analysis graph
    analysis_graph = Graph(nodes=(AnalyzeStructure, DetectPatterns, CalculateMetrics), state_type=AnalysisState)

    # Create state
    state = AnalysisState(file_path="code_agent/agent.py", agent=agent)

    # Execute graph
    print("\nExecuting analysis workflow...")
    result = await analysis_graph.run(AnalyzeStructure(), state=state)

    print("\nWorkflow completed!")
    print(f"Result: {result.output}")


# ============================================================================
# Example 3: Graph with Persistence
# ============================================================================


async def example_graph_persistence() -> None:
    """
    Example 3: Graph execution with persistence.

    Demonstrates:
    - Graph state persistence
    - Checkpoint creation and recovery
    - Resumable workflows
    """
    print("\n" + "=" * 60)
    print("Example 3: Graph with Persistence")
    print("=" * 60)

    # Create code agent with graph support
    agent = CodeAgent(
        enable_graph=True, graph_config=GraphConfig(enable_persistence=True, checkpoint_dir=Path(".graph_checkpoints"))
    )

    # Create graph
    count_down_graph = Graph(nodes=[CountDown])

    # Get persistence adapter
    if agent.state.graph_persistence_adapter:
        persistence = agent.state.graph_persistence_adapter.create_file_persistence("countdown_example")

        # Initialize graph with persistence
        state = CountDownState(counter=10, agent=agent)
        await count_down_graph.initialize(CountDown(), state=state, persistence=persistence)

        print("\nGraph initialized with persistence")
        print("Checkpoint saved to: .graph_checkpoints/countdown_example.json")

        # List checkpoints
        checkpoints = agent.state.graph_persistence_adapter.list_checkpoints()
        print(f"\nAvailable checkpoints: {checkpoints}")


# ============================================================================
# Example 4: Graph Monitoring
# ============================================================================


async def example_graph_monitoring() -> None:
    """
    Example 4: Graph execution monitoring.

    Demonstrates:
    - Graph execution metrics
    - Health monitoring
    - Statistics collection
    """
    print("\n" + "=" * 60)
    print("Example 4: Graph Monitoring")
    print("=" * 60)

    # Create code agent
    agent = CodeAgent(enable_graph=True)

    # Execute multiple graphs
    for i in range(3):
        count_down_graph = Graph(nodes=[CountDown])
        state = CountDownState(counter=3, agent=agent)

        if agent.state.graph_state:
            # Register graph
            graph_id = f"countdown_{i}"
            agent.state.graph_state.register_graph(graph_id, count_down_graph)  # type: ignore[arg-type]

            # Start tracking
            agent.state.graph_state.start_execution(graph_id)

            # Execute
            await count_down_graph.run(CountDown(), state=state)

            # Complete tracking
            agent.state.graph_state.complete_execution(graph_id, success=True)

    # Show statistics
    if agent.state.graph_state:
        stats = agent.state.get_graph_statistics()
        health = agent.state.get_graph_health()

        print("\nGraph Statistics:")
        print(f"  Total Executions: {stats['total_executions']}")
        print(f"  Successful: {stats['successful_executions']}")
        print(f"  Failed: {stats['failed_executions']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Health Status: {health}")


# ============================================================================
# Main Runner
# ============================================================================


async def main() -> None:
    """Run all graph examples."""

    print("\n" + "=" * 60)
    print("CODE AGENT GRAPH INTEGRATION EXAMPLES")
    print("=" * 60)

    # Run examples
    await example_countdown()
    # await example_analysis_workflow()  # Commented out as it requires actual file analysis
    await example_graph_persistence()
    await example_graph_monitoring()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
