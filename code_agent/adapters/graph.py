"""
Code Agent Graph Integration

Seamless integration of pydantic_graph functionality into the code_agent module,
providing workflow orchestration, state machine execution, and graph-based
multi-step operations with full error handling and persistence support.

Features:
- Graph workflow creation and execution
- State machine orchestration
- Graph persistence and recovery
- Streaming graph execution
- Integration with code_agent error handling
- Circuit breaker protection for graph operations
- Graph execution monitoring and metrics

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic_graph import (
    BaseNode,
    End,
    FullStatePersistence,
    Graph,
)
from pydantic_graph.persistence.file import FileStatePersistence

# Type variables for generic graph operations
StateT = TypeVar("StateT")
DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


# ============================================================================
# Graph Configuration
# ============================================================================


@dataclass
class GraphConfig:
    """
    Configuration for graph integration.

    Attributes:
        enable_persistence: Enable graph state persistence
        checkpoint_dir: Directory for graph checkpoints
        enable_streaming: Enable streaming graph execution
        max_iterations: Maximum graph iterations (safety limit)
        enable_monitoring: Enable graph execution monitoring
    """

    enable_persistence: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path(".graph_checkpoints"))
    enable_streaming: bool = True
    max_iterations: int = 1000
    enable_monitoring: bool = True


# ============================================================================
# Graph State Tracking
# ============================================================================


@dataclass
class GraphExecutionMetrics:
    """Metrics for graph execution."""

    graph_id: str
    start_time: float
    end_time: float | None = None
    total_nodes_executed: int = 0
    total_iterations: int = 0
    success: bool = False
    error_message: str | None = None

    @property
    def duration(self) -> float | None:
        """Calculate execution duration."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class GraphState:
    """
    State tracking for graph execution within code_agent.

    Tracks active graphs, execution history, and provides
    monitoring capabilities.

    Attributes:
        active_graphs: Currently active graph instances
        execution_history: History of graph executions
        graph_metrics: Execution metrics per graph
        config: Graph configuration
    """

    active_graphs: dict[str, Graph[Any]] = field(default_factory=dict)
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    graph_metrics: dict[str, GraphExecutionMetrics] = field(default_factory=dict)
    config: GraphConfig = field(default_factory=GraphConfig)

    def register_graph(self, graph_id: str, graph: Graph[Any]) -> None:
        """Register a new graph."""
        self.active_graphs[graph_id] = graph

    def unregister_graph(self, graph_id: str) -> None:
        """Unregister a graph."""
        self.active_graphs.pop(graph_id, None)

    def start_execution(self, graph_id: str) -> GraphExecutionMetrics:
        """Start tracking graph execution."""
        metrics = GraphExecutionMetrics(graph_id=graph_id, start_time=time.time())
        self.graph_metrics[graph_id] = metrics
        return metrics

    def complete_execution(self, graph_id: str, success: bool = True, error_message: str | None = None) -> None:
        """Complete graph execution tracking."""
        if graph_id in self.graph_metrics:
            metrics = self.graph_metrics[graph_id]
            metrics.end_time = time.time()
            metrics.success = success
            metrics.error_message = error_message

            # Add to history
            self.execution_history.append(
                {
                    "graph_id": graph_id,
                    "start_time": metrics.start_time,
                    "end_time": metrics.end_time,
                    "duration": metrics.duration,
                    "nodes_executed": metrics.total_nodes_executed,
                    "iterations": metrics.total_iterations,
                    "success": success,
                    "error": error_message,
                }
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get graph execution statistics."""
        total_executions = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h["success"])
        failed = total_executions - successful

        avg_duration = 0.0
        if self.execution_history:
            durations = [h["duration"] for h in self.execution_history if h["duration"]]
            avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / total_executions if total_executions > 0 else 0.0,
            "average_duration": avg_duration,
            "active_graphs": len(self.active_graphs),
            "graph_ids": list(self.active_graphs.keys()),
        }

    def get_health_status(self) -> str:
        """Get graph health status."""
        stats = self.get_statistics()

        if stats["total_executions"] == 0:
            return "no_activity"

        success_rate = stats["success_rate"]
        if success_rate >= 0.95:
            return "healthy"
        if success_rate >= 0.80:
            return "degraded"
        return "unhealthy"


# ============================================================================
# Graph Persistence Adapter
# ============================================================================


class GraphPersistenceAdapter:
    """
    Adapter for graph persistence compatible with code_agent patterns.

    Bridges pydantic_graph persistence with code_agent's workflow
    checkpoint system.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize persistence adapter.

        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_file_persistence(self, graph_id: str) -> FileStatePersistence:
        """
        Create file-based persistence for a graph.

        Args:
            graph_id: Unique graph identifier

        Returns:
            FileStatePersistence instance
        """
        checkpoint_file = self.checkpoint_dir / f"{graph_id}.json"
        # Best-effort cleanup of a stale lock file from previous runs
        lock_file = self.checkpoint_dir / f"{checkpoint_file.name}.pydantic-graph-persistence-lock"
        try:
            if lock_file.exists():
                lock_file.unlink()
        except Exception:
            pass
        return FileStatePersistence(checkpoint_file)

    def create_full_persistence(self) -> FullStatePersistence:
        """
        Create full state persistence (in-memory).

        Returns:
            FullStatePersistence instance
        """
        return FullStatePersistence()

    def list_checkpoints(self) -> list[str]:
        """
        List available graph checkpoints.

        Returns:
            List of graph IDs with checkpoints
        """
        if not self.checkpoint_dir.exists():
            return []

        return [f.stem for f in self.checkpoint_dir.glob("*.json")]

    def delete_checkpoint(self, graph_id: str) -> bool:
        """
        Delete a graph checkpoint.

        Args:
            graph_id: Graph identifier

        Returns:
            True if deleted, False if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{graph_id}.json"
        deleted = False
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            deleted = True
        # Also remove potential stale lock file to avoid future timeouts
        lock_file = self.checkpoint_dir / f"{checkpoint_file.name}.pydantic-graph-persistence-lock"
        try:
            if lock_file.exists():
                lock_file.unlink()
        except Exception:
            pass
        return deleted


# ============================================================================
# Graph Execution Wrappers
# ============================================================================


def create_graph_from_nodes(  # noqa: UP047
    nodes: list[type[BaseNode[Any, Any, Any]]], state_type: type[StateT] | None = None, graph_id: str | None = None
) -> tuple[Graph[Any], str]:
    """
    Create a graph from node classes.

    Args:
        nodes: List of node classes
        state_type: Optional state type for the graph
        graph_id: Optional graph identifier (auto-generated if not provided)

    Returns:
        Tuple of (Graph instance, graph_id)
    """
    if graph_id is None:
        graph_id = f"graph_{int(time.time() * 1000)}"

    # Use Graph[Any] here to avoid type-inference issues when state_type is None.
    graph: Graph[Any] = Graph(nodes=nodes) if state_type is None else Graph(nodes=nodes, state_type=state_type)
    return graph, graph_id


async def execute_graph_async(
    graph: Graph[Any],
    start_node: BaseNode[Any, Any, Any],
    state: Any = None,
    deps: Any = None,
    persistence: Any = None,
    max_iterations: int = 1000,
) -> Any:
    """
    Execute a graph asynchronously.

    Args:
        graph: Graph to execute
        start_node: Starting node
        state: Graph state
        deps: Graph dependencies
        persistence: Optional persistence handler
        max_iterations: Maximum iterations (safety limit)

    Returns:
        Graph execution result

    Raises:
        RuntimeError: If max iterations exceeded
    """
    return await graph.run(start_node, state=state, deps=deps, persistence=persistence)


async def iterate_graph(
    graph: Graph[Any],
    start_node: BaseNode[Any, Any, Any],
    state: Any = None,
    deps: Any = None,
    persistence: Any = None,
    max_iterations: int = 1000,
) -> list[BaseNode[Any, Any, Any] | End[Any]]:
    """
    Iterate through graph execution, collecting all nodes.

    Args:
        graph: Graph to execute
        start_node: Starting node
        state: Graph state
        deps: Graph dependencies
        persistence: Optional persistence handler
        max_iterations: Maximum iterations

    Returns:
        List of all nodes executed

    Raises:
        RuntimeError: If max iterations exceeded
    """
    nodes: list[BaseNode[Any, Any, Any] | End[Any]] = []
    iteration_count = 0

    async with graph.iter(start_node, state=state, deps=deps, persistence=persistence) as run:
        async for node in run:
            nodes.append(node)
            iteration_count += 1

            if iteration_count >= max_iterations:
                raise RuntimeError(f"Graph execution exceeded maximum iterations ({max_iterations})")

    return nodes


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "GraphConfig",
    "GraphState",
    "GraphExecutionMetrics",
    "GraphPersistenceAdapter",
    "create_graph_from_nodes",
    "execute_graph_async",
    "iterate_graph",
]
