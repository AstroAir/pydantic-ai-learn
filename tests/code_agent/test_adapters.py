"""
Adapters Tests

Tests for adapter modules.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import pytest

from code_agent.adapters import (
    ContextManager,
    GraphConfig,
    GraphState,
    ImportanceLevel,
    PruningStrategy,
    WorkflowOrchestrator,
    WorkflowState,
)
from code_agent.adapters.context import ContextConfig, create_context_manager


class TestContextManager:
    """Test ContextManager."""

    def test_context_manager_creation(self):
        """Test context manager creation."""
        config = ContextConfig(max_tokens=100_000)
        manager = ContextManager(config=config)

        assert manager.config.max_tokens == 100_000
        assert len(manager.segments) == 0

    def test_add_message(self):
        """Test adding context message."""
        manager = create_context_manager()

        segment = manager.add_message(
            content="test content",
            message_type="user",
            importance=ImportanceLevel.HIGH,
        )

        assert segment.content == "test content"
        assert manager.total_tokens > 0

    def test_get_stats(self):
        """Test getting context stats."""
        manager = create_context_manager()
        manager.add_message("content1", message_type="user")
        manager.add_message("content2", message_type="assistant")

        stats = manager.get_statistics()

        assert stats["segments_count"] == 2
        assert stats["total_tokens"] > 0

    def test_manual_prune(self):
        """Test manual pruning."""
        manager = create_context_manager(max_tokens=1000)
        for i in range(15):
            manager.add_message(f"content{i}", message_type="user")

        manager.manual_prune(strategy=PruningStrategy.RECENCY)

        assert len(manager.segments) <= 15

    def test_context_health_status(self):
        """Test context health status."""
        manager = create_context_manager(max_tokens=1000)
        manager.add_message("content", message_type="user")

        health = manager.get_health_status()

        assert health in ["healthy", "warning", "critical"]


class TestWorkflowOrchestrator:
    """Test WorkflowOrchestrator."""

    def test_workflow_orchestrator_creation(self):
        """Test workflow orchestrator creation."""
        orchestrator = WorkflowOrchestrator(operation_name="test_workflow")

        assert orchestrator.operation_name == "test_workflow"
        assert orchestrator.current_state == WorkflowState.PENDING

    def test_transition_state(self):
        """Test workflow state transition."""
        orchestrator = WorkflowOrchestrator(operation_name="test")

        orchestrator.transition_to(WorkflowState.RUNNING)

        assert orchestrator.current_state == WorkflowState.RUNNING

    def test_create_checkpoint(self):
        """Test creating workflow checkpoint."""
        orchestrator = WorkflowOrchestrator(operation_name="test")

        checkpoint = orchestrator.create_checkpoint(input_data={"test": "data"}, output_data={"result": "success"})

        assert checkpoint.checkpoint_id is not None
        assert len(orchestrator.checkpoints) == 1

    def test_workflow_status(self):
        """Test getting workflow status."""
        orchestrator = WorkflowOrchestrator(operation_name="test")
        orchestrator.transition_to(WorkflowState.RUNNING)

        status = orchestrator.get_workflow_status()

        assert status["operation_name"] == "test"
        assert status["current_state"] == WorkflowState.RUNNING.value


class TestGraphConfig:
    """Test GraphConfig."""

    def test_graph_config_creation(self):
        """Test graph config creation."""
        config = GraphConfig(
            enable_persistence=True, enable_streaming=True, max_iterations=1000, enable_monitoring=True
        )

        assert config.enable_persistence is True
        assert config.enable_streaming is True
        assert config.max_iterations == 1000

    def test_graph_state_creation(self):
        """Test graph state creation."""
        state = GraphState()

        assert len(state.active_graphs) == 0
        assert len(state.execution_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
