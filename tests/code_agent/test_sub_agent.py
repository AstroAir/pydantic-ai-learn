"""
Tests for Sub-Agent System

Tests the hierarchical sub-agent functionality including registration,
delegation, and result reporting.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from code_agent.core.sub_agent import (
    DelegatedTask,
    SubAgent,
    SubAgentInfo,
    SubAgentResult,
    SubAgentStatus,
    TaskStatus,
)


class TestSubAgentInfo:
    """Test SubAgentInfo dataclass."""

    def test_creation(self):
        """Test creating SubAgentInfo."""
        info = SubAgentInfo(
            agent_id="test-123",
            name="Test Agent",
            description="A test agent",
            capabilities=["analysis", "refactoring"],
        )

        assert info.agent_id == "test-123"
        assert info.name == "Test Agent"
        assert info.description == "A test agent"
        assert "analysis" in info.capabilities
        assert info.status == SubAgentStatus.IDLE

    def test_default_values(self):
        """Test default values."""
        info = SubAgentInfo(
            agent_id="test-123",
            name="Test Agent",
            description="A test agent",
        )

        assert info.capabilities == []
        assert info.status == SubAgentStatus.IDLE
        assert isinstance(info.created_at, datetime)
        assert isinstance(info.last_heartbeat, datetime)


class TestDelegatedTask:
    """Test DelegatedTask dataclass."""

    def test_creation(self):
        """Test creating DelegatedTask."""
        task = DelegatedTask(
            task_id="task-123",
            agent_id="agent-456",
            prompt="Analyze this code",
        )

        assert task.task_id == "task-123"
        assert task.agent_id == "agent-456"
        assert task.prompt == "Analyze this code"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None

    def test_duration_calculation(self):
        """Test task duration calculation."""
        task = DelegatedTask(
            task_id="task-123",
            agent_id="agent-456",
            prompt="Test",
        )

        # No duration when not started
        assert task.duration_seconds() is None

        # Set start and end times
        task.started_at = datetime.now(UTC)
        task.completed_at = task.started_at + timedelta(seconds=5)

        duration = task.duration_seconds()
        assert duration is not None
        assert 4.9 < duration < 5.1  # Allow small variance


class TestSubAgentResult:
    """Test SubAgentResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = SubAgentResult(
            task_id="task-123",
            agent_id="agent-456",
            output="Analysis complete",
            success=True,
        )

        assert result.task_id == "task-123"
        assert result.agent_id == "agent-456"
        assert result.output == "Analysis complete"
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Test failure result."""
        result = SubAgentResult(
            task_id="task-123",
            agent_id="agent-456",
            output=None,
            success=False,
            error="Task failed",
        )

        assert result.success is False
        assert result.error == "Task failed"
        assert result.output is None


class MockSubAgent(SubAgent):
    """Mock sub-agent for testing."""

    async def execute_task(self, task: DelegatedTask) -> SubAgentResult:
        """Execute a task (mock implementation)."""
        return SubAgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            output=f"Processed: {task.prompt}",
            success=True,
        )


class TestSubAgent:
    """Test SubAgent base class."""

    def test_initialization(self):
        """Test sub-agent initialization."""
        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
            capabilities=["analysis"],
        )

        assert agent.name == "Test Agent"
        assert agent.description == "A test agent"
        assert "analysis" in agent.capabilities
        assert agent.status == SubAgentStatus.IDLE
        assert agent.agent_id is not None  # Auto-generated

    def test_custom_agent_id(self):
        """Test custom agent ID."""
        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
            agent_id="custom-123",
        )

        assert agent.agent_id == "custom-123"

    def test_get_info(self):
        """Test getting agent info."""
        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
            capabilities=["analysis", "refactoring"],
        )

        info = agent.get_info()

        assert isinstance(info, SubAgentInfo)
        assert info.name == "Test Agent"
        assert info.description == "A test agent"
        assert len(info.capabilities) == 2

    def test_heartbeat(self):
        """Test heartbeat update."""
        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
        )

        old_heartbeat = agent.last_heartbeat

        # Wait a tiny bit
        import time

        time.sleep(0.01)

        agent.heartbeat()

        assert agent.last_heartbeat > old_heartbeat

    def test_set_status(self):
        """Test setting agent status."""
        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
        )

        assert agent.status == SubAgentStatus.IDLE

        agent.set_status(SubAgentStatus.BUSY)
        assert agent.status == SubAgentStatus.BUSY

        agent.set_status(SubAgentStatus.ERROR)
        assert agent.status == SubAgentStatus.ERROR

    @pytest.mark.anyio
    async def test_execute_task(self):
        """Test task execution."""
        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
        )

        task = DelegatedTask(
            task_id="task-123",
            agent_id=agent.agent_id,
            prompt="Analyze code",
        )

        result = await agent.execute_task(task)

        assert isinstance(result, SubAgentResult)
        assert result.task_id == "task-123"
        assert result.success is True
        assert "Processed" in result.output

    def test_task_storage(self):
        """Test task storage in agent."""
        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
        )

        task = DelegatedTask(
            task_id="task-123",
            agent_id=agent.agent_id,
            prompt="Test",
        )

        # Store task
        agent._tasks[task.task_id] = task

        # Retrieve task
        retrieved = agent.get_task("task-123")
        assert retrieved is not None
        assert retrieved.task_id == "task-123"

        # Non-existent task
        assert agent.get_task("nonexistent") is None

    def test_list_tasks(self):
        """Test listing tasks."""
        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
        )

        # Add tasks with different statuses
        task1 = DelegatedTask(
            task_id="task-1",
            agent_id=agent.agent_id,
            prompt="Task 1",
            status=TaskStatus.PENDING,
        )
        task2 = DelegatedTask(
            task_id="task-2",
            agent_id=agent.agent_id,
            prompt="Task 2",
            status=TaskStatus.COMPLETED,
        )

        agent._tasks["task-1"] = task1
        agent._tasks["task-2"] = task2

        # List all tasks
        all_tasks = agent.list_tasks()
        assert len(all_tasks) == 2

        # List pending tasks
        pending = agent.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].task_id == "task-1"

        # List completed tasks
        completed = agent.list_tasks(status=TaskStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].task_id == "task-2"
