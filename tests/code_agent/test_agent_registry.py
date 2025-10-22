"""
Tests for Agent Registry

Tests the agent registration and discovery system.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from code_agent.core.agent_registry import AgentRegistry
from code_agent.core.sub_agent import (
    DelegatedTask,
    SubAgent,
    SubAgentResult,
    SubAgentStatus,
)


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


class TestAgentRegistry:
    """Test AgentRegistry."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = AgentRegistry()

        assert len(registry.list_agents()) == 0
        assert registry.heartbeat_timeout == 30

    def test_custom_heartbeat_timeout(self):
        """Test custom heartbeat timeout."""
        registry = AgentRegistry(heartbeat_timeout=60)

        assert registry.heartbeat_timeout == 60

    def test_register_agent(self):
        """Test registering an agent."""
        registry = AgentRegistry()

        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
            capabilities=["analysis"],
        )

        registry.register(agent)

        # Verify registration
        assert len(registry.list_agents()) == 1
        retrieved = registry.get_agent(agent.agent_id)
        assert retrieved is not None
        assert retrieved.name == "Test Agent"

    def test_register_duplicate_agent(self):
        """Test registering duplicate agent raises error."""
        registry = AgentRegistry()

        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
            agent_id="test-123",
        )

        registry.register(agent)

        # Try to register again
        with pytest.raises(ValueError, match="already registered"):
            registry.register(agent)

    def test_deregister_agent(self):
        """Test deregistering an agent."""
        registry = AgentRegistry()

        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
        )

        registry.register(agent)
        assert len(registry.list_agents()) == 1

        # Deregister
        result = registry.deregister(agent.agent_id)
        assert result is True
        assert len(registry.list_agents()) == 0

    def test_deregister_nonexistent_agent(self):
        """Test deregistering non-existent agent."""
        registry = AgentRegistry()

        result = registry.deregister("nonexistent")
        assert result is False

    def test_get_agent_by_name(self):
        """Test getting agent by name."""
        registry = AgentRegistry()

        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
        )

        registry.register(agent)

        retrieved = registry.get_agent_by_name("Test Agent")
        assert retrieved is not None
        assert retrieved.agent_id == agent.agent_id

    def test_get_nonexistent_agent_by_name(self):
        """Test getting non-existent agent by name."""
        registry = AgentRegistry()

        retrieved = registry.get_agent_by_name("Nonexistent")
        assert retrieved is None

    def test_find_by_capability(self):
        """Test finding agents by capability."""
        registry = AgentRegistry()

        agent1 = MockSubAgent(
            name="Agent 1",
            description="Agent 1",
            capabilities=["analysis", "refactoring"],
        )
        agent2 = MockSubAgent(
            name="Agent 2",
            description="Agent 2",
            capabilities=["analysis", "generation"],
        )
        agent3 = MockSubAgent(
            name="Agent 3",
            description="Agent 3",
            capabilities=["testing"],
        )

        registry.register(agent1)
        registry.register(agent2)
        registry.register(agent3)

        # Find agents with "analysis" capability
        analysis_agents = registry.find_by_capability("analysis")
        assert len(analysis_agents) == 2

        # Find agents with "testing" capability
        testing_agents = registry.find_by_capability("testing")
        assert len(testing_agents) == 1

        # Find agents with non-existent capability
        none_agents = registry.find_by_capability("nonexistent")
        assert len(none_agents) == 0

    def test_find_by_capabilities(self):
        """Test finding agents by multiple capabilities."""
        registry = AgentRegistry()

        agent1 = MockSubAgent(
            name="Agent 1",
            description="Agent 1",
            capabilities=["analysis", "refactoring", "testing"],
        )
        agent2 = MockSubAgent(
            name="Agent 2",
            description="Agent 2",
            capabilities=["analysis", "refactoring"],
        )
        agent3 = MockSubAgent(
            name="Agent 3",
            description="Agent 3",
            capabilities=["analysis"],
        )

        registry.register(agent1)
        registry.register(agent2)
        registry.register(agent3)

        # Find agents with both "analysis" and "refactoring"
        agents = registry.find_by_capabilities(["analysis", "refactoring"])
        assert len(agents) == 2

        # Find agents with all three capabilities
        agents = registry.find_by_capabilities(["analysis", "refactoring", "testing"])
        assert len(agents) == 1
        assert agents[0].name == "Agent 1"

    def test_list_agents_by_status(self):
        """Test listing agents by status."""
        registry = AgentRegistry()

        agent1 = MockSubAgent(
            name="Agent 1",
            description="Agent 1",
        )
        agent2 = MockSubAgent(
            name="Agent 2",
            description="Agent 2",
        )

        registry.register(agent1)
        registry.register(agent2)

        # Set different statuses
        agent1.set_status(SubAgentStatus.BUSY)
        agent2.set_status(SubAgentStatus.IDLE)

        # List idle agents
        idle_agents = registry.list_agents(status=SubAgentStatus.IDLE)
        assert len(idle_agents) == 1
        assert idle_agents[0].name == "Agent 2"

        # List busy agents
        busy_agents = registry.list_agents(status=SubAgentStatus.BUSY)
        assert len(busy_agents) == 1
        assert busy_agents[0].name == "Agent 1"

    def test_list_agent_infos(self):
        """Test listing agent infos."""
        registry = AgentRegistry()

        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
        )

        registry.register(agent)

        infos = registry.list_agent_infos()
        assert len(infos) == 1
        assert infos[0].name == "Test Agent"

    def test_check_health(self):
        """Test health check."""
        registry = AgentRegistry(heartbeat_timeout=1)

        agent1 = MockSubAgent(
            name="Agent 1",
            description="Agent 1",
        )
        agent2 = MockSubAgent(
            name="Agent 2",
            description="Agent 2",
        )

        registry.register(agent1)
        registry.register(agent2)

        # Make agent1 appear offline by setting old heartbeat
        agent1.last_heartbeat = datetime.now(UTC) - timedelta(seconds=10)

        health = registry.check_health()

        assert health["total"] == 2
        assert health["healthy"] == 1
        assert health["offline"] == 1
        assert agent1.agent_id in health["offline_agents"]
        assert agent2.agent_id in health["healthy_agents"]

    def test_get_stats(self):
        """Test getting registry statistics."""
        registry = AgentRegistry()

        agent1 = MockSubAgent(
            name="Agent 1",
            description="Agent 1",
            capabilities=["analysis", "refactoring"],
        )
        agent2 = MockSubAgent(
            name="Agent 2",
            description="Agent 2",
            capabilities=["analysis"],
        )

        registry.register(agent1)
        registry.register(agent2)

        stats = registry.get_stats()

        assert stats["total_agents"] == 2
        assert stats["total_capabilities"] == 2
        assert "analysis" in stats["capabilities"]
        assert stats["capabilities"]["analysis"] == 2
        assert stats["capabilities"]["refactoring"] == 1

    def test_clear(self):
        """Test clearing registry."""
        registry = AgentRegistry()

        agent = MockSubAgent(
            name="Test Agent",
            description="A test agent",
        )

        registry.register(agent)
        assert len(registry.list_agents()) == 1

        registry.clear()
        assert len(registry.list_agents()) == 0
