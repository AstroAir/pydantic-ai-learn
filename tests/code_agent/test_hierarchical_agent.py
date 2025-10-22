"""
Tests for Hierarchical Agent System

Tests the hierarchical agent functionality including delegation and
integration with CodeAgent.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from code_agent.core.hierarchical_agent import (
    CodeSubAgent,
    HierarchicalAgent,
    HierarchicalAgentConfig,
)
from code_agent.core.sub_agent import (
    DelegatedTask,
    SubAgentResult,
)


class MockCodeAgent:
    """Mock CodeAgent for testing."""

    def __init__(self):
        """Initialize mock agent."""
        self.agent = MagicMock()  # PydanticAI agent
        self._last_prompt = None

    async def run(self, prompt: str) -> Any:
        """Mock run method."""
        self._last_prompt = prompt
        result = MagicMock()
        result.output = f"Mock response to: {prompt}"
        return result


class TestCodeSubAgent:
    """Test CodeSubAgent."""

    @pytest.mark.anyio
    async def test_execute_task(self):
        """Test task execution with CodeAgent."""
        mock_agent = MockCodeAgent()

        sub_agent = CodeSubAgent(
            name="Test Sub-Agent",
            description="A test sub-agent",
            agent=mock_agent,
            capabilities=["analysis"],
        )

        task = DelegatedTask(
            task_id="task-123",
            agent_id=sub_agent.agent_id,
            prompt="Analyze this code",
        )

        result = await sub_agent.execute_task(task)

        assert isinstance(result, SubAgentResult)
        assert result.task_id == "task-123"
        assert result.success is True
        assert "Mock response" in result.output

    @pytest.mark.anyio
    async def test_execute_task_with_error(self):
        """Test task execution with error."""
        mock_agent = MockCodeAgent()

        # Make the agent raise an error
        async def failing_run(prompt):
            raise ValueError("Test error")

        mock_agent.run = failing_run

        sub_agent = CodeSubAgent(
            name="Test Sub-Agent",
            description="A test sub-agent",
            agent=mock_agent,
        )

        task = DelegatedTask(
            task_id="task-123",
            agent_id=sub_agent.agent_id,
            prompt="Analyze this code",
        )

        result = await sub_agent.execute_task(task)

        assert result.success is False
        assert result.error is not None
        assert "Test error" in result.error


class TestHierarchicalAgentConfig:
    """Test HierarchicalAgentConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = HierarchicalAgentConfig()

        assert config.enable_sub_agents is True
        assert config.enable_a2a is False
        assert config.max_delegation_depth == 3
        assert config.auto_register_sub_agents is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = HierarchicalAgentConfig(
            enable_sub_agents=False,
            enable_a2a=True,
            max_delegation_depth=5,
        )

        assert config.enable_sub_agents is False
        assert config.enable_a2a is True
        assert config.max_delegation_depth == 5


class TestHierarchicalAgent:
    """Test HierarchicalAgent."""

    def test_initialization(self):
        """Test hierarchical agent initialization."""
        parent_agent = MockCodeAgent()

        h_agent = HierarchicalAgent(parent_agent)

        assert h_agent.parent_agent is parent_agent
        assert h_agent.registry is not None
        assert h_agent.delegator is not None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        parent_agent = MockCodeAgent()
        config = HierarchicalAgentConfig(
            enable_sub_agents=True,
            enable_a2a=False,
        )

        h_agent = HierarchicalAgent(parent_agent, config)

        assert h_agent.config.enable_sub_agents is True
        assert h_agent.config.enable_a2a is False

    def test_register_sub_agent(self):
        """Test registering a sub-agent."""
        parent_agent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent_agent)

        sub_agent_instance = MockCodeAgent()

        sub_agent = h_agent.register_sub_agent(
            name="Analysis Agent",
            description="Specialized in code analysis",
            agent=sub_agent_instance,
            capabilities=["analysis"],
        )

        assert isinstance(sub_agent, CodeSubAgent)
        assert sub_agent.name == "Analysis Agent"
        assert "analysis" in sub_agent.capabilities

        # Verify it's in the registry
        agents = h_agent.get_sub_agents()
        assert len(agents) == 1
        assert agents[0].name == "Analysis Agent"

    def test_deregister_sub_agent(self):
        """Test deregistering a sub-agent."""
        parent_agent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent_agent)

        sub_agent_instance = MockCodeAgent()
        sub_agent = h_agent.register_sub_agent(
            name="Test Agent",
            description="Test",
            agent=sub_agent_instance,
        )

        assert len(h_agent.get_sub_agents()) == 1

        result = h_agent.deregister_sub_agent(sub_agent.agent_id)
        assert result is True
        assert len(h_agent.get_sub_agents()) == 0

    @pytest.mark.anyio
    async def test_delegate(self):
        """Test delegating a task."""
        parent_agent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent_agent)

        # Register a sub-agent
        sub_agent_instance = MockCodeAgent()
        _sub_agent = h_agent.register_sub_agent(
            name="Analysis Agent",
            description="Specialized in code analysis",
            agent=sub_agent_instance,
            capabilities=["analysis"],
        )

        # Delegate a task
        result = await h_agent.delegate(
            prompt="Analyze this code",
            required_capabilities=["analysis"],
        )

        assert isinstance(result, SubAgentResult)
        assert result.success is True

    @pytest.mark.anyio
    async def test_delegate_multiple(self):
        """Test delegating multiple tasks."""
        parent_agent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent_agent)

        # Register sub-agents
        for i in range(3):
            sub_agent_instance = MockCodeAgent()
            h_agent.register_sub_agent(
                name=f"Agent {i}",
                description=f"Agent {i}",
                agent=sub_agent_instance,
                capabilities=["analysis"],
            )

        # Delegate multiple tasks
        prompts = [
            "Analyze file1.py",
            "Analyze file2.py",
            "Analyze file3.py",
        ]

        results = await h_agent.delegate_multiple(
            prompts=prompts,
            required_capabilities=["analysis"],
        )

        assert len(results) == 3
        for result in results:
            assert isinstance(result, SubAgentResult)

    @pytest.mark.anyio
    async def test_run_with_delegation_no_sub_agents(self):
        """Test running with delegation when no sub-agents available."""
        parent_agent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent_agent)

        # No sub-agents registered, should use parent
        result = await h_agent.run_with_delegation("Analyze code")

        # Result is a MagicMock with output attribute
        assert hasattr(result, "output")
        assert "Mock response" in result.output

    @pytest.mark.anyio
    async def test_run_with_delegation_disabled(self):
        """Test running with delegation disabled."""
        parent_agent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent_agent)

        # Register a sub-agent
        sub_agent_instance = MockCodeAgent()
        h_agent.register_sub_agent(
            name="Test Agent",
            description="Test",
            agent=sub_agent_instance,
        )

        # Run with delegation disabled
        result = await h_agent.run_with_delegation(
            "Analyze code",
            use_sub_agents=False,
        )

        # Result is a MagicMock with output attribute
        assert hasattr(result, "output")
        assert "Mock response" in result.output

    @pytest.mark.anyio
    async def test_run_with_delegation_with_sub_agents(self):
        """Test running with delegation using sub-agents."""
        parent_agent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent_agent)

        # Register a sub-agent
        sub_agent_instance = MockCodeAgent()
        h_agent.register_sub_agent(
            name="Analysis Agent",
            description="Specialized in analysis",
            agent=sub_agent_instance,
            capabilities=["analysis"],
        )

        # Run with delegation
        result = await h_agent.run_with_delegation("Analyze code")

        # Should get result from sub-agent
        assert result is not None

    def test_get_stats(self):
        """Test getting hierarchical agent statistics."""
        parent_agent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent_agent)

        # Register some sub-agents
        for i in range(2):
            sub_agent_instance = MockCodeAgent()
            h_agent.register_sub_agent(
                name=f"Agent {i}",
                description=f"Agent {i}",
                agent=sub_agent_instance,
            )

        stats = h_agent.get_stats()

        assert "registry" in stats
        assert "delegator" in stats
        assert stats["registry"]["total_agents"] == 2

    def test_get_a2a_app_disabled(self):
        """Test getting A2A app when disabled."""
        parent_agent = MockCodeAgent()
        config = HierarchicalAgentConfig(enable_a2a=False)
        h_agent = HierarchicalAgent(parent_agent, config)

        app = h_agent.get_a2a_app()
        assert app is None
