"""
Comprehensive Tests for Hierarchical Agent and A2A Integration

Tests the features:
- Sub-agent functionality
- A2A protocol integration
- Service discovery
- Intelligent routing
- Task delegation enhancements

Author: The Augster
Python Version: 3.12+
"""

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest

from code_agent.core.agent import CodeAgent
from code_agent.core.agent_registry import AgentRegistry
from code_agent.core.hierarchical_agent import (
    HierarchicalAgent,
    HierarchicalAgentConfig,
)
from code_agent.core.sub_agent import (
    SubAgent,
    SubAgentResult,
    SubAgentStatus,
)
from code_agent.core.task_delegator import TaskDelegator

# ============================================================================
# Mock Agents for Testing
# ============================================================================


class MockCodeAgent:
    """Mock CodeAgent for testing."""

    def __init__(self):
        self.agent = self  # Mock the internal agent
        self.run_count = 0

    async def run(self, prompt: str, **kwargs: Any) -> Mock:
        """Mock run method."""
        self.run_count += 1
        return Mock(output=f"Mock response to: {prompt}", usage=lambda: Mock(input_tokens=10, output_tokens=20))


# ============================================================================
# Test CodeAgent.to_a2a()
# ============================================================================


class TestCodeAgentA2A:
    """Test CodeAgent A2A integration."""

    def test_to_a2a_method_exists(self):
        """Test that to_a2a method exists on CodeAgent."""
        agent = CodeAgent(model="openai:gpt-4")
        assert hasattr(agent, "to_a2a")
        assert callable(agent.to_a2a)

    @pytest.mark.skip(reason="Requires fasta2a to be installed")
    def test_to_a2a_creates_app(self):
        """Test that to_a2a creates an ASGI application."""
        agent = CodeAgent(model="openai:gpt-4")
        app = agent.to_a2a(
            name="Test Agent",
            description="Test description",
            version="1.0.0",
        )
        assert app is not None


# ============================================================================
# Test HierarchicalAgent.to_a2a()
# ============================================================================


class TestHierarchicalAgentA2A:
    """Test HierarchicalAgent A2A integration."""

    def test_hierarchical_to_a2a_method_exists(self):
        """Test that to_a2a method exists on HierarchicalAgent."""
        parent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent)
        assert hasattr(h_agent, "to_a2a")
        assert callable(h_agent.to_a2a)

    @pytest.mark.skip(reason="Requires fasta2a to be installed")
    def test_hierarchical_to_a2a_aggregates_skills(self):
        """Test that to_a2a aggregates skills from sub-agents."""
        parent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent)

        # Register sub-agents with different capabilities
        sub1 = MockCodeAgent()
        h_agent.register_sub_agent("Agent1", "Description", sub1, capabilities=["analysis"])

        sub2 = MockCodeAgent()
        h_agent.register_sub_agent("Agent2", "Description", sub2, capabilities=["refactoring"])

        # Create A2A app (should aggregate capabilities)
        app = h_agent.to_a2a(name="Test Hierarchical Agent")
        # Skills should be aggregated from sub-agents
        assert app is not None


# ============================================================================
# Test Enhanced Agent Registry
# ============================================================================


class TestEnhancedAgentRegistry:
    """Test enhanced agent registry features."""

    def test_discover_agents_by_capabilities(self):
        """Test discovering agents by capabilities."""
        registry = AgentRegistry()

        # Create mock sub-agents
        agent1 = SubAgent("Agent1", "Desc1", capabilities=["analysis", "python"])
        agent2 = SubAgent("Agent2", "Desc2", capabilities=["refactoring", "python"])
        agent3 = SubAgent("Agent3", "Desc3", capabilities=["testing"])

        registry.register(agent1)
        registry.register(agent2)
        registry.register(agent3)

        # Discover Python agents
        python_agents = registry.discover_agents(capabilities=["python"])
        assert len(python_agents) == 2

        # Discover analysis + python agents
        analysis_python = registry.discover_agents(capabilities=["analysis", "python"])
        assert len(analysis_python) == 1
        assert analysis_python[0].agent_id == agent1.agent_id

    def test_discover_agents_by_status(self):
        """Test discovering agents by status."""
        registry = AgentRegistry()

        agent1 = SubAgent("Agent1", "Desc1")
        agent1.set_status(SubAgentStatus.IDLE)

        agent2 = SubAgent("Agent2", "Desc2")
        agent2.set_status(SubAgentStatus.BUSY)

        registry.register(agent1)
        registry.register(agent2)

        # Discover idle agents
        idle_agents = registry.discover_agents(status=SubAgentStatus.IDLE)
        assert len(idle_agents) == 1
        assert idle_agents[0].agent_id == agent1.agent_id

    def test_select_best_agent_load_balanced(self):
        """Test selecting best agent with load balancing."""
        registry = AgentRegistry()

        agent1 = SubAgent("Agent1", "Desc1", capabilities=["analysis"])
        agent1.set_status(SubAgentStatus.IDLE)

        agent2 = SubAgent("Agent2", "Desc2", capabilities=["analysis"])
        agent2.set_status(SubAgentStatus.BUSY)

        registry.register(agent1)
        registry.register(agent2)

        # Should prefer idle agent
        best = registry.select_best_agent(capabilities=["analysis"], strategy="load_balanced")
        assert best is not None
        assert best.agent_id == agent1.agent_id

    def test_list_capabilities(self):
        """Test listing all available capabilities."""
        registry = AgentRegistry()

        agent1 = SubAgent("Agent1", "Desc1", capabilities=["analysis", "python"])
        agent2 = SubAgent("Agent2", "Desc2", capabilities=["refactoring", "testing"])

        registry.register(agent1)
        registry.register(agent2)

        capabilities = registry.list_capabilities()
        assert set(capabilities) == {"analysis", "python", "refactoring", "testing"}

    def test_get_agents_with_all_capabilities(self):
        """Test getting agents sorted by capability count."""
        registry = AgentRegistry()

        agent1 = SubAgent("Agent1", "Desc1", capabilities=["a"])
        agent2 = SubAgent("Agent2", "Desc2", capabilities=["a", "b", "c"])
        agent3 = SubAgent("Agent3", "Desc3", capabilities=["a", "b"])

        registry.register(agent1)
        registry.register(agent2)
        registry.register(agent3)

        agents = registry.get_agents_with_all_capabilities()
        assert len(agents) == 3
        assert len(agents[0].capabilities) == 3  # Most capable first
        assert len(agents[1].capabilities) == 2
        assert len(agents[2].capabilities) == 1


# ============================================================================
# Test Enhanced Task Delegator
# ============================================================================


class TestEnhancedTaskDelegator:
    """Test enhanced task delegator features."""

    @pytest.mark.anyio
    async def test_delegate_with_fallback_success(self):
        """Test delegation with fallback on primary success."""
        registry = AgentRegistry()

        # Create successful primary agent
        primary = SubAgent("Primary", "Primary agent")
        registry.register(primary)

        # Mock successful execution
        async def mock_execute(task):
            return SubAgentResult(
                task_id=task.task_id,
                agent_id=primary.agent_id,
                output="Success",
                success=True,
            )

        primary.execute_task = mock_execute

        delegator = TaskDelegator(registry)

        result = await delegator.delegate_with_fallback(
            prompt="Test task",
            primary_agent_id=primary.agent_id,
        )

        assert result.success
        assert result.agent_id == primary.agent_id

    @pytest.mark.anyio
    async def test_delegate_with_timeout(self):
        """Test delegation with timeout."""
        registry = AgentRegistry()

        agent = SubAgent("Agent", "Test agent")
        registry.register(agent)

        # Mock slow execution
        async def mock_slow_execute(task):
            await asyncio.sleep(5)  # Longer than timeout
            return SubAgentResult(
                task_id=task.task_id,
                agent_id=agent.agent_id,
                output="Done",
                success=True,
            )

        agent.execute_task = mock_slow_execute

        delegator = TaskDelegator(registry)

        result = await delegator.delegate_with_timeout(
            prompt="Test task",
            timeout_seconds=0.1,  # Very short timeout
            agent_id=agent.agent_id,
        )

        assert not result.success
        assert "timed out" in result.error.lower()

    def test_score_agent_for_task(self):
        """Test agent scoring for task routing."""
        registry = AgentRegistry()
        agent = SubAgent("Agent", "Test", capabilities=["analysis", "python"])
        agent.set_status(SubAgentStatus.IDLE)
        registry.register(agent)

        delegator = TaskDelegator(registry)

        # Score for matching capabilities
        score = delegator._score_agent_for_task(
            agent,
            required_capabilities=["analysis"],
            task_metadata=None,
        )
        assert score > 0

        # Score for non-matching capabilities
        score_no_match = delegator._score_agent_for_task(
            agent,
            required_capabilities=["nonexistent"],
            task_metadata=None,
        )
        assert score_no_match == 0


# ============================================================================
# Test Broadcast and Chain Delegation
# ============================================================================


class TestAdvancedDelegation:
    """Test advanced delegation patterns."""

    @pytest.mark.anyio
    async def test_broadcast(self):
        """Test broadcasting to multiple agents."""
        parent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent)

        # Register multiple agents
        for i in range(3):
            agent = MockCodeAgent()
            h_agent.register_sub_agent(
                f"Agent{i}",
                f"Description {i}",
                agent,
                capabilities=["analysis"],
            )

        # Mock delegate_and_wait to return results
        async def mock_delegate(prompt, agent_id=None, required_capabilities=None):
            return SubAgentResult(
                task_id=f"task-{agent_id}",
                agent_id=agent_id or "test",
                output=f"Result for {prompt}",
                success=True,
            )

        h_agent.delegator.delegate_and_wait = mock_delegate

        # Broadcast task
        results = await h_agent.broadcast("Test broadcast")

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.anyio
    async def test_chain_delegation(self):
        """Test sequential task chaining."""
        parent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent)

        # Register specialized agents
        for name, caps in [("Analyzer", ["analysis"]), ("Refactorer", ["refactoring"])]:
            agent = MockCodeAgent()
            h_agent.register_sub_agent(name, name, agent, capabilities=caps)

        # Mock delegate_and_wait
        call_count = [0]

        async def mock_delegate(prompt, agent_id=None, required_capabilities=None):
            call_count[0] += 1
            return SubAgentResult(
                task_id=f"task-{call_count[0]}",
                agent_id="test",
                output=f"Result {call_count[0]}",
                success=True,
            )

        h_agent.delegator.delegate_and_wait = mock_delegate

        # Chain tasks
        tasks = [
            ("Analyze code", ["analysis"]),
            ("Refactor code", ["refactoring"]),
        ]

        results = await h_agent.chain_delegation(tasks)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert "Result 1" in results[0].output
        assert "Result 2" in results[1].output


# ============================================================================
# Test A2A Client Integration
# ============================================================================


class TestA2AClientIntegration:
    """Test A2A client communication."""

    @pytest.mark.anyio
    @pytest.mark.skip(reason="Requires running A2A server")
    async def test_call_agent_via_a2a(self):
        """Test calling remote agent via A2A protocol."""
        parent = MockCodeAgent()
        config = HierarchicalAgentConfig(enable_a2a=True)
        h_agent = HierarchicalAgent(parent, config)

        # This would require a running A2A server
        # Just test that the method exists and is callable
        assert hasattr(h_agent, "call_agent_via_a2a")
        assert callable(h_agent.call_agent_via_a2a)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.anyio
    async def test_complete_workflow(self):
        """Test a complete hierarchical agent workflow."""
        # Create parent agent
        parent = MockCodeAgent()
        h_agent = HierarchicalAgent(parent)

        # Register sub-agents
        analysis_agent = MockCodeAgent()
        h_agent.register_sub_agent(
            "Analyzer",
            "Code analyzer",
            analysis_agent,
            capabilities=["analysis"],
        )

        # Verify registration
        sub_agents = h_agent.get_sub_agents()
        assert len(sub_agents) == 1

        # Get statistics
        stats = h_agent.get_stats()
        assert stats["registry"]["total_agents"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
