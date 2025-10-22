"""
Core Agent Tests

Tests for the core agent functionality.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import pytest

from code_agent.core import AgentConfig, CodeAgent, create_code_agent
from code_agent.core.types import AgentState


class TestAgentConfig:
    """Test AgentConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig()
        assert config.model == "openai:gpt-4"
        assert config.enable_streaming is False
        assert config.enable_logging is True
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            model="openai:gpt-3.5-turbo",
            enable_streaming=True,
            max_retries=5,
        )
        assert config.model == "openai:gpt-3.5-turbo"
        assert config.enable_streaming is True
        assert config.max_retries == 5

    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = AgentConfig(model="openai:gpt-4")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model"] == "openai:gpt-4"
        assert "enable_streaming" in config_dict


class TestAgentState:
    """Test AgentState."""

    def test_initial_state(self):
        """Test initial agent state."""
        state = AgentState()

        assert state.message_history == []
        assert state.total_usage["input_tokens"] == 0
        assert state.total_usage["output_tokens"] == 0
        assert state.error_history == []
        assert state.streaming_enabled is False

    def test_state_updates(self):
        """Test state updates."""
        state = AgentState()

        state.message_history.append({"role": "user", "content": "test"})
        state.total_usage["input_tokens"] += 100

        assert len(state.message_history) == 1
        assert state.total_usage["input_tokens"] == 100


class TestCodeAgent:
    """Test CodeAgent."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = CodeAgent()

        assert agent.config is not None
        assert agent.state is not None
        assert agent.agent is not None

    def test_agent_with_config(self):
        """Test agent creation with config."""
        config = AgentConfig(model="openai:gpt-3.5-turbo")
        agent = CodeAgent(config)

        assert agent.config.model == "openai:gpt-3.5-turbo"

    def test_get_state(self):
        """Test getting agent state."""
        agent = CodeAgent()
        state = agent.get_state()

        # get_state() returns a dict snapshot, not AgentState object
        assert isinstance(state, dict)
        assert "config" in state
        assert "message_history" in state
        assert "total_usage" in state

    def test_get_error_summary(self):
        """Test getting error summary."""
        agent = CodeAgent()
        summary = agent.get_error_summary()

        assert "total_errors" in summary
        assert summary["total_errors"] == 0

    def test_get_usage_summary(self):
        """Test getting usage summary."""
        agent = CodeAgent()
        summary = agent.get_usage_summary()

        # get_usage_summary() returns a formatted string, not a dict
        assert isinstance(summary, str)
        assert "Input Tokens" in summary or "input_tokens" in summary.lower()


class TestPydanticAIIntegration:
    """Test PydanticAI framework integration."""

    def test_agent_has_pydantic_ai_methods(self):
        """Test that agent has PydanticAI methods."""
        agent = CodeAgent()

        assert hasattr(agent, "run_sync")
        assert hasattr(agent, "run_stream")
        assert hasattr(agent, "iter_nodes")

    def test_pydantic_ai_agent_instance(self):
        """Test that internal agent is PydanticAI Agent instance."""
        from pydantic_ai import Agent as PydanticAIAgent

        agent = CodeAgent()
        assert isinstance(agent.agent, PydanticAIAgent)

    def test_agent_with_usage_limits(self):
        """Test agent creation with PydanticAI UsageLimits."""
        from pydantic_ai import UsageLimits

        usage_limits = UsageLimits(
            request_limit=10,
            total_tokens_limit=5000,
        )
        config = AgentConfig(usage_limits=usage_limits)
        agent = CodeAgent(config)

        assert agent.config.usage_limits is not None
        assert agent.config.usage_limits.request_limit == 10

    def test_agent_tool_decorator_support(self):
        """Test that agent supports @agent.tool decorator."""
        agent = CodeAgent()

        assert hasattr(agent.agent, "tool")
        assert callable(agent.agent.tool)

    def test_agent_model_configuration(self):
        """Test agent model configuration with PydanticAI."""
        config = AgentConfig(model="openai:gpt-4")
        agent = CodeAgent(config)

        assert agent.config.model == "openai:gpt-4"
        assert agent.agent is not None


class TestCreateCodeAgent:
    """Test create_code_agent factory function."""

    def test_create_with_defaults(self):
        """Test creating agent with defaults."""
        agent = create_code_agent()

        assert agent is not None
        assert agent.config.model == "openai:gpt-4"

    def test_create_with_custom_model(self):
        """Test creating agent with custom model."""
        agent = create_code_agent(model="openai:gpt-3.5-turbo")

        assert agent.config.model == "openai:gpt-3.5-turbo"

    def test_create_with_streaming(self):
        """Test creating agent with streaming enabled."""
        agent = create_code_agent(enable_streaming=True)

        assert agent.config.enable_streaming is True

    def test_create_with_usage_limits(self):
        """Test creating agent with usage limits."""
        from pydantic_ai import UsageLimits

        usage_limits = UsageLimits(request_limit=5)
        agent = create_code_agent(usage_limits=usage_limits)

        assert agent.config.usage_limits is not None
        assert agent.config.usage_limits.request_limit == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
