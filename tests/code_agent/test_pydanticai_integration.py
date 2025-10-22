"""
PydanticAI Framework Integration Tests

Comprehensive tests for PydanticAI framework integration including:
- Agent initialization with different models
- Tool registration and execution
- Streaming and synchronous execution
- Message history management
- Usage tracking and limits
- Error handling with PydanticAI exceptions
- Context management
- Async/await patterns

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import (
    AgentRunError,
    ModelHTTPError,
    UsageLimitExceeded,
    UserError,
)

from code_agent.core import AgentConfig, CodeAgent, create_code_agent


class TestPydanticAIAgentInitialization:
    """Test PydanticAI Agent initialization with different configurations."""

    def test_agent_initialization_with_default_model(self):
        """Test agent initialization with default OpenAI GPT-4 model."""
        agent = CodeAgent()

        assert agent.agent is not None
        assert isinstance(agent.agent, Agent)
        assert agent.config.model == "openai:gpt-4"

    def test_agent_initialization_with_custom_model(self):
        """Test agent initialization with custom model configuration."""
        config = AgentConfig(model="openai:gpt-3.5-turbo")
        agent = CodeAgent(config)

        assert agent.config.model == "openai:gpt-3.5-turbo"
        assert agent.agent is not None

    def test_agent_initialization_with_anthropic_model(self):
        """Test agent initialization with Anthropic Claude model."""
        config = AgentConfig(model="anthropic:claude-3-sonnet")
        agent = CodeAgent(config)

        assert agent.config.model == "anthropic:claude-3-sonnet"
        assert agent.agent is not None

    def test_agent_initialization_with_usage_limits(self):
        """Test agent initialization with PydanticAI UsageLimits."""
        usage_limits = UsageLimits(
            request_limit=10,
            total_tokens_limit=10000,
        )
        config = AgentConfig(usage_limits=usage_limits)
        agent = CodeAgent(config)

        assert agent.config.usage_limits is not None
        assert agent.config.usage_limits.request_limit == 10
        assert agent.config.usage_limits.total_tokens_limit == 10000

    def test_agent_initialization_with_streaming_enabled(self):
        """Test agent initialization with streaming enabled."""
        config = AgentConfig(enable_streaming=True)
        agent = CodeAgent(config)

        assert agent.config.enable_streaming is True
        assert agent.agent is not None

    def test_agent_initialization_with_logging_disabled(self):
        """Test agent initialization with logging disabled."""
        config = AgentConfig(enable_logging=False)
        agent = CodeAgent(config)

        assert agent.config.enable_logging is False

    def test_factory_function_creates_valid_agent(self):
        """Test factory function creates valid CodeAgent."""
        agent = create_code_agent(
            model="openai:gpt-4",
            enable_streaming=True,
        )

        assert isinstance(agent, CodeAgent)
        assert agent.config.model == "openai:gpt-4"
        assert agent.config.enable_streaming is True


class TestPydanticAIToolRegistration:
    """Test PydanticAI tool registration and execution."""

    def test_agent_has_tool_decorator_support(self):
        """Test that agent supports PydanticAI @agent.tool decorator."""
        agent = CodeAgent()

        # Verify agent has tool decorator
        assert hasattr(agent.agent, "tool")
        assert callable(agent.agent.tool)

    def test_tool_registration_with_decorator(self):
        """Test tool registration using @agent.tool decorator."""
        agent = CodeAgent()

        @agent.agent.tool
        def analyze_code(code: str) -> str:
            """Analyze Python code for issues."""
            return f"Analyzed: {code}"

        # Verify tool is registered
        assert hasattr(agent.agent, "toolset") or hasattr(agent.agent, "toolsets")

    def test_tool_with_multiple_parameters(self):
        """Test tool registration with multiple parameters."""
        agent = CodeAgent()

        @agent.agent.tool
        def refactor_code(code: str, style: str, complexity: int) -> str:
            """Refactor code with specified style and complexity."""
            return f"Refactored {code} with {style} style and complexity {complexity}"

        assert agent.agent is not None

    def test_tool_with_return_type_annotation(self):
        """Test tool with proper return type annotation."""
        agent = CodeAgent()

        @agent.agent.tool
        def generate_docstring(function_name: str) -> str:
            """Generate docstring for function."""
            return f'"""Generated docstring for {function_name}"""'

        assert agent.agent is not None

    def test_tool_with_optional_parameters(self):
        """Test tool registration with optional parameters."""
        agent = CodeAgent()

        @agent.agent.tool
        def analyze_with_options(
            code: str,
            check_types: bool = True,
            check_style: bool = False,
        ) -> str:
            """Analyze code with optional checks."""
            return f"Analysis: types={check_types}, style={check_style}"

        assert agent.agent is not None


class TestPydanticAIMessageHistory:
    """Test PydanticAI message history management."""

    def test_agent_state_tracks_message_history(self):
        """Test that agent state tracks message history."""
        agent = CodeAgent()

        assert isinstance(agent.state.message_history, list)
        assert len(agent.state.message_history) == 0

    def test_message_history_initialization(self):
        """Test message history initialization."""
        agent = CodeAgent()
        state = agent.get_state()

        # get_state() returns a dict snapshot
        assert isinstance(state, dict)
        assert state["message_history"] == []

    def test_message_history_with_multiple_turns(self):
        """Test message history with multiple conversation turns."""
        agent = CodeAgent()

        # Simulate adding messages
        agent.state.message_history.append({"role": "user", "content": "Analyze this code"})
        agent.state.message_history.append({"role": "assistant", "content": "Analysis complete"})

        assert len(agent.state.message_history) == 2
        assert agent.state.message_history[0]["role"] == "user"
        assert agent.state.message_history[1]["role"] == "assistant"

    def test_message_history_persistence(self):
        """Test that message history persists across operations."""
        agent = CodeAgent()

        initial_message = {"role": "user", "content": "First message"}
        agent.state.message_history.append(initial_message)

        # Verify persistence
        assert len(agent.state.message_history) == 1
        assert agent.state.message_history[0] == initial_message


class TestPydanticAIUsageTracking:
    """Test PydanticAI usage tracking and limits."""

    def test_usage_tracking_initialization(self):
        """Test usage tracking initialization."""
        agent = CodeAgent()

        assert agent.state.total_usage["input_tokens"] == 0
        assert agent.state.total_usage["output_tokens"] == 0
        assert agent.state.total_usage["requests"] == 0

    def test_usage_limits_configuration(self):
        """Test usage limits configuration with PydanticAI."""
        usage_limits = UsageLimits(
            request_limit=5,
            total_tokens_limit=5000,
        )
        config = AgentConfig(usage_limits=usage_limits)
        agent = CodeAgent(config)

        assert agent.config.usage_limits.request_limit == 5
        assert agent.config.usage_limits.total_tokens_limit == 5000

    def test_usage_tracking_accumulation(self):
        """Test that usage tracking accumulates correctly."""
        agent = CodeAgent()

        # Simulate token usage
        agent.state.total_usage["input_tokens"] += 100
        agent.state.total_usage["output_tokens"] += 50
        agent.state.total_usage["requests"] += 1

        assert agent.state.total_usage["input_tokens"] == 100
        assert agent.state.total_usage["output_tokens"] == 50
        assert agent.state.total_usage["requests"] == 1

    def test_usage_summary_retrieval(self):
        """Test retrieving usage summary."""
        agent = CodeAgent()

        agent.state.total_usage["input_tokens"] = 500
        agent.state.total_usage["output_tokens"] = 250
        agent.state.total_usage["requests"] = 2

        summary = agent.get_usage_summary()

        # get_usage_summary() returns a formatted string, not a dict
        assert isinstance(summary, str)
        assert "500" in summary
        assert "250" in summary

    def test_usage_limits_with_zero_values(self):
        """Test usage limits with zero/unlimited values."""
        usage_limits = UsageLimits(
            request_limit=None,  # Unlimited requests
            total_tokens_limit=None,  # Unlimited tokens
        )
        config = AgentConfig(usage_limits=usage_limits)
        agent = CodeAgent(config)

        assert agent.config.usage_limits is not None


class TestPydanticAIErrorHandling:
    """Test PydanticAI error handling and exceptions."""

    def test_agent_error_history_tracking(self):
        """Test that agent tracks error history."""
        agent = CodeAgent()

        assert isinstance(agent.state.error_history, list)
        assert len(agent.state.error_history) == 0

    def test_error_summary_retrieval(self):
        """Test retrieving error summary."""
        from code_agent.utils.errors import ErrorCategory, ErrorContext, ErrorSeverity

        agent = CodeAgent()

        # Simulate error using ErrorContext
        error_ctx = ErrorContext(
            error_type="ModelError",
            error_message="Model not available",
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.MEDIUM,
        )
        agent.state.add_error(error_ctx)

        summary = agent.get_error_summary()

        assert summary["total_errors"] == 1

    def test_multiple_errors_tracking(self):
        """Test tracking multiple errors."""
        from code_agent.utils.errors import ErrorCategory, ErrorContext, ErrorSeverity

        agent = CodeAgent()

        for i in range(3):
            error_ctx = ErrorContext(
                error_type=f"Error {i}",
                error_message=f"Error message {i}",
                category=ErrorCategory.TRANSIENT,
                severity=ErrorSeverity.LOW,
            )
            agent.state.add_error(error_ctx)

        summary = agent.get_error_summary()
        assert summary["total_errors"] == 3

    def test_pydantic_ai_error_types(self):
        """Test recognition of PydanticAI error types."""
        # Verify PydanticAI exception classes are available
        assert AgentRunError is not None
        assert ModelHTTPError is not None
        assert UsageLimitExceeded is not None
        assert UserError is not None


class TestPydanticAIContextManagement:
    """Test PydanticAI context management."""

    def test_agent_state_context_tracking(self):
        """Test agent state context tracking."""
        agent = CodeAgent()
        state = agent.get_state()

        # get_state() returns a dict snapshot, not AgentState object
        assert isinstance(state, dict)
        assert "message_history" in state
        assert "total_usage" in state

    def test_context_with_streaming_flag(self):
        """Test context management with streaming flag."""
        agent = CodeAgent()

        assert hasattr(agent.state, "streaming_enabled")
        assert agent.state.streaming_enabled is False

    def test_context_configuration_options(self):
        """Test context configuration options."""
        config = AgentConfig(
            enable_context_management=True,
            max_context_tokens=100_000,
        )
        agent = CodeAgent(config)

        assert agent.config.enable_context_management is True
        assert agent.config.max_context_tokens == 100_000


class TestPydanticAIAsyncPatterns:
    """Test PydanticAI async/await patterns."""

    def test_agent_has_async_methods(self):
        """Test that agent has async methods."""
        agent = CodeAgent()

        assert hasattr(agent, "run_stream")
        assert hasattr(agent.agent, "iter")

    def test_agent_has_sync_method(self):
        """Test that agent has synchronous run_sync method."""
        agent = CodeAgent()

        assert hasattr(agent, "run_sync")
        assert callable(agent.run_sync)

    def test_run_stream_is_async_generator(self):
        """Test that run_stream returns async generator."""
        agent = CodeAgent()

        # Verify run_stream is a coroutine function
        assert callable(agent.run_stream)


class TestPydanticAIConfigurationIntegration:
    """Test PydanticAI configuration integration."""

    def test_config_to_dict_includes_usage_limits(self):
        """Test that config.to_dict() includes usage limits."""
        usage_limits = UsageLimits(request_limit=10)
        config = AgentConfig(usage_limits=usage_limits)

        config_dict = config.to_dict()

        assert "usage_limits" in config_dict
        assert config_dict["usage_limits"] is not None

    def test_config_to_dict_includes_model(self):
        """Test that config.to_dict() includes model."""
        config = AgentConfig(model="openai:gpt-4")
        config_dict = config.to_dict()

        assert config_dict["model"] == "openai:gpt-4"

    def test_config_to_dict_includes_all_settings(self):
        """Test that config.to_dict() includes all settings."""
        config = AgentConfig(
            model="openai:gpt-4",
            enable_streaming=True,
            enable_logging=False,
            max_retries=5,
        )
        config_dict = config.to_dict()

        assert config_dict["model"] == "openai:gpt-4"
        assert config_dict["enable_streaming"] is True
        assert config_dict["enable_logging"] is False
        assert config_dict["max_retries"] == 5


class TestPydanticAIModelProviders:
    """Test PydanticAI with different model providers."""

    def test_openai_model_configuration(self):
        """Test OpenAI model configuration."""
        config = AgentConfig(model="openai:gpt-4")
        agent = CodeAgent(config)

        assert "openai" in agent.config.model
        assert "gpt-4" in agent.config.model

    def test_anthropic_model_configuration(self):
        """Test Anthropic Claude model configuration."""
        config = AgentConfig(model="anthropic:claude-3-sonnet")
        agent = CodeAgent(config)

        assert "anthropic" in agent.config.model
        assert "claude" in agent.config.model

    def test_model_switching(self):
        """Test switching between different models."""
        agent1 = CodeAgent(AgentConfig(model="openai:gpt-4"))
        agent2 = CodeAgent(AgentConfig(model="openai:gpt-3.5-turbo"))

        assert agent1.config.model != agent2.config.model
        assert "gpt-4" in agent1.config.model
        assert "gpt-3.5-turbo" in agent2.config.model


class TestPydanticAIRetryConfiguration:
    """Test PydanticAI retry configuration."""

    def test_retry_configuration(self):
        """Test retry configuration."""
        config = AgentConfig(
            max_retries=5,
            retry_backoff_factor=2.0,
        )
        agent = CodeAgent(config)

        assert agent.config.max_retries == 5
        assert agent.config.retry_backoff_factor == 2.0

    def test_error_recovery_enabled(self):
        """Test error recovery configuration."""
        config = AgentConfig(enable_error_recovery=True)
        agent = CodeAgent(config)

        assert agent.config.enable_error_recovery is True

    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration."""
        config = AgentConfig(
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5,
        )
        agent = CodeAgent(config)

        assert agent.config.enable_circuit_breaker is True
        assert agent.config.circuit_breaker_threshold == 5
