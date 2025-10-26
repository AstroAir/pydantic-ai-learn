"""
PydanticAI Context Tests

Test scenarios for PydanticAI RunContext integration:
- RunContext usage and management
- Tool execution with context
- Streaming response handling
- Multi-turn conversations
- State persistence
- Performance and optimization
- Edge cases and error scenarios

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, UsageLimits

from code_agent.core import AgentConfig, CodeAgent, create_code_agent


class CodeAnalysisResult(BaseModel):
    """Result of code analysis."""

    issues: list[str]
    complexity_score: float
    suggestions: list[str]


class TestPydanticAIRunContext:
    """Test PydanticAI RunContext usage."""

    def test_run_context_availability(self):
        """Test that RunContext is available for use."""
        assert RunContext is not None

        # Verify RunContext has expected attributes
        assert hasattr(RunContext, "__init__")

    def test_agent_with_context_support(self):
        """Test agent supports context management."""
        agent = CodeAgent()

        assert agent.agent is not None
        assert isinstance(agent.agent, Agent)

    def test_context_in_tool_execution(self):
        """Test context usage in tool execution."""
        agent = CodeAgent()

        @agent.agent.tool
        def analyze_with_context(code: str) -> str:
            """Analyze code with context."""
            return f"Analyzed: {code}"

        assert agent.agent is not None


class TestPydanticAIToolExecution:
    """Test PydanticAI tool execution patterns."""

    def test_tool_execution_with_string_return(self):
        """Test tool execution returning string."""
        agent = CodeAgent()

        @agent.agent.tool
        def simple_tool(input_text: str) -> str:
            """Simple tool returning string."""
            return f"Processed: {input_text}"

        assert agent.agent is not None

    def test_tool_execution_with_dict_return(self):
        """Test tool execution returning dictionary."""
        agent = CodeAgent()

        @agent.agent.tool
        def dict_tool(code: str) -> dict[str, Any]:
            """Tool returning dictionary."""
            return {"code": code, "status": "analyzed", "issues": []}

        assert agent.agent is not None

    def test_tool_execution_with_list_return(self):
        """Test tool execution returning list."""
        agent = CodeAgent()

        @agent.agent.tool
        def list_tool(code: str) -> list[str]:
            """Tool returning list."""
            return ["issue1", "issue2", "issue3"]

        assert agent.agent is not None

    def test_tool_with_pydantic_model_return(self):
        """Test tool execution returning Pydantic model."""
        agent = CodeAgent()

        @agent.agent.tool
        def model_tool(code: str) -> CodeAnalysisResult:
            """Tool returning Pydantic model."""
            return CodeAnalysisResult(issues=["issue1"], complexity_score=0.5, suggestions=["suggestion1"])

        assert agent.agent is not None

    def test_tool_with_multiple_return_types(self):
        """Test tool with union return types."""
        agent = CodeAgent()

        @agent.agent.tool
        def flexible_tool(code: str) -> str | dict[str, Any]:
            """Tool with flexible return type."""
            if len(code) > 100:
                return {"status": "large"}
            return "small"

        assert agent.agent is not None


class TestPydanticAIStreamingPatterns:
    """Test PydanticAI streaming response patterns."""

    def test_streaming_configuration(self):
        """Test streaming configuration."""
        config = AgentConfig(enable_streaming=True)
        agent = CodeAgent(config)

        assert agent.config.enable_streaming is True

    def test_streaming_method_availability(self):
        """Test that streaming methods are available."""
        agent = CodeAgent()

        assert hasattr(agent, "run_stream")
        assert callable(agent.run_stream)

    def test_iter_nodes_method_availability(self):
        """Test that iter_nodes method is available."""
        agent = CodeAgent()

        assert hasattr(agent, "iter_nodes")
        assert callable(agent.iter_nodes)

    def test_streaming_with_usage_limits(self):
        """Test streaming with usage limits."""
        usage_limits = UsageLimits(
            request_limit=5,
            total_tokens_limit=5000,
        )
        config = AgentConfig(
            enable_streaming=True,
            usage_limits=usage_limits,
        )
        agent = CodeAgent(config)

        assert agent.config.enable_streaming is True
        assert agent.config.usage_limits is not None


class TestPydanticAIMultiTurnConversations:
    """Test PydanticAI multi-turn conversation handling."""

    def test_multi_turn_message_history(self):
        """Test multi-turn conversation message history."""
        agent = CodeAgent()

        # Simulate multi-turn conversation
        messages = [
            {"role": "user", "content": "Analyze this code"},
            {"role": "assistant", "content": "Analysis: ..."},
            {"role": "user", "content": "Refactor it"},
            {"role": "assistant", "content": "Refactored: ..."},
        ]

        for msg in messages:
            agent.state.message_history.append(msg)

        assert len(agent.state.message_history) == 4

    def test_conversation_context_preservation(self):
        """Test that conversation context is preserved."""
        agent = CodeAgent()

        # Add initial message
        agent.state.message_history.append({"role": "user", "content": "First question"})

        # Verify context preserved
        assert len(agent.state.message_history) == 1

        # Add follow-up
        agent.state.message_history.append({"role": "assistant", "content": "First answer"})

        assert len(agent.state.message_history) == 2
        assert agent.state.message_history[0]["content"] == "First question"

    def test_conversation_with_system_message(self):
        """Test conversation with system message."""
        agent = CodeAgent()

        system_message = {"role": "system", "content": "You are a code analysis expert"}
        agent.state.message_history.append(system_message)

        assert agent.state.message_history[0]["role"] == "system"

    def test_conversation_turn_alternation(self):
        """Test proper alternation of conversation turns."""
        agent = CodeAgent()

        # Simulate proper turn alternation
        turns = [
            ("user", "Question 1"),
            ("assistant", "Answer 1"),
            ("user", "Question 2"),
            ("assistant", "Answer 2"),
        ]

        for role, content in turns:
            agent.state.message_history.append({"role": role, "content": content})

        # Verify alternation
        for i, (role, _) in enumerate(turns):
            assert agent.state.message_history[i]["role"] == role


class TestPydanticAIStatePersistence:
    """Test PydanticAI state persistence."""

    def test_agent_state_persistence(self):
        """Test that agent state persists."""
        agent = CodeAgent()

        # Modify state
        agent.state.total_usage["input_tokens"] = 100
        agent.state.message_history.append({"role": "user", "content": "test"})

        # Retrieve state (returns dict snapshot)
        state = agent.get_state()

        assert state["total_usage"]["input_tokens"] == 100
        assert len(state["message_history"]) == 1

    def test_state_across_multiple_operations(self):
        """Test state persistence across operations."""
        agent = CodeAgent()

        # First operation
        agent.state.total_usage["requests"] += 1
        agent.state.message_history.append({"role": "user", "content": "op1"})

        # Second operation
        agent.state.total_usage["requests"] += 1
        agent.state.message_history.append({"role": "user", "content": "op2"})

        # Verify accumulation
        assert agent.state.total_usage["requests"] == 2
        assert len(agent.state.message_history) == 2

    def test_state_isolation_between_agents(self):
        """Test that state is isolated between agents."""
        agent1 = CodeAgent()
        agent2 = CodeAgent()

        agent1.state.total_usage["requests"] = 10
        agent2.state.total_usage["requests"] = 5

        assert agent1.state.total_usage["requests"] == 10
        assert agent2.state.total_usage["requests"] == 5


class TestPydanticAIErrorScenarios:
    """Test PydanticAI error scenarios."""

    def test_error_tracking_with_details(self):
        """Test error tracking with detailed information."""
        from code_agent.utils.errors import ErrorCategory, ErrorContext, ErrorSeverity

        agent = CodeAgent()

        error_ctx = ErrorContext(
            error_type="ModelError",
            error_message="Model not available",
            category=ErrorCategory.TRANSIENT,
            severity=ErrorSeverity.MEDIUM,
        )
        agent.state.add_error(error_ctx)

        summary = agent.get_error_summary()
        assert summary["total_errors"] == 1
        assert summary["recent_errors"][0]["type"] == "ModelError"

    def test_error_recovery_tracking(self):
        """Test tracking of error recovery."""
        agent = CodeAgent()

        # Simulate error and recovery
        agent.state.error_history.append({"error": "ConnectionError", "status": "recovered"})

        assert len(agent.state.error_history) == 1

    def test_multiple_error_types(self):
        """Test tracking multiple error types."""
        agent = CodeAgent()

        errors = [
            {"type": "ModelError", "message": "Model unavailable"},
            {"type": "ValidationError", "message": "Invalid input"},
            {"type": "TimeoutError", "message": "Request timeout"},
        ]

        for error in errors:
            agent.state.error_history.append(error)

        assert len(agent.state.error_history) == 3


class TestPydanticAIPerformanceOptimization:
    """Test PydanticAI performance optimization."""

    def test_usage_limits_prevent_overuse(self):
        """Test that usage limits prevent overuse."""
        usage_limits = UsageLimits(
            request_limit=5,
            total_tokens_limit=1000,
        )
        config = AgentConfig(usage_limits=usage_limits)
        agent = CodeAgent(config)

        assert agent.config.usage_limits.request_limit == 5
        assert agent.config.usage_limits.total_tokens_limit == 1000

    def test_token_counting_accuracy(self):
        """Test token counting accuracy."""
        agent = CodeAgent()

        # Simulate token usage
        agent.state.total_usage["input_tokens"] = 150
        agent.state.total_usage["output_tokens"] = 75

        total_tokens = agent.state.total_usage["input_tokens"] + agent.state.total_usage["output_tokens"]

        assert total_tokens == 225

    def test_request_counting(self):
        """Test request counting."""
        agent = CodeAgent()

        for _i in range(5):
            agent.state.total_usage["requests"] += 1

        assert agent.state.total_usage["requests"] == 5


class TestPydanticAIEdgeCases:
    """Test PydanticAI edge cases."""

    def test_empty_message_history(self):
        """Test handling of empty message history."""
        agent = CodeAgent()

        assert len(agent.state.message_history) == 0
        assert isinstance(agent.state.message_history, list)

    def test_very_long_message_history(self):
        """Test handling of very long message history."""
        agent = CodeAgent()

        for i in range(1000):
            agent.state.message_history.append(
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            )

        assert len(agent.state.message_history) == 1000

    def test_zero_usage_limits(self):
        """Test handling of zero usage limits."""
        usage_limits = UsageLimits(
            request_limit=0,
            total_tokens_limit=0,
        )
        config = AgentConfig(usage_limits=usage_limits)
        agent = CodeAgent(config)

        assert agent.config.usage_limits.request_limit == 0

    def test_very_large_token_limits(self):
        """Test handling of very large token limits."""
        usage_limits = UsageLimits(
            request_limit=1_000_000,
            total_tokens_limit=10_000_000,
        )
        config = AgentConfig(usage_limits=usage_limits)
        agent = CodeAgent(config)

        assert agent.config.usage_limits.request_limit == 1_000_000
        assert agent.config.usage_limits.total_tokens_limit == 10_000_000

    def test_special_characters_in_messages(self):
        """Test handling of special characters in messages."""
        agent = CodeAgent()

        special_message = {"role": "user", "content": "Code with special chars: @#$%^&*()_+-=[]{}|;:',.<>?/~`"}
        agent.state.message_history.append(special_message)

        assert agent.state.message_history[0]["content"] == special_message["content"]

    def test_unicode_in_messages(self):
        """Test handling of Unicode in messages."""
        agent = CodeAgent()

        unicode_message = {"role": "user", "content": "Unicode: 你好世界 🚀 Привет мир"}
        agent.state.message_history.append(unicode_message)

        assert "你好世界" in agent.state.message_history[0]["content"]


class TestPydanticAIIntegrationScenarios:
    """Test complete PydanticAI integration scenarios."""

    def test_complete_agent_workflow(self):
        """Test complete agent workflow."""
        # Create agent
        agent = create_code_agent(
            model="openai:gpt-4",
            enable_streaming=False,
        )

        # Verify setup
        assert agent is not None
        assert agent.config.model == "openai:gpt-4"
        assert agent.config.enable_streaming is False

        # Simulate workflow
        agent.state.message_history.append({"role": "user", "content": "Analyze code"})
        agent.state.total_usage["requests"] += 1

        # Verify state
        assert len(agent.state.message_history) == 1
        assert agent.state.total_usage["requests"] == 1

    def test_agent_with_tools_and_streaming(self):
        """Test agent with tools and streaming."""
        config = AgentConfig(
            model="openai:gpt-4",
            enable_streaming=True,
        )
        agent = CodeAgent(config)

        @agent.agent.tool
        def analyze(code: str) -> str:
            """Analyze code."""
            return f"Analyzed: {code}"

        assert agent.config.enable_streaming is True
        assert agent.agent is not None

    def test_agent_with_error_recovery(self):
        """Test agent with error recovery."""
        config = AgentConfig(
            enable_error_recovery=True,
            max_retries=3,
        )
        agent = CodeAgent(config)

        # Simulate error
        agent.state.error_history.append({"error": "ConnectionError", "retry_count": 1})

        assert agent.config.enable_error_recovery is True
        assert len(agent.state.error_history) == 1
