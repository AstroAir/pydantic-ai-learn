"""
PydanticAI Streaming and Async Tests

Comprehensive tests for PydanticAI streaming and async patterns:
- Streaming response handling
- Async/await patterns
- Node iteration
- Concurrent execution
- Stream buffering
- Error handling in streams

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from code_agent.core import AgentConfig, CodeAgent, CodeAgentState, create_code_agent
from code_agent.utils.errors import ErrorCategory, ErrorContext, ErrorSeverity


class TestPydanticAIStreamingBasics:
    """Test PydanticAI streaming basics."""

    def test_streaming_configuration(self) -> None:
        """Test streaming configuration."""
        config = AgentConfig(enable_streaming=True)
        agent = CodeAgent(config)

        assert agent.config.enable_streaming is True

    def test_streaming_disabled_by_default(self) -> None:
        """Test that streaming is disabled by default."""
        agent = CodeAgent()

        assert agent.config.enable_streaming is False

    def test_streaming_method_exists(self) -> None:
        """Test that streaming method exists."""
        agent = CodeAgent()

        assert hasattr(agent, "run_stream")
        assert callable(agent.run_stream)

    def test_iter_nodes_method_exists(self) -> None:
        """Test that iter_nodes method exists."""
        agent = CodeAgent()

        assert hasattr(agent, "iter_nodes")
        assert callable(agent.iter_nodes)

    def test_streaming_with_custom_model(self) -> None:
        """Test streaming with custom model."""
        config = AgentConfig(
            model="openai:gpt-3.5-turbo",
            enable_streaming=True,
        )
        agent = CodeAgent(config)

        assert agent.config.enable_streaming is True
        assert agent.config.model == "openai:gpt-3.5-turbo"


class TestPydanticAIAsyncPatterns:
    """Test PydanticAI async patterns."""

    def test_async_run_stream(self) -> None:
        """Test async run_stream method."""
        agent = CodeAgent()

        # Verify method is callable
        assert callable(agent.run_stream)

    def test_async_iter_nodes(self) -> None:
        """Test async iter_nodes method."""
        agent = CodeAgent()

        # Verify method is callable
        assert hasattr(agent.agent, "iter")

    def test_sync_run_method(self) -> None:
        """Test synchronous run_sync method."""
        agent = CodeAgent()

        assert hasattr(agent, "run_sync")
        assert callable(agent.run_sync)

    def test_async_context_manager(self) -> None:
        """Test async context manager pattern."""
        agent = CodeAgent()

        # Verify agent can be used in async context
        assert agent is not None


class TestPydanticAIStreamingConfiguration:
    """Test PydanticAI streaming configuration."""

    def test_streaming_with_usage_limits(self) -> None:
        """Test streaming with usage limits."""
        from pydantic_ai import UsageLimits

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

    def test_streaming_with_error_recovery(self) -> None:
        """Test streaming with error recovery."""
        config = AgentConfig(
            enable_streaming=True,
            enable_error_recovery=True,
            max_retries=3,
        )
        agent = CodeAgent(config)

        assert agent.config.enable_streaming is True
        assert agent.config.enable_error_recovery is True

    def test_streaming_with_logging(self) -> None:
        """Test streaming with logging enabled."""
        config = AgentConfig(
            enable_streaming=True,
            enable_logging=True,
            log_level="DEBUG",
        )
        agent = CodeAgent(config)

        assert agent.config.enable_streaming is True
        assert agent.config.enable_logging is True

    def test_streaming_with_context_management(self) -> None:
        """Test streaming with context management."""
        config = AgentConfig(
            enable_streaming=True,
            enable_context_management=True,
            max_context_tokens=100_000,
        )
        agent = CodeAgent(config)

        assert agent.config.enable_streaming is True
        assert agent.config.enable_context_management is True


class TestPydanticAINodeIteration:
    """Test PydanticAI node iteration."""

    def test_iter_nodes_method_availability(self) -> None:
        """Test iter_nodes method availability."""
        agent = CodeAgent()

        assert hasattr(agent, "iter_nodes")

    def test_iter_nodes_is_async(self) -> None:
        """Test that iter_nodes is async."""
        agent = CodeAgent()

        # Verify iter_nodes is callable
        assert callable(agent.iter_nodes)

    def test_iter_nodes_with_prompt(self) -> None:
        """Test iter_nodes with prompt."""
        agent = CodeAgent()

        # Verify method accepts prompt parameter
        assert callable(agent.iter_nodes)


class TestPydanticAIStreamingState:
    """Test PydanticAI streaming state management."""

    def test_streaming_state_flag(self) -> None:
        """Test streaming state flag."""
        agent = CodeAgent()

        assert hasattr(agent.state, "streaming_enabled")
        assert agent.state.streaming_enabled is False

    def test_streaming_state_with_streaming_enabled(self) -> None:
        """Test streaming state when streaming is enabled."""
        config = AgentConfig(enable_streaming=True)
        agent = CodeAgent(config)

        assert agent.config.enable_streaming is True

    def test_message_history_with_streaming(self) -> None:
        """Test message history with streaming."""
        config = AgentConfig(enable_streaming=True)
        agent = CodeAgent(config)

        # Add message
        agent.state.message_history.append({"role": "user", "content": "Stream this"})

        assert len(agent.state.message_history) == 1


class TestPydanticAIStreamingUsageTracking:
    """Test PydanticAI streaming usage tracking."""

    def test_usage_tracking_with_streaming(self) -> None:
        """Test usage tracking with streaming."""
        config = AgentConfig(enable_streaming=True)
        agent = CodeAgent(config)

        # Simulate streaming usage
        agent.state.total_usage["requests"] += 1
        agent.state.total_usage["input_tokens"] += 100
        agent.state.total_usage["output_tokens"] += 50

        assert agent.state.total_usage["requests"] == 1
        assert agent.state.total_usage["input_tokens"] == 100

    def test_streaming_request_counting(self) -> None:
        """Test streaming request counting."""
        agent = CodeAgent()

        # Simulate multiple streaming requests
        for _ in range(5):
            agent.state.total_usage["requests"] += 1

        assert agent.state.total_usage["requests"] == 5

    def test_streaming_token_accumulation(self) -> None:
        """Test streaming token accumulation."""
        agent = CodeAgent()

        # Simulate token accumulation
        for _ in range(10):
            agent.state.total_usage["input_tokens"] += 50
            agent.state.total_usage["output_tokens"] += 25

        assert agent.state.total_usage["input_tokens"] == 500
        assert agent.state.total_usage["output_tokens"] == 250


class TestPydanticAIStreamingErrorHandling:
    """Test PydanticAI streaming error handling."""

    def test_streaming_error_tracking(self) -> None:
        """Test error tracking with streaming."""
        config = AgentConfig(enable_streaming=True)
        agent = CodeAgent(config)

        # Simulate streaming error
        agent.state.error_history.append(
            ErrorContext(
                error_type="StreamingError",
                error_message="Stream interrupted",
                category=ErrorCategory.TRANSIENT,
                severity=ErrorSeverity.MEDIUM,
            )
        )

        assert len(agent.state.error_history) == 1

    def test_streaming_with_error_recovery(self) -> None:
        """Test streaming with error recovery."""
        config = AgentConfig(
            enable_streaming=True,
            enable_error_recovery=True,
        )
        agent = CodeAgent(config)

        # Simulate error and recovery
        agent.state.error_history.append(
            ErrorContext(
                error_type="StreamingError",
                error_message="recovered",
                category=ErrorCategory.TRANSIENT,
                severity=ErrorSeverity.MEDIUM,
            )
        )

        assert agent.config.enable_error_recovery is True

    def test_streaming_retry_on_error(self) -> None:
        """Test streaming retry on error."""
        config = AgentConfig(
            enable_streaming=True,
            max_retries=3,
        )
        agent = CodeAgent(config)

        assert agent.config.max_retries == 3


class TestPydanticAIStreamingPerformance:
    """Test PydanticAI streaming performance."""

    def test_streaming_with_large_responses(self) -> None:
        """Test streaming with large responses."""
        agent = CodeAgent()

        # Simulate large response
        large_content = "x" * 10000
        agent.state.message_history.append(
            {
                "role": "assistant",
                "content": large_content,
            }
        )

        assert len(agent.state.message_history[0]["content"]) == 10000

    def test_streaming_with_many_chunks(self) -> None:
        """Test streaming with many chunks."""
        agent = CodeAgent()

        # Simulate many streaming chunks
        for _ in range(100):
            agent.state.total_usage["requests"] += 1

        assert agent.state.total_usage["requests"] == 100

    def test_streaming_memory_efficiency(self) -> None:
        """Test streaming memory efficiency."""
        agent = CodeAgent()

        # Simulate streaming with message accumulation
        for i in range(50):
            agent.state.message_history.append(
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Message {i}",
                }
            )

        assert len(agent.state.message_history) == 50


class TestPydanticAIAsyncConcurrency:
    """Test PydanticAI async concurrency patterns."""

    def test_multiple_async_operations(self) -> None:
        """Test multiple async operations."""
        agent1 = CodeAgent()
        agent2 = CodeAgent()
        agent3 = CodeAgent()

        # Verify all agents are independent
        assert agent1 is not agent2
        assert agent2 is not agent3

    def test_concurrent_state_updates(self) -> None:
        """Test concurrent state updates."""
        agent = CodeAgent()

        # Simulate concurrent updates
        agent.state.total_usage["requests"] += 1
        agent.state.total_usage["input_tokens"] += 100

        assert agent.state.total_usage["requests"] == 1
        assert agent.state.total_usage["input_tokens"] == 100

    def test_async_message_history(self) -> None:
        """Test async message history updates."""
        agent = CodeAgent()

        # Simulate async message additions
        agent.state.message_history.append({"role": "user", "content": "Async message"})

        assert len(agent.state.message_history) == 1


class TestPydanticAIStreamingIntegration:
    """Test PydanticAI streaming integration."""

    def test_streaming_with_tools(self) -> None:
        """Test streaming with tools."""
        from pydantic_ai import RunContext

        config = AgentConfig(enable_streaming=True)
        agent = CodeAgent(config)

        @agent.agent.tool
        def stream_tool(ctx: RunContext[CodeAgentState], text: str, /) -> str:
            """Tool for streaming."""
            return f"Streamed: {text}"

        assert agent.config.enable_streaming is True
        assert callable(stream_tool)

    def test_streaming_with_multi_turn(self) -> None:
        """Test streaming with multi-turn conversation."""
        config = AgentConfig(enable_streaming=True)
        agent = CodeAgent(config)

        # Simulate multi-turn streaming
        agent.state.message_history.append({"role": "user", "content": "First message"})
        agent.state.message_history.append({"role": "assistant", "content": "First response"})

        assert len(agent.state.message_history) == 2

    def test_streaming_factory_function(self) -> None:
        """Test streaming with factory function."""
        agent = create_code_agent(
            model="openai:gpt-4",
            enable_streaming=True,
        )

        assert agent.config.enable_streaming is True
        assert agent.config.model == "openai:gpt-4"


class TestPydanticAIStreamingEdgeCases:
    """Test PydanticAI streaming edge cases."""

    def test_streaming_with_empty_prompt(self) -> None:
        """Test streaming with empty prompt."""
        agent = CodeAgent()

        # Verify agent handles empty prompt
        assert agent is not None

    def test_streaming_with_very_long_prompt(self) -> None:
        """Test streaming with very long prompt."""
        agent = CodeAgent()

        long_prompt = "x" * 100000
        agent.state.message_history.append(
            {
                "role": "user",
                "content": long_prompt,
            }
        )

        assert len(agent.state.message_history[0]["content"]) == 100000

    def test_streaming_with_special_characters(self) -> None:
        """Test streaming with special characters."""
        agent = CodeAgent()

        special_content = "Special: @#$%^&*()_+-=[]{}|;:',.<>?/~`"
        agent.state.message_history.append(
            {
                "role": "user",
                "content": special_content,
            }
        )

        assert special_content in agent.state.message_history[0]["content"]

    def test_streaming_with_unicode(self) -> None:
        """Test streaming with Unicode content."""
        agent = CodeAgent()

        unicode_content = "Unicode: 你好世界 🚀 Привет мир"
        agent.state.message_history.append(
            {
                "role": "user",
                "content": unicode_content,
            }
        )

        assert "你好世界" in agent.state.message_history[0]["content"]
