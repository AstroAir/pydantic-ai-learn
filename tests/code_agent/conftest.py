"""
Pytest configuration and fixtures for Code Agent tests.

Provides:
- Mock fixtures for PydanticAI Agent
- Mock fixtures for API clients
- Configuration fixtures
- Utility fixtures

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, UsageLimits

# Check if trio is available
try:
    import trio  # type: ignore[import-not-found]  # noqa: F401

    TRIO_AVAILABLE = True
except ImportError:
    TRIO_AVAILABLE = False

# Skip trio tests if trio is not installed
skip_if_no_trio = pytest.mark.skipif(not TRIO_AVAILABLE, reason="Requires trio to be installed")


def pytest_collection_modifyitems(config, items):
    """Skip trio-backed tests (backend not supported in this repo)."""
    skip_trio = pytest.mark.skip(reason="Trio backend not supported; run with asyncio backend")
    for item in items:
        if "[trio]" in item.nodeid or " backend=trio" in item.nodeid or "-trio" in item.nodeid:
            item.add_marker(skip_trio)


# ============================================================================
# Test Reporter Fixture
# ============================================================================


class TestReporter:
    """Utility for reporting test results."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def test_passed(self, test_name: str, details: str = "") -> None:
        """Record a passed test."""
        self.passed += 1
        msg = f"✅ {test_name}"
        if details:
            msg += f": {details}"
        print(msg)

    def test_failed(self, test_name: str, error: str) -> None:
        """Record a failed test."""
        self.failed += 1
        msg = f"❌ {test_name}: {error}"
        self.errors.append(msg)
        print(msg)

    def print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        if self.errors:
            print("\nFailed Tests:")
            for error in self.errors:
                print(f"  {error}")
        print("=" * 80)

    def get_exit_code(self) -> int:
        """Get exit code based on test results."""
        return 0 if self.failed == 0 else 1


@pytest.fixture
def reporter() -> TestReporter:
    """Provide a test reporter instance."""
    return TestReporter()


class MockAgent:
    """Mock PydanticAI Agent for testing."""

    def __init__(self, model: str = "openai:gpt-4", usage_limits: UsageLimits | None = None):
        """Initialize mock agent."""
        self.model = model
        self.usage_limits = usage_limits
        self._tools: dict[str, Callable[..., Any]] = {}
        self._last_input: str | None = None
        self._last_output: str | None = None

    def tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Mock tool decorator."""
        self._tools[func.__name__] = func
        return func

    def run_sync(self, prompt: str, **kwargs: Any) -> str:
        """Mock synchronous run."""
        self._last_input = prompt
        self._last_output = f"Mock response to: {prompt}"
        return self._last_output

    async def run(self, prompt: str, **kwargs: Any) -> str:
        """Mock async run."""
        self._last_input = prompt
        self._last_output = f"Mock async response to: {prompt}"
        return self._last_output

    def run_stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """Mock streaming run."""
        self._last_input = prompt
        yield f"Mock stream response to: {prompt}"

    async def iter(self, prompt: str, **kwargs: Any) -> AsyncGenerator[dict[str, str], None]:
        """Mock async node iteration."""
        self._last_input = prompt
        yield {"type": "text", "content": f"Mock node response to: {prompt}"}


@pytest.fixture
def mock_agent():
    """Fixture providing a mock PydanticAI Agent."""
    return MockAgent()


@pytest.fixture
def mock_agent_with_tools():
    """Fixture providing a mock agent with tools."""
    agent = MockAgent()

    @agent.tool
    def analyze_code(code: str) -> str:
        """Mock code analysis tool."""
        return f"Analysis of: {code}"

    @agent.tool
    def refactor_code(code: str) -> str:
        """Mock code refactoring tool."""
        return f"Refactored: {code}"

    return agent


@pytest.fixture
def mock_usage_limits():
    """Fixture providing mock usage limits."""
    return UsageLimits(
        request_limit=100,
        total_tokens_limit=10000,
    )


@pytest.fixture
def mock_agent_config():
    """Fixture providing mock agent configuration."""
    return {
        "model": "openai:gpt-4",
        "enable_streaming": False,
        "enable_logging": True,
        "enable_error_recovery": True,
        "max_retries": 3,
        "timeout": 30,
    }


@pytest.fixture
def mock_agent_state():
    """Fixture providing mock agent state."""
    return {
        "message_history": [],
        "total_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0,
        },
        "error_history": [],
        "is_streaming": False,
    }


@pytest.fixture
def patch_agent_initialization(monkeypatch):
    """Fixture to patch Agent initialization to use mock."""

    def mock_init(self, model=None, *args, **kwargs):
        """Mock Agent.__init__ that accepts all arguments."""
        self.model = model or "openai:gpt-4"
        self.usage_limits = kwargs.get("usage_limits")
        self.retries = kwargs.get("retries")
        # Don't set deps_type as it's a read-only property in PydanticAI
        # Store it internally if needed
        self._deps_type = kwargs.get("deps_type")
        self.system_prompt = kwargs.get("system_prompt", "")
        self.prepare_tools = kwargs.get("prepare_tools")
        self._tools = {}
        # Accept but ignore all other kwargs to prevent errors
        return

    monkeypatch.setattr(Agent, "__init__", mock_init)
    # tool decorator should accept any arguments and return the function
    monkeypatch.setattr(Agent, "tool", lambda self, func=None, **kwargs: func if func else lambda f: f)
    monkeypatch.setattr(Agent, "run_sync", lambda self, prompt, **kw: f"Mock: {prompt}")
    monkeypatch.setattr(Agent, "run", AsyncMock(return_value="Mock async response"))
    monkeypatch.setattr(Agent, "run_stream", lambda self, prompt, **kw: iter([f"Mock stream: {prompt}"]))
    monkeypatch.setattr(Agent, "iter", AsyncMock(return_value=iter([{"type": "text"}])))


@pytest.fixture(autouse=True)
def setup_mock_agent(patch_agent_initialization):
    """Auto-use fixture to patch Agent initialization for all tests."""
    pass


@pytest.fixture
def sample_code():
    """Fixture providing sample code for testing."""
    return """
def hello_world():
    print("Hello, World!")
    return True
"""


@pytest.fixture
def sample_complex_code():
    """Fixture providing complex code for testing."""
    return """
class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.results = []

    def process(self):
        for item in self.data:
            if item > 0:
                self.results.append(item * 2)
        return self.results

    def filter_results(self, threshold):
        return [r for r in self.results if r > threshold]
"""


@pytest.fixture
def sample_pydantic_model() -> type[BaseModel]:
    """Fixture providing a sample Pydantic model."""

    class SampleModel(BaseModel):
        name: str
        value: int
        active: bool = True

    return SampleModel


@pytest.fixture
def mock_run_context() -> MagicMock:
    """Fixture providing a mock RunContext."""
    context = MagicMock(spec=RunContext)
    context.agent = MagicMock()
    context.messages = []
    context.usage = MagicMock()
    context.usage.requests = 0
    context.usage.input_tokens = 0
    context.usage.output_tokens = 0
    return context


@pytest.fixture
def mock_openai_provider() -> MagicMock:
    """Fixture providing a mock OpenAI provider."""
    provider = MagicMock()
    provider.name = "openai"
    provider.model_name = "gpt-4"
    return provider


@pytest.fixture
def mock_anthropic_provider():
    """Fixture providing a mock Anthropic provider."""
    provider = MagicMock()
    provider.name = "anthropic"
    provider.model_name = "claude-3-sonnet"
    return provider


@pytest.fixture
def mock_error_response():
    """Fixture providing a mock error response."""
    return {
        "error": "ModelHTTPError",
        "message": "API request failed",
        "status_code": 500,
        "timestamp": "2025-10-17T10:00:00Z",
    }


@pytest.fixture
def mock_success_response():
    """Fixture providing a mock success response."""
    return {
        "status": "success",
        "data": "Mock response data",
        "timestamp": "2025-10-17T10:00:00Z",
    }


@pytest.fixture
def mock_streaming_response():
    """Fixture providing a mock streaming response."""

    def stream_generator():
        yield {"type": "text", "content": "Chunk 1"}
        yield {"type": "text", "content": "Chunk 2"}
        yield {"type": "text", "content": "Chunk 3"}

    return stream_generator()


@pytest.fixture
def mock_async_streaming_response():
    """Fixture providing a mock async streaming response."""

    async def async_stream_generator():
        yield {"type": "text", "content": "Async Chunk 1"}
        yield {"type": "text", "content": "Async Chunk 2"}
        yield {"type": "text", "content": "Async Chunk 3"}

    return async_stream_generator()


@pytest.fixture
def mock_tool_result():
    """Fixture providing a mock tool result."""
    return {
        "tool_name": "analyze_code",
        "status": "success",
        "result": "Analysis complete",
        "execution_time": 0.5,
    }


@pytest.fixture
def mock_message_history():
    """Fixture providing a mock message history."""
    return [
        {"role": "user", "content": "Analyze this code"},
        {"role": "assistant", "content": "Analysis result"},
        {"role": "user", "content": "Refactor it"},
        {"role": "assistant", "content": "Refactored code"},
    ]


@pytest.fixture
def mock_usage_data():
    """Fixture providing mock usage data."""
    return {
        "input_tokens": 1000,
        "output_tokens": 500,
        "requests": 5,
        "total_tokens": 1500,
    }


@pytest.fixture
def mock_error_history():
    """Fixture providing mock error history."""
    return [
        {
            "error_type": "ModelHTTPError",
            "message": "API error",
            "timestamp": "2025-10-17T10:00:00Z",
        },
        {
            "error_type": "UsageLimitExceeded",
            "message": "Token limit exceeded",
            "timestamp": "2025-10-17T10:01:00Z",
        },
    ]


# ============================================================================
# Code Execution Fixtures
# ============================================================================


@pytest.fixture
def valid_code():
    """Fixture providing valid Python code for execution tests."""
    return """
result = 2 + 2
print(f"Result: {result}")
"""


@pytest.fixture
def invalid_syntax_code():
    """Fixture providing code with syntax errors."""
    return """
def broken_function(
    print("Missing closing parenthesis")
"""


@pytest.fixture
def dangerous_code():
    """Fixture providing code with security issues."""
    return """
import os
os.system('ls -la')
"""


@pytest.fixture
def code_with_eval():
    """Fixture providing code using eval (dangerous)."""
    return """
user_input = "print('hello')"
eval(user_input)
"""


@pytest.fixture
def code_with_loops():
    """Fixture providing code with loops."""
    return """
for i in range(10):
    print(i)
"""


@pytest.fixture
def complex_code():
    """Fixture providing complex code for complexity testing."""
    return """
def complex_function(a, b, c, d, e):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0
"""


@pytest.fixture
def mock_execution_context():
    """Fixture providing a mock execution context."""
    from code_agent.core.types import ExecutionContext, ExecutionMode

    return ExecutionContext(
        mode=ExecutionMode.SAFE,
        timeout=30.0,
        blocked_imports=["os", "sys", "subprocess"],
        blocked_builtins=["eval", "exec", "compile"],
    )


@pytest.fixture
def mock_validation_config():
    """Fixture providing a mock validation configuration."""
    from code_agent.config.execution import ValidationConfig

    return ValidationConfig(
        enable_syntax_check=True,
        enable_security_check=True,
        enable_import_check=True,
        enable_complexity_check=False,
    )


@pytest.fixture
def mock_execution_config():
    """Fixture providing a mock execution configuration."""
    from code_agent.config.execution import create_safe_config

    return create_safe_config()


@pytest.fixture
def mock_execution_result():
    """Fixture providing a mock execution result."""
    from code_agent.core.types import ExecutionResult, ExecutionStatus

    return ExecutionResult(
        code="print('test')",
        status=ExecutionStatus.COMPLETED,
        output="test\n",
        error="",
        exit_code=0,
        execution_time=0.1,
    )
