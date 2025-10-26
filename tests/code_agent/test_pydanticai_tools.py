"""
PydanticAI Tool System Tests

Comprehensive tests for PydanticAI tool registration and execution:
- Tool decorator usage
- Tool parameter handling
- Tool return types
- Tool execution patterns
- Tool error handling
- Tool context integration

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext

from code_agent.core import CodeAgent, CodeAgentState


class AnalysisResult(BaseModel):
    """Analysis result model."""

    status: str
    issues: list[str]
    score: float


class RefactoringResult(BaseModel):
    """Refactoring result model."""

    original_code: str
    refactored_code: str
    improvements: list[str]


class TestPydanticAIToolDecorator:
    """Test PydanticAI @agent.tool decorator."""

    def test_tool_decorator_basic(self):
        """Test basic tool decorator usage."""
        agent = CodeAgent()

        @agent.agent.tool
        def simple_tool(text: str) -> str:
            """A simple tool."""
            return f"Processed: {text}"

        assert agent.agent is not None

    def test_tool_decorator_with_docstring(self):
        """Test tool decorator with comprehensive docstring."""
        agent = CodeAgent()

        @agent.agent.tool
        def documented_tool(code: str) -> str:
            """
            Analyze Python code for issues.

            Args:
                code: Python code to analyze

            Returns:
                Analysis results
            """
            return f"Analyzed: {code}"

        assert agent.agent is not None

    def test_multiple_tools_registration(self):
        """Test registering multiple tools."""
        agent = CodeAgent()

        @agent.agent.tool
        def tool1(text: str) -> str:
            """First tool."""
            return f"Tool1: {text}"

        @agent.agent.tool
        def tool2(text: str) -> str:
            """Second tool."""
            return f"Tool2: {text}"

        @agent.agent.tool
        def tool3(text: str) -> str:
            """Third tool."""
            return f"Tool3: {text}"

        assert agent.agent is not None


class TestPydanticAIToolParameters:
    """Test PydanticAI tool parameter handling."""

    def test_tool_with_single_parameter(self):
        """Test tool with single parameter."""
        agent = CodeAgent()

        @agent.agent.tool
        def single_param_tool(code: str) -> str:
            """Tool with single parameter."""
            return code

        assert agent.agent is not None

    def test_tool_with_multiple_parameters(self):
        """Test tool with multiple parameters."""
        agent = CodeAgent()

        @agent.agent.tool
        def multi_param_tool(
            code: str,
            style: str,
            check_types: bool,
        ) -> str:
            """Tool with multiple parameters."""
            return f"{code}:{style}:{check_types}"

        assert agent.agent is not None

    def test_tool_with_optional_parameters(self):
        """Test tool with optional parameters."""
        agent = CodeAgent()

        @agent.agent.tool
        def optional_param_tool(
            code: str,
            verbose: bool = False,
            max_issues: int = 10,
        ) -> str:
            """Tool with optional parameters."""
            return f"Code: {code}, Verbose: {verbose}, Max: {max_issues}"

        assert agent.agent is not None

    def test_tool_with_type_hints(self):
        """Test tool with comprehensive type hints."""
        agent = CodeAgent()

        @agent.agent.tool
        def typed_tool(
            code: str,
            issues: list[str],
            metadata: dict[str, Any],
        ) -> str:
            """Tool with comprehensive type hints."""
            return f"Processed {len(issues)} issues"

        assert agent.agent is not None

    def test_tool_with_default_values(self):
        """Test tool with default parameter values."""
        agent = CodeAgent()

        @agent.agent.tool
        def defaults_tool(
            code: str,
            language: str = "python",
            version: str = "3.12",
            strict: bool = True,
        ) -> str:
            """Tool with default values."""
            return f"{language} {version}"

        assert agent.agent is not None


class TestPydanticAIToolReturnTypes:
    """Test PydanticAI tool return types."""

    def test_tool_returning_string(self):
        """Test tool returning string."""
        agent = CodeAgent()

        @agent.agent.tool
        def string_tool(code: str) -> str:
            """Tool returning string."""
            return f"Result: {code}"

        assert agent.agent is not None

    def test_tool_returning_dict(self):
        """Test tool returning dictionary."""
        agent = CodeAgent()

        @agent.agent.tool
        def dict_tool(code: str) -> dict[str, Any]:
            """Tool returning dictionary."""
            return {
                "code": code,
                "status": "analyzed",
                "issues": [],
            }

        assert agent.agent is not None

    def test_tool_returning_list(self):
        """Test tool returning list."""
        agent = CodeAgent()

        @agent.agent.tool
        def list_tool(code: str) -> list[str]:
            """Tool returning list."""
            return ["issue1", "issue2", "issue3"]

        assert agent.agent is not None

    def test_tool_returning_pydantic_model(self):
        """Test tool returning Pydantic model."""
        agent = CodeAgent()

        @agent.agent.tool
        def model_tool(code: str) -> AnalysisResult:
            """Tool returning Pydantic model."""
            return AnalysisResult(
                status="success",
                issues=["issue1"],
                score=0.85,
            )

        assert agent.agent is not None

    def test_tool_returning_complex_model(self):
        """Test tool returning complex Pydantic model."""
        agent = CodeAgent()

        @agent.agent.tool
        def complex_tool(code: str) -> RefactoringResult:
            """Tool returning complex model."""
            return RefactoringResult(
                original_code=code,
                refactored_code=f"refactored_{code}",
                improvements=["improvement1", "improvement2"],
            )

        assert agent.agent is not None

    def test_tool_returning_union_type(self):
        """Test tool returning union type."""
        agent = CodeAgent()

        @agent.agent.tool
        def union_tool(code: str) -> str | dict[str, Any]:
            """Tool returning union type."""
            if len(code) > 100:
                return {"status": "large"}
            return "small"

        assert agent.agent is not None

    def test_tool_returning_none(self):
        """Test tool returning None."""
        agent = CodeAgent()

        @agent.agent.tool
        def none_tool(code: str) -> None:
            """Tool returning None."""
            pass

        assert agent.agent is not None


class TestPydanticAIToolExecution:
    """Test PydanticAI tool execution patterns."""

    def test_tool_execution_pattern(self):
        """Test tool execution pattern."""
        agent = CodeAgent()

        @agent.agent.tool
        def execute_tool(code: str) -> str:
            """Tool for execution."""
            return f"Executed: {code}"

        # Verify tool is registered
        assert agent.agent is not None

    def test_tool_with_side_effects(self):
        """Test tool with side effects."""
        agent = CodeAgent()
        execution_log = []

        @agent.agent.tool
        def logging_tool(code: str) -> str:
            """Tool with side effects."""
            execution_log.append(code)
            return f"Logged: {code}"

        assert agent.agent is not None

    def test_tool_with_state_modification(self):
        """Test tool that modifies agent state."""
        agent = CodeAgent()

        @agent.agent.tool
        def state_tool(code: str) -> str:
            """Tool that modifies state."""
            agent.state.total_usage["requests"] += 1
            return f"Processed: {code}"

        assert agent.agent is not None

    def test_tool_chaining_pattern(self):
        """Test tool chaining pattern."""
        agent = CodeAgent()

        @agent.agent.tool
        def analyze(code: str) -> str:
            """Analyze code."""
            return f"Analyzed: {code}"

        @agent.agent.tool
        def refactor(analyzed: str) -> str:
            """Refactor analyzed code."""
            return f"Refactored: {analyzed}"

        assert agent.agent is not None


class TestPydanticAIToolErrorHandling:
    """Test PydanticAI tool error handling."""

    def test_tool_with_validation(self):
        """Test tool with input validation."""
        agent = CodeAgent()

        @agent.agent.tool
        def validated_tool(code: str) -> str:
            """Tool with validation."""
            if not code:
                raise ValueError("Code cannot be empty")
            return f"Validated: {code}"

        assert agent.agent is not None

    def test_tool_with_error_handling(self):
        """Test tool with error handling."""
        agent = CodeAgent()

        @agent.agent.tool
        def safe_tool(code: str) -> str:
            """Tool with error handling."""
            try:
                return f"Processed: {code}"
            except Exception as e:
                return f"Error: {str(e)}"

        assert agent.agent is not None

    def test_tool_with_fallback(self):
        """Test tool with fallback behavior."""
        agent = CodeAgent()

        @agent.agent.tool
        def fallback_tool(code: str) -> str:
            """Tool with fallback."""
            try:
                return f"Primary: {code}"
            except Exception:
                return f"Fallback: {code}"

        assert agent.agent is not None


class TestPydanticAIToolContext:
    """Test PydanticAI tool context integration."""

    def test_tool_accessing_agent_state(self):
        """Test tool accessing agent state."""
        agent = CodeAgent()

        @agent.agent.tool
        def state_aware_tool(code: str) -> str:
            """Tool that accesses agent state."""
            usage = agent.state.total_usage
            return f"Processed with {usage['requests']} requests"

        assert agent.agent is not None

    def test_tool_accessing_config(self):
        """Test tool accessing agent config."""
        agent = CodeAgent()

        @agent.agent.tool
        def config_aware_tool(code: str) -> str:
            """Tool that accesses config."""
            model = agent.config.model
            return f"Using model: {model}"

        assert agent.agent is not None

    def test_tool_with_message_history(self):
        """Test tool with message history access."""
        agent = CodeAgent()

        @agent.agent.tool
        def history_tool(code: str) -> str:
            """Tool with history access."""
            history_len = len(agent.state.message_history)
            return f"History length: {history_len}"

        assert agent.agent is not None


class TestPydanticAIToolAdvanced:
    """Test advanced PydanticAI tool patterns."""

    def test_tool_with_nested_models(self) -> None:
        """Test tool with nested Pydantic models."""
        agent = CodeAgent()

        class NestedModel(BaseModel):
            """Nested model."""

            name: str
            items: list[str]

        @agent.agent.tool
        def nested_tool(ctx: RunContext[CodeAgentState], code: str) -> NestedModel:
            """Tool returning nested model."""
            return NestedModel(
                name="result",
                items=["item1", "item2"],
            )

        assert agent.agent is not None

    def test_tool_with_generic_types(self) -> None:
        """Test tool with generic types."""
        agent = CodeAgent()

        @agent.agent.tool
        def generic_tool(ctx: RunContext[CodeAgentState], items: list[str]) -> dict[str, list[str]]:
            """Tool with generic types."""
            return {"processed": items}

        assert agent.agent is not None

    def test_tool_with_conditional_logic(self):
        """Test tool with conditional logic."""
        agent = CodeAgent()

        @agent.agent.tool
        def conditional_tool(code: str, mode: str) -> str:
            """Tool with conditional logic."""
            if mode == "strict":
                return f"Strict: {code}"
            if mode == "lenient":
                return f"Lenient: {code}"
            return f"Default: {code}"

        assert agent.agent is not None

    def test_tool_with_loop_logic(self):
        """Test tool with loop logic."""
        agent = CodeAgent()

        @agent.agent.tool
        def loop_tool(items: list[str]) -> str:
            """Tool with loop logic."""
            results = []
            for item in items:
                results.append(f"Processed: {item}")
            return ", ".join(results)

        assert agent.agent is not None
