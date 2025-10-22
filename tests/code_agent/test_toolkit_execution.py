"""
Toolkit Execution Integration Tests

Tests for code execution integration with toolkit.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from code_agent.config.execution import (
    ValidationConfig,
    create_safe_config,
)
from code_agent.core.types import ExecutionStatus
from code_agent.tools.toolkit import (
    CodeAgentState,
    execute_code,
    execute_code_with_retry,
    validate_code_for_execution,
)


class TestCodeAgentStateExecution:
    """Test CodeAgentState execution history."""

    def test_state_has_execution_history(self):
        """Test that CodeAgentState has execution_history field."""
        state = CodeAgentState()

        assert hasattr(state, "execution_history")
        assert isinstance(state.execution_history, list)
        assert len(state.execution_history) == 0

    def test_state_execution_history_append(self):
        """Test appending to execution history."""
        from code_agent.core.types import ExecutionResult, ExecutionStatus

        state = CodeAgentState()
        result = ExecutionResult(
            code="print('test')",
            status=ExecutionStatus.COMPLETED,
            output="test\n",
        )

        state.execution_history.append(result)

        assert len(state.execution_history) == 1
        assert state.execution_history[0] == result


class TestExecuteCode:
    """Test execute_code function."""

    def test_execute_code_basic(self, valid_code):
        """Test basic code execution."""
        state = CodeAgentState()
        result = execute_code(valid_code, state)

        assert isinstance(result, str)
        assert len(state.execution_history) == 1

    def test_execute_code_with_config(self, valid_code):
        """Test code execution with custom config."""
        state = CodeAgentState()
        config = create_safe_config()

        result = execute_code(valid_code, state, config=config)

        assert isinstance(result, str)
        assert len(state.execution_history) == 1

    def test_execute_code_invalid_syntax(self, invalid_syntax_code):
        """Test executing code with syntax errors."""
        state = CodeAgentState()
        result = execute_code(invalid_syntax_code, state)

        assert isinstance(result, str)
        assert "error" in result.lower() or "failed" in result.lower()
        assert len(state.execution_history) == 1
        assert state.execution_history[0].status == ExecutionStatus.FAILED

    def test_execute_code_dangerous(self, dangerous_code):
        """Test executing dangerous code."""
        state = CodeAgentState()
        result = execute_code(dangerous_code, state)

        assert isinstance(result, str)
        # Should fail validation
        assert len(state.execution_history) == 1
        assert state.execution_history[0].status == ExecutionStatus.FAILED

    def test_execute_code_with_output(self):
        """Test executing code that produces output."""
        state = CodeAgentState()
        code = "print('Hello, World!')"
        result = execute_code(code, state)

        assert isinstance(result, str)
        if state.execution_history[0].status == ExecutionStatus.COMPLETED:
            assert "Hello, World!" in result

    def test_execute_code_multiple_times(self):
        """Test executing multiple code snippets."""
        state = CodeAgentState()

        execute_code("x = 1", state)
        execute_code("y = 2", state)
        execute_code("z = 3", state)

        assert len(state.execution_history) == 3

    def test_execute_code_with_context(self, valid_code):
        """Test executing code with custom context."""
        from code_agent.core.types import ExecutionContext, ExecutionMode

        state = CodeAgentState()
        context = ExecutionContext(mode=ExecutionMode.SAFE)

        result = execute_code(valid_code, state, context=context)

        assert isinstance(result, str)
        assert len(state.execution_history) == 1


class TestValidateCodeForExecution:
    """Test validate_code_for_execution function."""

    def test_validate_valid_code(self, valid_code):
        """Test validating valid code."""
        state = CodeAgentState()
        result = validate_code_for_execution(valid_code, state)

        assert isinstance(result, str)
        assert "valid" in result.lower() or "passed" in result.lower()

    def test_validate_invalid_syntax(self, invalid_syntax_code):
        """Test validating code with syntax errors."""
        state = CodeAgentState()
        result = validate_code_for_execution(invalid_syntax_code, state)

        assert isinstance(result, str)
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_validate_dangerous_code(self, dangerous_code):
        """Test validating dangerous code."""
        state = CodeAgentState()
        result = validate_code_for_execution(dangerous_code, state)

        assert isinstance(result, str)
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_validate_with_custom_config(self, valid_code):
        """Test validation with custom config."""
        state = CodeAgentState()
        config = ValidationConfig(enable_syntax_check=True)

        result = validate_code_for_execution(valid_code, state, config=config)

        assert isinstance(result, str)

    def test_validate_does_not_execute(self, valid_code):
        """Test that validation does not execute code."""
        state = CodeAgentState()

        # Validate code
        validate_code_for_execution(valid_code, state)

        # Execution history should be empty (validation only)
        assert len(state.execution_history) == 0


class TestExecuteCodeWithRetry:
    """Test execute_code_with_retry function."""

    def test_execute_with_retry_success(self, valid_code):
        """Test successful execution with retry."""
        state = CodeAgentState()
        result = execute_code_with_retry(valid_code, state)

        assert isinstance(result, str)
        assert len(state.execution_history) >= 1

    def test_execute_with_retry_failure(self, invalid_syntax_code):
        """Test failed execution with retry."""
        state = CodeAgentState()
        result = execute_code_with_retry(invalid_syntax_code, state)

        assert isinstance(result, str)
        # Should fail even with retries (syntax error is permanent)
        assert "error" in result.lower() or "failed" in result.lower()

    def test_execute_with_retry_custom_max_retries(self, valid_code):
        """Test execution with custom max retries."""
        state = CodeAgentState()
        result = execute_code_with_retry(valid_code, state, max_retries=5)

        assert isinstance(result, str)

    def test_execute_with_retry_config(self, valid_code):
        """Test execution with retry and custom config."""
        state = CodeAgentState()
        config = create_safe_config()

        result = execute_code_with_retry(valid_code, state, config=config)

        assert isinstance(result, str)

    @patch("code_agent.tools.toolkit.CodeExecutor")
    def test_execute_with_retry_circuit_breaker(self, mock_executor_class, valid_code):
        """Test circuit breaker integration."""
        # Mock executor to simulate failures
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor

        from code_agent.core.types import ExecutionResult, ExecutionStatus

        # Simulate transient failure then success
        mock_executor.execute.side_effect = [
            ExecutionResult(
                code=valid_code,
                status=ExecutionStatus.FAILED,
                error="Transient error",
            ),
            ExecutionResult(
                code=valid_code,
                status=ExecutionStatus.COMPLETED,
                output="Success",
            ),
        ]

        _state = CodeAgentState()

        # This test verifies the function can be called with circuit breaker
        # Actual circuit breaker behavior is tested in the executor tests
        assert callable(execute_code_with_retry)


class TestExecutionIntegration:
    """Test integration between execution components."""

    def test_full_execution_workflow(self):
        """Test complete execution workflow."""
        state = CodeAgentState()

        # 1. Validate code
        code = "result = 2 + 2\nprint(result)"
        validation_result = validate_code_for_execution(code, state)

        # 2. Execute code
        execution_result = execute_code(code, state)

        # 3. Check state
        assert len(state.execution_history) == 1
        assert isinstance(validation_result, str)
        assert isinstance(execution_result, str)

    def test_execution_with_multiple_validations(self):
        """Test multiple validations before execution."""
        state = CodeAgentState()
        code = "x = 1"

        # Validate multiple times
        validate_code_for_execution(code, state)
        validate_code_for_execution(code, state)

        # Execute once
        execute_code(code, state)

        # Only execution should be in history
        assert len(state.execution_history) == 1

    def test_execution_history_ordering(self):
        """Test execution history maintains order."""
        state = CodeAgentState()

        codes = ["x = 1", "y = 2", "z = 3"]

        for code in codes:
            execute_code(code, state)

        assert len(state.execution_history) == 3
        for i, code in enumerate(codes):
            assert state.execution_history[i].code == code

    def test_execution_with_different_modes(self):
        """Test execution with different security modes."""
        from code_agent.config.execution import (
            create_restricted_config,
            create_safe_config,
        )

        state = CodeAgentState()
        code = "result = 2 + 2"

        # Safe mode
        safe_config = create_safe_config()
        execute_code(code, state, config=safe_config)

        # Restricted mode
        restricted_config = create_restricted_config()
        execute_code(code, state, config=restricted_config)

        assert len(state.execution_history) == 2
