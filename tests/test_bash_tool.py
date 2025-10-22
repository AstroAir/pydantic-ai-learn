"""
Unit tests for BashTool

Tests cover:
- Input validation
- Synchronous execution
- Asynchronous execution
- Timeout handling
- Error handling
- Background execution
- Output truncation
- Session persistence
- Context managers

Run with: pytest tests/test_bash_tool.py -v
"""

import asyncio
import os
import sys

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.bash_tool import (
    BashCommandInput,
    BashTimeoutError,
    BashTool,
    run_bash_command,
    run_bash_command_async,
)

# ============================================================================
# Input Validation Tests
# ============================================================================


class TestBashCommandInput:
    """Test Pydantic input validation."""

    def test_valid_input_minimal(self):
        """Test minimal valid input."""
        cmd = BashCommandInput(command="echo 'test'")
        assert cmd.command == "echo 'test'"
        assert cmd.timeout == 120000  # Default
        assert cmd.description is None
        assert cmd.run_in_background is False

    def test_valid_input_full(self):
        """Test full valid input."""
        cmd = BashCommandInput(command="ls -la", timeout=5000, description="List files", run_in_background=True)
        assert cmd.command == "ls -la"
        assert cmd.timeout == 5000
        assert cmd.description == "List files"
        assert cmd.run_in_background is True

    def test_timeout_validation_too_large(self):
        """Test timeout exceeds maximum."""
        with pytest.raises(ValueError, match="cannot exceed 600000"):
            BashCommandInput(command="test", timeout=700000)

    def test_timeout_validation_zero(self):
        """Test timeout is zero."""
        with pytest.raises(ValueError, match="greater than 0"):
            BashCommandInput(command="test", timeout=0)

    def test_timeout_validation_negative(self):
        """Test timeout is negative."""
        with pytest.raises(ValueError, match="greater than 0"):
            BashCommandInput(command="test", timeout=-1000)

    def test_empty_command(self):
        """Test empty command fails validation."""
        with pytest.raises(ValueError):
            BashCommandInput(command="")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            BashCommandInput(command="test", extra_field="not allowed")


# ============================================================================
# Synchronous Execution Tests
# ============================================================================


class TestSyncExecution:
    """Test synchronous command execution."""

    def test_simple_command(self):
        """Test simple echo command."""
        with BashTool() as bash:
            result = bash.run_command("echo 'hello'")
            assert result.success
            assert "hello" in result.output
            assert result.exit_code == 0
            assert not result.timed_out
            assert not result.truncated

    def test_command_with_exit_code(self):
        """Test command that returns non-zero exit code."""
        with BashTool() as bash:
            result = bash.run_command("exit 42")
            assert not result.success
            assert result.exit_code == 42

    def test_command_with_pipes(self):
        """Test command with pipes."""
        with BashTool() as bash:
            result = bash.run_command("echo 'test' | tr 'a-z' 'A-Z'")
            assert result.success
            assert "TEST" in result.output

    def test_persistent_environment(self):
        """Test environment variables persist across commands."""
        with BashTool() as bash:
            bash.run_command("export TEST_VAR='persistent'")
            result = bash.run_command("echo $TEST_VAR")
            assert "persistent" in result.output

    def test_persistent_directory(self):
        """Test working directory persists across commands."""
        with BashTool() as bash:
            bash.run_command("cd /tmp")
            result = bash.run_command("pwd")
            assert "/tmp" in result.output

    def test_reset_clears_state(self):
        """Test reset clears session state."""
        with BashTool() as bash:
            bash.run_command("export TEST_VAR='value'")
            bash.reset()
            result = bash.run_command("echo $TEST_VAR")
            # Should be empty after reset
            assert result.output.strip() == ""

    def test_context_manager_cleanup(self):
        """Test context manager properly cleans up."""
        bash = BashTool()
        with bash:
            assert bash.is_alive()
        # After exiting context, process should be cleaned up
        assert not bash.is_alive()


# ============================================================================
# Asynchronous Execution Tests
# ============================================================================


class TestAsyncExecution:
    """Test asynchronous command execution."""

    @pytest.mark.asyncio
    async def test_simple_async_command(self):
        """Test simple async command."""
        async with BashTool() as bash:
            result = await bash.run_command_async("echo 'async test'")
            assert result.success
            assert "async test" in result.output

    @pytest.mark.asyncio
    async def test_concurrent_commands(self):
        """Test multiple concurrent commands."""
        async with BashTool() as bash:
            tasks = [bash.run_command_async(f"echo 'task {i}'") for i in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            for i, result in enumerate(results):
                assert result.success
                assert f"task {i}" in result.output

    @pytest.mark.asyncio
    async def test_background_execution(self):
        """Test background execution."""
        async with BashTool() as bash:
            # Start background task
            task = await bash.run_command_async("sleep 1 && echo 'done'", background=True)

            # Task should be a Task object
            assert isinstance(task, asyncio.Task)

            # Wait for completion
            result = await task
            assert result.success
            assert "done" in result.output


# ============================================================================
# Timeout Tests
# ============================================================================


class TestTimeout:
    """Test timeout handling."""

    def test_sync_timeout(self):
        """Test synchronous timeout."""
        with BashTool() as bash, pytest.raises(BashTimeoutError):
            bash.run_command("sleep 10", timeout_ms=1000)

    @pytest.mark.asyncio
    async def test_async_timeout(self):
        """Test asynchronous timeout."""
        async with BashTool() as bash:
            with pytest.raises(BashTimeoutError):
                await bash.run_command_async("sleep 10", timeout_ms=1000)

    def test_command_completes_before_timeout(self):
        """Test command that completes before timeout."""
        with BashTool() as bash:
            result = bash.run_command("sleep 0.1", timeout_ms=5000)
            assert result.success
            assert not result.timed_out


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_command_failure(self):
        """Test handling of failed commands."""
        with BashTool() as bash:
            result = bash.run_command("ls /nonexistent/path")
            assert not result.success
            assert result.exit_code != 0

    def test_background_in_sync_context_raises(self):
        """Test that background execution in sync context raises error."""
        with BashTool() as bash, pytest.raises(RuntimeError, match="Background execution requires async"):
            bash.execute(BashCommandInput(command="echo 'test'", run_in_background=True))


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_run_bash_command(self):
        """Test run_bash_command convenience function."""
        result = run_bash_command("echo 'convenience'")
        assert result.success
        assert "convenience" in result.output

    @pytest.mark.asyncio
    async def test_run_bash_command_async(self):
        """Test run_bash_command_async convenience function."""
        result = await run_bash_command_async("echo 'async convenience'")
        assert result.success
        assert "async convenience" in result.output


# ============================================================================
# Output Tests
# ============================================================================


class TestOutput:
    """Test output handling."""

    def test_output_capture(self):
        """Test stdout is captured."""
        with BashTool() as bash:
            result = bash.run_command("echo 'captured output'")
            assert "captured output" in result.output

    def test_stderr_merged(self):
        """Test stderr is merged with stdout."""
        with BashTool() as bash:
            result = bash.run_command("echo 'error' >&2")
            assert "error" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
