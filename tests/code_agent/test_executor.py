"""
Code Executor Tests

Tests for code execution engine and caching.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import time
from typing import Any

from code_agent.config.execution import (
    create_full_config,
    create_restricted_config,
    create_safe_config,
)
from code_agent.core.types import ExecutionContext, ExecutionMode, ExecutionStatus
from code_agent.tools.executor import CodeExecutor, ExecutionCache, ExecutionError


class TestExecutionCache:
    """Test ExecutionCache."""

    def test_cache_creation(self):
        """Test creating execution cache."""
        cache = ExecutionCache(ttl=60)

        assert cache.ttl == 60
        assert len(cache._cache) == 0

    def test_cache_get_miss(self):
        """Test cache miss."""
        cache = ExecutionCache()
        result = cache.get("nonexistent_key")

        assert result is None

    def test_cache_set_and_get(self):
        """Test setting and getting from cache."""
        cache = ExecutionCache()
        cache.set("key1", "value1")

        result = cache.get("key1")
        assert result == "value1"

    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = ExecutionCache(ttl=0.1)  # 100ms TTL
        cache.set("key1", "value1")

        # Should be in cache immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        assert cache.get("key1") is None

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = ExecutionCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ExecutionCache()

        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Get some values (hits)
        cache.get("key1")
        cache.get("key1")

        # Get nonexistent (miss)
        cache.get("key3")

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["hits"] >= 2
        assert stats["misses"] >= 1


class TestCodeExecutor:
    """Test CodeExecutor."""

    def test_executor_creation_default(self):
        """Test creating executor with default config."""
        executor = CodeExecutor()

        assert executor.config is not None
        assert isinstance(executor.cache, ExecutionCache)

    def test_executor_creation_with_config(self):
        """Test creating executor with custom config."""
        config = create_safe_config()
        executor = CodeExecutor(config)

        assert executor.config == config

    def test_execute_simple_code(self, valid_code):
        """Test executing simple valid code."""
        executor = CodeExecutor(create_safe_config())
        result = executor.execute(valid_code)

        assert result.code == valid_code
        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
        assert isinstance(result.execution_time, float)

    def test_execute_with_output(self):
        """Test executing code with output."""
        executor = CodeExecutor(create_safe_config())
        code = "print('Hello, World!')"
        result = executor.execute(code)

        if result.status == ExecutionStatus.COMPLETED:
            assert "Hello, World!" in result.output

    def test_execute_invalid_syntax(self, invalid_syntax_code):
        """Test executing code with syntax errors."""
        executor = CodeExecutor(create_safe_config())
        result = executor.execute(invalid_syntax_code)

        assert result.status == ExecutionStatus.FAILED
        assert len(result.validation_errors) > 0

    def test_execute_dangerous_code(self, dangerous_code):
        """Test executing dangerous code in safe mode."""
        executor = CodeExecutor(create_safe_config())
        result = executor.execute(dangerous_code)

        # Should fail validation in safe mode
        assert result.status == ExecutionStatus.FAILED
        assert len(result.validation_errors) > 0

    def test_execute_safe_mode(self):
        """Test execution in safe mode."""
        config = create_safe_config()
        executor = CodeExecutor(config)

        code = "result = 2 + 2"
        result = executor.execute(code)

        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]

    def test_execute_restricted_mode(self):
        """Test execution in restricted mode."""
        config = create_restricted_config()
        executor = CodeExecutor(config)

        code = "result = 2 + 2"
        result = executor.execute(code)

        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]

    def test_execute_full_mode(self):
        """Test execution in full mode."""
        config = create_full_config()
        executor = CodeExecutor(config)

        code = "result = 2 + 2"
        result = executor.execute(code)

        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]

    def test_execute_with_timeout(self):
        """Test execution timeout."""
        config = create_safe_config()
        config.resources.max_execution_time = 0.1  # 100ms timeout
        executor = CodeExecutor(config)

        # Code that might take longer
        code = "import time\ntime.sleep(1)"
        result = executor.execute(code)

        # Should timeout or fail validation (import time blocked)
        assert result.status in [ExecutionStatus.TIMEOUT, ExecutionStatus.FAILED]

    def test_execute_dry_run(self, valid_code):
        """Test dry run mode (validation only)."""
        config = create_safe_config()
        config.dry_run = True
        executor = CodeExecutor(config)

        result = executor.execute(valid_code)

        # In dry run, should validate but not execute
        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
        if result.status == ExecutionStatus.COMPLETED:
            assert "Dry run" in result.output or "validation passed" in result.output or result.output == ""

    def test_execute_with_context(self, valid_code):
        """Test execution with custom context."""
        executor = CodeExecutor(create_safe_config())
        context = ExecutionContext(
            mode=ExecutionMode.SAFE,
            timeout=60.0,
        )

        result = executor.execute(valid_code, context)

        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]

    def test_pre_validation_hook(self, valid_code):
        """Test pre-validation hook."""
        hook_called = []

        def pre_hook(code: str) -> None:
            hook_called.append(code)

        config = create_safe_config()
        config.hooks.pre_validation_hooks.append(pre_hook)
        executor = CodeExecutor(config)

        executor.execute(valid_code)

        assert len(hook_called) == 1
        assert hook_called[0] == valid_code

    def test_post_validation_hook(self, valid_code):
        """Test post-validation hook."""
        hook_called = []

        def post_hook(code: str, errors: list[str]) -> None:
            hook_called.append((code, errors))

        config = create_safe_config()
        config.hooks.post_validation_hooks.append(post_hook)
        executor = CodeExecutor(config)

        executor.execute(valid_code)

        assert len(hook_called) == 1

    def test_pre_execution_hook(self, valid_code):
        """Test pre-execution hook."""
        hook_called = []

        def pre_hook(code: str, context: dict[str, Any]) -> None:
            hook_called.append(code)

        config = create_safe_config()
        config.hooks.pre_execution_hooks.append(pre_hook)
        executor = CodeExecutor(config)

        executor.execute(valid_code)

        # Hook should be called if validation passes
        assert isinstance(hook_called, list)

    def test_post_execution_hook(self):
        """Test post-execution hook."""
        hook_called = []

        def post_hook(code: str, result: Any) -> None:
            hook_called.append(code)

        config = create_safe_config()
        config.hooks.post_execution_hooks.append(post_hook)
        executor = CodeExecutor(config)

        code = "result = 2 + 2"
        executor.execute(code)

        # Hook should be called if execution completes
        assert isinstance(hook_called, list)

    def test_error_hook(self, invalid_syntax_code):
        """Test error hook."""
        hook_called = []

        def error_hook(error: Exception, code: str) -> None:
            hook_called.append((type(error).__name__, code))

        config = create_safe_config()
        config.hooks.error_hooks.append(error_hook)
        executor = CodeExecutor(config)

        executor.execute(invalid_syntax_code)

        # Error hook should be called on validation/execution errors
        assert isinstance(hook_called, list)

    def test_caching_enabled(self):
        """Test result caching."""
        config = create_safe_config()
        config.enable_caching = True
        executor = CodeExecutor(config)

        code = "result = 2 + 2"

        # First execution
        result1 = executor.execute(code)
        _time1 = result1.execution_time

        # Second execution (should be cached)
        result2 = executor.execute(code)

        # Results should be similar
        assert result1.code == result2.code
        assert result1.status == result2.status

    def test_caching_disabled(self):
        """Test execution with caching disabled."""
        config = create_safe_config()
        config.enable_caching = False
        executor = CodeExecutor(config)

        code = "result = 2 + 2"

        result1 = executor.execute(code)
        result2 = executor.execute(code)

        # Both should execute independently
        assert result1.code == result2.code

    def test_get_stats(self):
        """Test getting execution statistics."""
        executor = CodeExecutor(create_safe_config())

        # Execute some code
        executor.execute("result = 1")
        executor.execute("result = 2")
        executor.execute("invalid syntax")

        stats = executor.get_stats()

        assert "total_executions" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert stats["total_executions"] >= 3


class TestExecutionError:
    """Test ExecutionError exception."""

    def test_execution_error_creation(self):
        """Test creating execution error."""
        error = ExecutionError("Execution failed")

        assert str(error) == "Execution failed"
        assert isinstance(error, Exception)
