"""
Code Execution Examples

Demonstrates the code execution features of the code agent including:
- Safe code execution with validation
- Custom hooks and callbacks
- Execution modes (safe, restricted, full)
- Error handling and recovery
- Output formatting
- Resource limits

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from code_agent.config.execution import (
    create_restricted_config,
    create_safe_config,
)
from code_agent.tools.executor import CodeExecutor
from code_agent.tools.validators import ExecutionValidator


def example_basic_execution() -> None:
    """Example: Basic code execution with safe mode."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Safe Execution")
    print("=" * 60)

    # Create executor with safe configuration
    config = create_safe_config()
    executor = CodeExecutor(config)

    # Execute simple code
    code = """
result = sum(range(10))
print(f"Sum of 0-9: {result}")
"""

    result = executor.execute(code)

    print(f"Status: {result.status.value}")
    print(f"Output: {result.output}")
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"Success: {result.is_success()}")


def example_validation() -> None:
    """Example: Pre-execution validation."""
    print("\n" + "=" * 60)
    print("Example 2: Code Validation")
    print("=" * 60)

    validator = ExecutionValidator()

    # Valid code
    valid_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""

    result = validator.validate(valid_code)
    print(f"Valid code: {result.is_valid}")

    # Invalid code (security issue)
    invalid_code = """
import os
os.system('ls -la')
"""

    result = validator.validate(invalid_code)
    print(f"\nInvalid code: {result.is_valid}")
    print(f"Errors: {result.errors}")


def example_custom_hooks() -> None:
    """Example: Custom hooks for execution lifecycle."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Hooks")
    print("=" * 60)

    # Define hooks
    def pre_validation_hook(code: str) -> None:
        print(f"[PRE-VALIDATION] Validating {len(code)} characters of code")

    def post_validation_hook(code: str, errors: list[str]) -> None:
        if errors:
            print(f"[POST-VALIDATION] Found {len(errors)} errors")
        else:
            print("[POST-VALIDATION] Validation passed")

    def pre_execution_hook(code: str, context: dict[str, Any]) -> None:
        print(f"[PRE-EXECUTION] Executing in {context.get('mode', 'unknown')} mode")

    def post_execution_hook(code: str, result: Any) -> None:
        print("[POST-EXECUTION] Execution completed")

    def error_hook(error: Exception, code: str) -> None:
        print(f"[ERROR] {type(error).__name__}: {error}")

    # Create configuration with hooks
    config = create_safe_config()
    config.hooks.pre_validation_hooks.append(pre_validation_hook)
    config.hooks.post_validation_hooks.append(post_validation_hook)
    config.hooks.pre_execution_hooks.append(pre_execution_hook)
    config.hooks.post_execution_hooks.append(post_execution_hook)
    config.hooks.error_hooks.append(error_hook)

    # Execute code
    executor = CodeExecutor(config)
    code = """
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
print(f"Total: {total}")
"""

    result = executor.execute(code)
    print(f"\nFinal result: {result.output}")


def example_custom_validators() -> None:
    """Example: Custom validation rules."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Validators")
    print("=" * 60)

    # Define custom validator
    def no_loops_validator(code: str) -> list[str]:
        """Disallow loops in code."""
        errors = []
        if "for " in code or "while " in code:
            errors.append("Loops are not allowed in this context")
        return errors

    def max_lines_validator(code: str) -> list[str]:
        """Limit code to 10 lines."""
        errors = []
        lines = code.strip().split("\n")
        if len(lines) > 10:
            errors.append(f"Code exceeds 10 lines ({len(lines)} lines)")
        return errors

    # Create configuration with custom validators
    config = create_safe_config()
    config.validation.custom_validators.append(no_loops_validator)
    config.validation.custom_validators.append(max_lines_validator)

    executor = CodeExecutor(config)

    # Try code with loop (should fail)
    code_with_loop = """
for i in range(10):
    print(i)
"""

    result = executor.execute(code_with_loop)
    print(f"Code with loop - Success: {result.is_success()}")
    print(f"Validation errors: {result.validation_errors}")

    # Try code without loop (should pass)
    code_without_loop = """
numbers = list(range(10))
print(numbers)
"""

    result = executor.execute(code_without_loop)
    print(f"\nCode without loop - Success: {result.is_success()}")
    print(f"Output: {result.output}")


def example_execution_modes() -> None:
    """Example: Different execution modes."""
    print("\n" + "=" * 60)
    print("Example 5: Execution Modes")
    print("=" * 60)

    code = """
result = [x**2 for x in range(5)]
print(f"Squares: {result}")
"""

    # Safe mode
    print("\n--- Safe Mode ---")
    safe_config = create_safe_config()
    executor = CodeExecutor(safe_config)
    result = executor.execute(code)
    print(f"Status: {result.status.value}")
    print(f"Output: {result.output}")

    # Restricted mode
    print("\n--- Restricted Mode ---")
    restricted_config = create_restricted_config()
    executor = CodeExecutor(restricted_config)
    result = executor.execute(code)
    print(f"Status: {result.status.value}")
    print(f"Output: {result.output}")


def example_dry_run() -> None:
    """Example: Dry run mode (validation only)."""
    print("\n" + "=" * 60)
    print("Example 6: Dry Run Mode")
    print("=" * 60)

    config = create_safe_config()
    config.dry_run = True

    executor = CodeExecutor(config)

    code = """
def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n-1)

print(calculate_factorial(5))
"""

    result = executor.execute(code)
    print(f"Dry run result: {result.output}")
    print(f"Status: {result.status.value}")
    print(f"Validation errors: {result.validation_errors}")


def example_resource_limits() -> None:
    """Example: Resource limits configuration."""
    print("\n" + "=" * 60)
    print("Example 7: Resource Limits")
    print("=" * 60)

    config = create_safe_config()
    config.resources.max_execution_time = 5.0  # 5 seconds
    config.resources.max_memory_mb = 100  # 100 MB

    executor = CodeExecutor(config)

    code = """
# Quick computation
result = sum(range(1000))
print(f"Sum: {result}")
"""

    result = executor.execute(code)
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"Output: {result.output}")


def example_execution_stats() -> None:
    """Example: Execution statistics."""
    print("\n" + "=" * 60)
    print("Example 8: Execution Statistics")
    print("=" * 60)

    executor = CodeExecutor(create_safe_config())

    # Execute multiple times
    codes = [
        "print('Hello, World!')",
        "result = 2 + 2\nprint(result)",
        "import os",  # This will fail validation
        "numbers = [1, 2, 3]\nprint(sum(numbers))",
    ]

    for code in codes:
        executor.execute(code)

    # Get statistics
    stats = executor.get_stats()
    print(f"Total executions: {stats['total_executions']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1%}")


def example_caching() -> None:
    """Example: Result caching."""
    print("\n" + "=" * 60)
    print("Example 9: Result Caching")
    print("=" * 60)

    config = create_safe_config()
    config.enable_caching = True
    config.cache_ttl = 60  # 60 seconds

    executor = CodeExecutor(config)

    code = """
result = sum(range(100))
print(f"Sum: {result}")
"""

    # First execution
    import time

    start = time.time()
    _result1 = executor.execute(code)
    time1 = time.time() - start

    # Second execution (should be cached)
    start = time.time()
    _result2 = executor.execute(code)
    time2 = time.time() - start

    print(f"First execution: {time1:.4f}s")
    print(f"Second execution (cached): {time2:.4f}s")
    print(f"Speedup: {time1 / time2:.1f}x")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CODE EXECUTION EXAMPLES")
    print("=" * 60)

    example_basic_execution()
    example_validation()
    example_custom_hooks()
    example_custom_validators()
    example_execution_modes()
    example_dry_run()
    example_resource_limits()
    example_execution_stats()
    example_caching()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
