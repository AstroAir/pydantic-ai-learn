"""
Example usage of BashTool

Demonstrates various features of the bash command execution tool including:
- Basic synchronous execution
- Asynchronous execution
- Background tasks
- Timeout handling
- Error handling
- Output truncation

Run with: python examples/tools/bash_tool_example.py
"""

import asyncio
import os
import sys
from typing import cast

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.bash_tool import (
    BashCommandInput,
    BashCommandResult,
    BashTimeoutError,
    BashTool,
    run_bash_command,
    run_bash_command_async,
)


def example_basic_sync():
    """Example 1: Basic synchronous command execution."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Synchronous Execution")
    print("=" * 60)

    with BashTool() as bash:
        # Simple command
        result = bash.run_command("echo 'Hello from BashTool!'")
        print(f"Output: {result.output.strip()}")
        print(f"Exit Code: {result.exit_code}")
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time_ms:.2f}ms")

        # Command with pipes
        result = bash.run_command("echo 'test' | tr 'a-z' 'A-Z'")
        print(f"\nPiped command output: {result.output.strip()}")

        # List files
        result = bash.run_command("ls -la | head -5")
        print(f"\nDirectory listing:\n{result.output}")


def example_persistent_session():
    """Example 2: Persistent session maintaining state."""
    print("\n" + "=" * 60)
    print("Example 2: Persistent Session State")
    print("=" * 60)

    with BashTool() as bash:
        # Set environment variable
        bash.run_command("export MY_VAR='persistent value'")

        # Change directory
        bash.run_command("cd /tmp")

        # Verify state is maintained
        result = bash.run_command("echo $MY_VAR")
        print(f"Environment variable: {result.output.strip()}")

        result = bash.run_command("pwd")
        print(f"Current directory: {result.output.strip()}")

        # Reset session
        bash.reset()

        # Verify state is cleared
        result = bash.run_command("echo $MY_VAR")
        print(f"After reset, MY_VAR: '{result.output.strip()}'")


def example_timeout_handling():
    """Example 3: Timeout handling."""
    print("\n" + "=" * 60)
    print("Example 3: Timeout Handling")
    print("=" * 60)

    with BashTool() as bash:
        try:
            # This will timeout
            result = bash.run_command(
                "sleep 10",
                timeout_ms=2000,  # 2 seconds
                description="Long running command",
            )
        except BashTimeoutError as e:
            print(f"Caught timeout error: {e}")

        # Quick command should succeed
        result = bash.run_command("echo 'Quick command'", timeout_ms=1000)
        print(f"Quick command succeeded: {result.output.strip()}")


def example_error_handling():
    """Example 4: Error handling."""
    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    with BashTool() as bash:
        # Command that fails
        result = bash.run_command("ls /nonexistent/directory")
        print(f"Failed command exit code: {result.exit_code}")
        print(f"Success: {result.success}")
        print(f"Error output: {result.output.strip()}")

        # Command with syntax error
        result = bash.run_command("echo 'unclosed quote")
        print(f"\nSyntax error exit code: {result.exit_code}")


async def example_async_execution():
    """Example 5: Asynchronous execution."""
    print("\n" + "=" * 60)
    print("Example 5: Asynchronous Execution")
    print("=" * 60)

    async with BashTool() as bash:
        # Single async command
        result = cast(BashCommandResult, await bash.run_command_async("echo 'Async hello'"))
        print(f"Async output: {result.output.strip()}")

        # Multiple concurrent commands
        results = []
        for i in range(3):
            result = cast(BashCommandResult, await bash.run_command_async(f"echo 'Task {i}'"))
            results.append(result)

        print("\nConcurrent execution results:")
        for i, result in enumerate(results):
            print(f"  Task {i}: {result.output.strip()}")


async def example_background_execution():
    """Example 6: Background execution."""
    print("\n" + "=" * 60)
    print("Example 6: Background Execution")
    print("=" * 60)

    async with BashTool() as bash:
        # Start background task
        print("Starting background task...")
        task = cast(
            asyncio.Task[BashCommandResult],
            await bash.run_command_async("sleep 3 && echo 'Background task complete'", background=True),
        )

        # Do other work while background task runs
        print("Doing other work while background task runs...")
        for i in range(3):
            result = cast(BashCommandResult, await bash.run_command_async(f"echo 'Foreground task {i}'"))
            print(f"  {result.output.strip()}")
            await asyncio.sleep(0.5)

        # Wait for background task
        print("Waiting for background task...")
        bg_result = await task
        print(f"Background task result: {bg_result.output.strip()}")


def example_convenience_functions():
    """Example 7: Convenience functions."""
    print("\n" + "=" * 60)
    print("Example 7: Convenience Functions")
    print("=" * 60)

    # One-off command execution
    result = run_bash_command("date")
    print(f"Current date: {result.output.strip()}")

    result = run_bash_command("uname -a")
    print(f"System info: {result.output.strip()}")


async def example_convenience_async():
    """Example 8: Async convenience functions."""
    print("\n" + "=" * 60)
    print("Example 8: Async Convenience Functions")
    print("=" * 60)

    result = await run_bash_command_async("whoami")
    print(f"Current user: {result.output.strip()}")


def example_pydantic_validation():
    """Example 9: Pydantic input validation."""
    print("\n" + "=" * 60)
    print("Example 9: Pydantic Input Validation")
    print("=" * 60)

    with BashTool() as bash:
        # Valid input
        cmd_input = BashCommandInput(command="echo 'test'", timeout=5000, description="Test command with validation")
        result = bash.execute(cmd_input)
        print(f"Valid input executed: {result.output.strip()}")

        # Test validation - timeout too large
        try:
            BashCommandInput(
                command="echo 'test'",
                timeout=700000,  # Exceeds 600000ms limit
            )
        except Exception as e:
            print(f"\nValidation error caught: {e}")

        # Test validation - empty command
        try:
            BashCommandInput(
                command="",  # Empty command
            )
        except Exception as e:
            print(f"Empty command error: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("BashTool Examples")
    print("=" * 60)

    # Synchronous examples
    example_basic_sync()
    example_persistent_session()
    example_timeout_handling()
    example_error_handling()
    example_convenience_functions()
    example_pydantic_validation()

    # Asynchronous examples
    asyncio.run(example_async_execution())
    asyncio.run(example_background_execution())
    asyncio.run(example_convenience_async())

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
