"""
Quick validation test for BashTool

This script performs basic validation to ensure the tool works correctly.
Run with: python examples/tools/bash_tool_quick_test.py
"""

import asyncio
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.bash_tool import (
    BashCommandInput,
    BashTimeoutError,
    BashTool,
    run_bash_command,
    run_bash_command_async,
)


def test_sync_basic() -> bool:
    """Test basic synchronous execution."""
    print("Testing synchronous execution...")
    try:
        with BashTool() as bash:
            result = bash.run_command("echo 'test'")
            assert result.success, "Command should succeed"
            assert "test" in result.output, "Output should contain 'test'"
            assert result.exit_code == 0, "Exit code should be 0"
            print("✓ Synchronous execution works")
            return True
    except Exception as e:
        print(f"✗ Synchronous execution failed: {e}")
        return False


def test_pydantic_validation() -> bool:
    """Test Pydantic input validation."""
    print("Testing Pydantic validation...")
    try:
        # Valid input
        cmd = BashCommandInput(command="echo 'test'", timeout=5000)
        assert cmd.timeout == 5000

        # Invalid timeout
        try:
            BashCommandInput(command="test", timeout=700000)
            print("✗ Validation should have failed for timeout > 600000")
            return False
        except ValueError:
            pass  # Expected

        print("✓ Pydantic validation works")
        return True
    except Exception as e:
        print(f"✗ Pydantic validation failed: {e}")
        return False


def test_timeout() -> bool:
    """Test timeout handling."""
    print("Testing timeout handling...")
    try:
        with BashTool() as bash:
            try:
                bash.run_command("sleep 10", timeout_ms=1000)
                print("✗ Timeout should have been raised")
                return False
            except BashTimeoutError:
                print("✓ Timeout handling works")
                return True
    except Exception as e:
        print(f"✗ Timeout test failed: {e}")
        return False


def test_convenience_function() -> bool:
    """Test convenience function."""
    print("Testing convenience function...")
    try:
        result = run_bash_command("echo 'convenience'")
        assert result.success
        assert "convenience" in result.output
        print("✓ Convenience function works")
        return True
    except Exception as e:
        print(f"✗ Convenience function failed: {e}")
        return False


async def test_async_basic() -> bool:
    """Test basic async execution."""
    print("Testing async execution...")
    try:
        async with BashTool() as bash:
            result = await bash.run_command_async("echo 'async test'")
            assert result.success
            assert "async test" in result.output
            print("✓ Async execution works")
            return True
    except Exception as e:
        print(f"✗ Async execution failed: {e}")
        return False


async def test_async_convenience() -> None:
    """Test async convenience function."""
    print("Testing async convenience function...")
    try:
        result = await run_bash_command_async("echo 'async convenience'")
        assert result.success
        assert "async convenience" in result.output
        print("✓ Async convenience function works")
        return True
    except Exception as e:
        print(f"✗ Async convenience function failed: {e}")
        return False


def main() -> int:
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("BashTool Quick Validation Test")
    print("=" * 60 + "\n")

    results = []

    # Synchronous tests
    results.append(test_sync_basic())
    results.append(test_pydantic_validation())
    results.append(test_timeout())
    results.append(test_convenience_function())

    # Async tests
    results.append(asyncio.run(test_async_basic()))
    results.append(asyncio.run(test_async_convenience()))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All validation tests passed!")
        print("=" * 60 + "\n")
        return 0
    print(f"✗ {total - passed} test(s) failed")
    print("=" * 60 + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
