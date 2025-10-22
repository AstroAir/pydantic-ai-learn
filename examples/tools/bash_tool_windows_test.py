"""
Windows-specific test for BashTool using PowerShell

This script tests the tool on Windows using PowerShell commands.
Run with: python examples/tools/bash_tool_windows_test.py
"""

import asyncio
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.bash_tool import (
    BashCommandInput,
    BashTool,
    run_bash_command_async,
)


async def test_powershell_async():
    """Test async execution with PowerShell."""
    print("Testing async PowerShell execution...")
    try:
        # Use PowerShell command
        result = await run_bash_command_async("echo 'PowerShell test'")
        print(f"Output: {result.output}")
        print(f"Exit code: {result.exit_code}")
        print(f"Success: {result.success}")
        print("✓ Async PowerShell execution works")
        return True
    except Exception as e:
        print(f"✗ Async PowerShell execution failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_powershell_commands():
    """Test various PowerShell commands."""
    print("\nTesting various PowerShell commands...")
    try:
        async with BashTool() as bash:
            # Test Get-Location (PowerShell equivalent of pwd)
            result = await bash.run_command_async("Get-Location")
            print(f"Current location: {result.output.strip()}")

            # Test Get-Date
            result = await bash.run_command_async("Get-Date")
            print(f"Current date: {result.output.strip()}")

            # Test simple echo
            result = await bash.run_command_async("Write-Output 'Hello from PowerShell'")
            print(f"Echo output: {result.output.strip()}")

            print("✓ PowerShell commands work")
            return True
    except Exception as e:
        print(f"✗ PowerShell commands failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run Windows-specific tests."""
    print("\n" + "=" * 60)
    print("BashTool Windows/PowerShell Test")
    print("=" * 60 + "\n")

    results = []

    # Async tests
    results.append(asyncio.run(test_powershell_async()))
    results.append(asyncio.run(test_powershell_commands()))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All Windows tests passed!")
        print("=" * 60 + "\n")
        return 0
    print(f"✗ {total - passed} test(s) failed")
    print("=" * 60 + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
