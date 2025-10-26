"""Test BashTool with PowerShell fallback

Run with: python examples/tools/test_powershell_mode.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.bash_tool import BashTool

print("Testing BashTool with PowerShell...")
print("=" * 60)

# Force PowerShell by providing explicit path
import shutil  # noqa: E402

powershell_path = shutil.which("powershell")
print(f"PowerShell path: {powershell_path}")

try:
    # Test async mode (recommended for PowerShell)
    async def test_async() -> bool:
        print("\nTesting async mode with PowerShell commands...")
        async with BashTool(shell_path=powershell_path) as bash:
            # Use PowerShell commands
            result = await bash.run_command_async("Write-Output 'Hello from PowerShell'")
            print(f"Output: {result.output.strip()}")
            print(f"Exit code: {result.exit_code}")
            print(f"Success: {result.success}")

            result = await bash.run_command_async("Get-Location")
            print(f"\nCurrent location: {result.output.strip()}")

            return result.success

    success = asyncio.run(test_async())

    if success:
        print("\n✓ PowerShell mode works!")
    else:
        print("\n✗ PowerShell mode failed")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback

    traceback.print_exc()
