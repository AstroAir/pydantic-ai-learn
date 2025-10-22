# BashTool Platform Compatibility Notes

## Overview

The BashTool is designed primarily for Unix-like systems (Linux, macOS) where bash is the native shell. Windows support is provided with limitations.

## Platform Support

### ✅ Fully Supported

- **Linux** - Native bash support
- **macOS** - Native bash support
- **WSL (Windows Subsystem for Linux)** - Full bash support when WSL is installed

### ⚠️ Limited Support

- **Windows (PowerShell)** - Basic functionality works but with limitations:
  - Character encoding issues may occur with non-ASCII output
  - PowerShell syntax differs from bash (use PowerShell commands, not bash commands)
  - Persistent session mode has limitations
  - Some bash-specific features won't work

### ❌ Not Supported

- **Windows CMD** - Not recommended, use PowerShell or WSL instead

## Recommendations by Platform

### Linux / macOS
Use the tool as documented - full bash functionality available.

```python
from tools.bash_tool import BashTool

with BashTool() as bash:
    result = bash.run_command("ls -la")
    print(result.output)
```

### Windows with WSL
Install WSL and the tool will automatically detect and use it:

```bash
# Install WSL (run in PowerShell as Administrator)
wsl --install
```

Then use the tool normally - it will use WSL bash automatically.

### Windows with Git Bash
Install Git for Windows (includes Git Bash) and the tool will detect it:

Download from: https://git-scm.com/download/win

### Windows with PowerShell Only
The tool will fall back to PowerShell. Use PowerShell syntax, not bash:

```python
from tools.bash_tool import BashTool

async with BashTool() as bash:
    # Use PowerShell commands
    result = await bash.run_command_async("Get-Location")
    result = await bash.run_command_async("Get-ChildItem")
    result = await bash.run_command_async("Write-Output 'Hello'")
```

**Known Issues on PowerShell:**
- Character encoding may cause garbled output for non-ASCII characters
- Persistent session mode may not work reliably
- Exit codes may not be captured correctly
- Recommend using async mode only

## Shell Detection Order

The tool detects shells in this order:

1. **Windows:**
   - WSL bash (if `wsl` command available)
   - Git Bash (common installation paths)
   - bash in PATH
   - PowerShell (fallback)

2. **Unix-like (Linux/macOS):**
   - bash in PATH
   - /bin/bash
   - /usr/bin/bash
   - /usr/local/bin/bash

## Testing

### Unix-like Systems
```bash
# Run full test suite
pytest tests/test_bash_tool.py -v

# Run examples
python tools/bash_tool_example.py
```

### Windows
```bash
# Quick validation (may have encoding issues)
python tools/bash_tool_windows_test.py

# For best results, install WSL first
wsl --install
# Then run normal tests
```

## Troubleshooting

### "No suitable shell found"
- **Linux/macOS:** Install bash: `sudo apt install bash` or `brew install bash`
- **Windows:** Install WSL (`wsl --install`) or Git Bash

### Garbled Output on Windows
- This is a known limitation with PowerShell encoding
- **Solution:** Install and use WSL for proper bash support
- **Workaround:** Use async mode and PowerShell-native commands

### Commands Not Working
- **Check your platform:** Bash commands won't work in PowerShell
- **Use appropriate syntax:**
  - Bash: `ls -la`, `pwd`, `echo 'test'`
  - PowerShell: `Get-ChildItem`, `Get-Location`, `Write-Output 'test'`

## Future Improvements

Potential enhancements for better Windows support:
- Better PowerShell encoding handling
- Automatic command translation (bash → PowerShell)
- Explicit shell selection parameter
- Better error messages for platform-specific issues

## Conclusion

**For production use:**
- Linux/macOS: Use as-is ✅
- Windows: Install WSL for best experience ✅
- Windows PowerShell only: Use with caution, async mode recommended ⚠️
