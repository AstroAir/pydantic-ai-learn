# BashTool - Python Bash Command Execution Tool

A production-grade Python implementation for executing bash commands in a persistent shell session with modern Python 3.12+ features.

## Features

✅ **Persistent Shell Session** - Maintains environment variables and working directory across commands
✅ **Configurable Timeouts** - Default 2 minutes, max 10 minutes
✅ **Background Execution** - Run commands asynchronously without blocking
✅ **Type-Safe** - Full type hints and Pydantic validation
✅ **Error Handling** - Comprehensive exception handling for all failure modes
✅ **Output Management** - Automatic truncation at 30,000 characters
✅ **Context Managers** - Proper resource cleanup with `with` and `async with`
✅ **Security Measures** - Subprocess security best practices

## Installation

No additional dependencies required beyond the project's existing dependencies:
- Python >= 3.12
- pydantic-ai >= 1.0.15 (includes Pydantic v2)

## Platform Compatibility

- ✅ **Linux** - Full support
- ✅ **macOS** - Full support
- ✅ **Windows (WSL)** - Full support when WSL is installed
- ⚠️ **Windows (PowerShell)** - Limited support (encoding issues, use async mode)

**For Windows users:** Install WSL for best experience: `wsl --install`

See [PLATFORM_NOTES.md](./PLATFORM_NOTES.md) for detailed platform-specific information.

## Quick Start

### Basic Synchronous Usage

```python
from tools.bash_tool import BashTool

with BashTool() as bash:
    result = bash.run_command("echo 'Hello, World!'")
    print(result.output)  # Hello, World!
    print(result.exit_code)  # 0
    print(result.success)  # True
```

### Asynchronous Usage

```python
from tools.bash_tool import BashTool

async with BashTool() as bash:
    result = await bash.run_command_async("ls -la")
    print(result.output)
```

### Background Execution

```python
async with BashTool() as bash:
    # Start long-running task in background
    task = await bash.run_command_async(
        "python train_model.py",
        timeout_ms=600000,  # 10 minutes
        background=True
    )

    # Do other work...
    other_result = await bash.run_command_async("echo 'Working...'")

    # Wait for background task when ready
    result = await task
```

### Convenience Functions

```python
from tools.bash_tool import run_bash_command, run_bash_command_async

# One-off command execution
result = run_bash_command("date")
print(result.output)

# Async one-off
result = await run_bash_command_async("whoami")
print(result.output)
```

## API Reference

### BashCommandInput

Pydantic model for input validation:

```python
class BashCommandInput(BaseModel):
    command: str  # Required: The bash command to execute
    timeout: int | None = 120000  # Optional: Timeout in ms (default: 2 min, max: 10 min)
    description: str | None = None  # Optional: Command description
    run_in_background: bool = False  # Optional: Run in background (async only)
```

### BashCommandResult

Dataclass containing execution results:

```python
@dataclass
class BashCommandResult:
    output: str  # Combined stdout/stderr (truncated at 30,000 chars)
    exit_code: int  # Command exit code (0 = success)
    truncated: bool  # Whether output was truncated
    execution_time_ms: float  # Execution time in milliseconds
    timed_out: bool  # Whether command timed out

    @property
    def success(self) -> bool:
        """Returns True if exit_code == 0 and not timed_out"""
```

### BashTool Class

Main class for command execution:

#### Methods

**`run_command(command, timeout_ms=120000, description=None) -> BashCommandResult`**
- Synchronous command execution
- Use within `with BashTool() as bash:` context

**`run_command_async(command, timeout_ms=120000, description=None, background=False) -> BashCommandResult | Task`**
- Asynchronous command execution
- Use within `async with BashTool() as bash:` context
- Returns `Task` if `background=True`

**`execute(cmd_input: BashCommandInput) -> BashCommandResult`**
- Execute with validated input model (sync)

**`execute_async(cmd_input: BashCommandInput) -> BashCommandResult | Task`**
- Execute with validated input model (async)

**`reset() -> None`**
- Reset bash session (clears environment variables, working directory)

**`is_alive() -> bool`**
- Check if bash process is running

### Exceptions

```python
BashToolError  # Base exception
BashTimeoutError  # Command timed out
BashProcessError  # Bash process failed
BashExecutionError  # Command execution failed
```

## Examples

### Persistent Session State

```python
with BashTool() as bash:
    # Set environment variable
    bash.run_command("export MY_VAR='test'")

    # Change directory
    bash.run_command("cd /tmp")

    # State is maintained
    result = bash.run_command("echo $MY_VAR && pwd")
    # Output: test\n/tmp

    # Reset session
    bash.reset()

    # State is cleared
    result = bash.run_command("echo $MY_VAR")
    # Output: (empty)
```

### Timeout Handling

```python
from tools.bash_tool import BashTimeoutError

with BashTool() as bash:
    try:
        result = bash.run_command(
            "sleep 10",
            timeout_ms=2000  # 2 seconds
        )
    except BashTimeoutError as e:
        print(f"Command timed out: {e}")
```

### Error Handling

```python
with BashTool() as bash:
    result = bash.run_command("ls /nonexistent")

    if not result.success:
        print(f"Command failed with exit code: {result.exit_code}")
        print(f"Error output: {result.output}")
```

### Concurrent Execution

```python
async with BashTool() as bash:
    # Run multiple commands concurrently
    tasks = [
        bash.run_command_async(f"echo 'Task {i}'")
        for i in range(5)
    ]

    results = await asyncio.gather(*tasks)

    for result in results:
        print(result.output)
```

### Input Validation

```python
from tools.bash_tool import BashCommandInput

# Valid input
cmd = BashCommandInput(
    command="ls -la",
    timeout=5000,
    description="List directory contents"
)

# Validation errors
try:
    invalid = BashCommandInput(
        command="test",
        timeout=700000  # Exceeds 600000ms limit
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## Security Considerations

⚠️ **Important Security Notes:**

1. **Arbitrary Command Execution**: This tool executes arbitrary bash commands with full shell capabilities
2. **User Input**: Always validate and sanitize user input before passing to this tool
3. **Access Control**: Implement proper access control when exposing this tool in applications
4. **Persistent Session**: Commands can modify the shell environment, affecting subsequent commands
5. **No Sandboxing**: Commands run with the same permissions as the Python process

## Testing

Run the test suite:

```bash
# Install pytest if not already installed
pip install pytest pytest-asyncio

# Run tests
pytest tests/test_bash_tool.py -v
```

Run examples:

```bash
python tools/bash_tool_example.py
```

## Architecture

### Synchronous Execution
- Uses `subprocess.Popen` with persistent bash process
- Commands sent via stdin, output read from stdout
- Timeout handled via threading

### Asynchronous Execution
- Uses `asyncio.create_subprocess_shell` for each command
- Timeout handled via `asyncio.wait_for`
- Background tasks use `asyncio.create_task`

### Output Handling
- stdout and stderr merged
- Truncated at 30,000 characters with clear indicator
- Exit codes captured via special marker

## Modern Python Features Used

- **Type Hints**: Full type annotations with Python 3.12+ union syntax (`|`)
- **Pydantic v2**: Input validation with field validators
- **Dataclasses**: Structured output with `@dataclass`
- **Context Managers**: Both sync (`__enter__`/`__exit__`) and async (`__aenter__`/`__aexit__`)
- **Async/Await**: Full asyncio support for concurrent execution
- **Pattern Matching**: Could be extended with match/case for error handling

## Limitations

- **Platform**: Requires bash (Unix/Linux/macOS or WSL on Windows)
- **Output Size**: Large outputs truncated at 30,000 characters
- **Timeout Granularity**: Minimum effective timeout ~100ms due to polling
- **Process Limit**: Each async command creates a new subprocess

## Contributing

When extending this tool:

1. Maintain type safety - add type hints to all new code
2. Update tests for new features
3. Follow existing error handling patterns
4. Document security implications
5. Ensure cleanup in all code paths

## License

Same as parent project.

## Author

The Augster - Created with modern Python best practices
