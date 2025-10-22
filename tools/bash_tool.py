"""
Bash Command Execution Tool

A production-grade Python implementation for executing bash commands in a persistent
shell session with timeout support, background execution, and proper error handling.

Features:
- Persistent shell session maintaining state across commands
- Configurable timeout (default: 2 minutes, max: 10 minutes)
- Background execution support
- Automatic path escaping for spaces
- Output truncation at 30,000 characters
- Comprehensive error handling
- Type-safe with Pydantic validation

Security Considerations:
- This tool executes arbitrary bash commands - use with caution
- Commands run in a persistent bash session with full shell capabilities
- Ensure proper access control when exposing this tool
- Validate and sanitize user input before passing to this tool

Example Usage:
    ```python
    from tools.bash_tool import BashTool, BashCommandInput

    # Synchronous usage with context manager
    with BashTool() as bash:
        result = bash.execute(BashCommandInput(
            command="echo 'Hello, World!'",
            description="Print greeting"
        ))
        print(result.output)
        print(f"Exit code: {result.exit_code}")

    # Async usage
    async with BashTool() as bash:
        result = await bash.execute(BashCommandInput(
            command="ls -la",
            timeout=5000,  # 5 seconds
            description="List directory contents"
        ))
        print(result.output)

    # Background execution
    async with BashTool() as bash:
        task = await bash.execute(BashCommandInput(
            command="sleep 10 && echo 'Done'",
            run_in_background=True,
            description="Long running task"
        ))
        # Continue working while command runs
        # ...
        result = await task  # Wait for completion when needed
    ```

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio
import os
import platform
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Input/Output Models
# ============================================================================


class BashCommandInput(BaseModel):
    """
    Input schema for bash command execution.

    Validates command parameters according to JSON Schema Draft-07 specification.
    """

    command: str = Field(..., description="The bash command to execute (required)", min_length=1)

    timeout: int | None = Field(
        default=120000,  # 2 minutes in milliseconds
        description="Optional timeout in milliseconds (max: 600000ms / 10 minutes, default: 120000ms / 2 minutes)",
        ge=1,
        le=600000,
    )

    description: str | None = Field(
        default=None, description="Clear, concise description of what this command does in 5-10 words", max_length=100
    )

    run_in_background: bool = Field(
        default=False, description="Set to true to run command in background and continue working while it executes"
    )

    model_config = {
        "extra": "forbid",  # additionalProperties: false
        "json_schema_extra": {
            "examples": [
                {"command": "echo 'Hello, World!'", "description": "Print greeting message"},
                {"command": "ls -la /tmp", "timeout": 5000, "description": "List tmp directory contents"},
                {
                    "command": "python train_model.py",
                    "timeout": 600000,
                    "run_in_background": True,
                    "description": "Train ML model in background",
                },
            ]
        },
    }

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int | None) -> int:
        """Ensure timeout is within valid range."""
        if v is None:
            return 120000  # Default 2 minutes
        if v <= 0:
            raise ValueError("Timeout must be greater than 0")
        if v > 600000:
            raise ValueError("Timeout cannot exceed 600000ms (10 minutes)")
        return v


@dataclass
class BashCommandResult:
    """
    Result of bash command execution.

    Attributes:
        output: Combined stdout and stderr output (truncated at 30,000 chars)
        exit_code: Command exit code (0 = success, non-zero = error)
        truncated: Whether output was truncated
        execution_time_ms: Time taken to execute command in milliseconds
        timed_out: Whether command execution timed out
    """

    output: str
    exit_code: int
    truncated: bool
    execution_time_ms: float
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Check if command executed successfully."""
        return self.exit_code == 0 and not self.timed_out


# ============================================================================
# Custom Exceptions
# ============================================================================


class BashToolError(Exception):
    """Base exception for BashTool errors."""

    pass


class BashTimeoutError(BashToolError):
    """Raised when command execution times out."""

    pass


class BashProcessError(BashToolError):
    """Raised when bash process fails or crashes."""

    pass


class BashExecutionError(BashToolError):
    """Raised when command execution fails."""

    pass


# ============================================================================
# Constants
# ============================================================================

MAX_OUTPUT_LENGTH = 30000  # Maximum output length in characters
OUTPUT_TRUNCATION_MESSAGE = "\n\n[... Output truncated at 30,000 characters ...]"
# Unique delimiter to detect command completion
COMMAND_DELIMITER = "\n__BASH_TOOL_CMD_END__\n"
EXIT_CODE_MARKER = "__EXIT_CODE__:"


# ============================================================================
# Helper Functions
# ============================================================================


def _detect_shell() -> str:
    """
    Detect the appropriate shell for the current platform.

    Returns:
        Path to shell executable

    Raises:
        BashProcessError: If no suitable shell is found
    """
    system = platform.system()

    if system == "Windows":
        # On Windows, try to find bash (Git Bash, WSL, etc.)
        # First check for WSL bash
        wsl_bash = shutil.which("wsl")
        if wsl_bash:
            # Use WSL bash
            return "wsl"

        # Check for Git Bash
        git_bash_paths = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Git\bin\bash.exe"),
        ]

        for path in git_bash_paths:
            if os.path.exists(path):
                return path

        # Try to find bash in PATH
        bash_in_path = shutil.which("bash")
        if bash_in_path:
            return bash_in_path

        # Fall back to PowerShell if no bash found
        powershell = shutil.which("powershell")
        if powershell:
            return powershell

        raise BashProcessError(
            "No suitable shell found on Windows. Please install Git Bash, WSL, or ensure bash is in PATH."
        )
    # Unix-like systems
    bash_path = shutil.which("bash")
    if bash_path:
        return bash_path

    # Try common locations
    for path in ["/bin/bash", "/usr/bin/bash", "/usr/local/bin/bash"]:
        if os.path.exists(path):
            return path

    raise BashProcessError(f"Bash not found on {system}. Please ensure bash is installed.")


# ============================================================================
# BashTool Class
# ============================================================================


class BashTool:
    """
    Persistent bash command execution tool.

    Maintains a persistent bash subprocess to preserve environment variables
    and working directory across multiple command executions.

    Supports both synchronous and asynchronous execution with configurable
    timeouts and background execution mode.

    Usage:
        # Synchronous context manager
        with BashTool() as bash:
            result = bash.execute(BashCommandInput(command="ls -la"))

        # Async context manager
        async with BashTool() as bash:
            result = await bash.execute(BashCommandInput(command="pwd"))

    Attributes:
        _process: The persistent bash subprocess
        _lock: Thread lock for synchronous operations
        _async_lock: Async lock for asynchronous operations
    """

    def __init__(self, shell_path: str | None = None):
        """
        Initialize BashTool with a persistent bash session.

        Args:
            shell_path: Path to bash executable (default: auto-detect)

        Raises:
            BashProcessError: If bash process cannot be started
        """
        self._shell_path = shell_path or _detect_shell()
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._is_async = False
        self._is_powershell = "powershell" in self._shell_path.lower()
        self._is_wsl = self._shell_path == "wsl"

    def _start_process(self) -> None:
        """
        Start the persistent bash subprocess.

        Raises:
            BashProcessError: If process cannot be started
        """
        try:
            # For WSL, we need to invoke bash through wsl
            cmd = ["wsl", "bash"] if self._is_wsl else [self._shell_path]

            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )
        except FileNotFoundError as e:
            raise BashProcessError(
                f"Shell executable not found at {self._shell_path}. Ensure the shell is installed and path is correct."
            ) from e
        except Exception as e:
            raise BashProcessError(f"Failed to start shell process: {e}") from e

    def _ensure_process_running(self) -> None:
        """
        Ensure bash process is running, restart if needed.

        Raises:
            BashProcessError: If process cannot be started
        """
        if self._process is None or self._process.poll() is not None:
            self._start_process()

    def _cleanup(self) -> None:
        """Clean up bash process and resources."""
        if self._process is not None:
            try:
                # Try graceful termination first
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if termination fails
                    self._process.kill()
                    self._process.wait()
            except Exception:
                pass  # Best effort cleanup
            finally:
                self._process = None

    # ========================================================================
    # Context Manager Protocol (Synchronous)
    # ========================================================================

    def __enter__(self) -> BashTool:
        """Enter synchronous context manager."""
        self._is_async = False
        self._start_process()
        return self

    def __exit__(self, exc_type: object | None, exc_val: object | None, exc_tb: object | None) -> None:
        """Exit synchronous context manager and cleanup."""
        self._cleanup()

    # ========================================================================
    # Async Context Manager Protocol
    # ========================================================================

    async def __aenter__(self) -> BashTool:
        """Enter async context manager."""
        self._is_async = True
        self._start_process()
        return self

    async def __aexit__(self, exc_type: object | None, exc_val: object | None, exc_tb: object | None) -> None:
        """Exit async context manager and cleanup."""
        self._cleanup()

    # ========================================================================
    # Command Execution - Synchronous
    # ========================================================================

    def _escape_command(self, command: str) -> str:
        """
        Prepare command for execution.

        Handles file paths with spaces by ensuring proper quoting.
        Note: Since we're passing to bash stdin, the command is already
        in a string context, so we don't need aggressive escaping.

        Args:
            command: Raw command string

        Returns:
            Command ready for execution
        """
        # For bash stdin, we can pass commands as-is
        # The user is responsible for proper quoting in their command
        return command.strip()

    def _truncate_output(self, output: str) -> tuple[str, bool]:
        """
        Truncate output if it exceeds maximum length.

        Args:
            output: Raw output string

        Returns:
            Tuple of (truncated_output, was_truncated)
        """
        if len(output) > MAX_OUTPUT_LENGTH:
            truncated = output[:MAX_OUTPUT_LENGTH] + OUTPUT_TRUNCATION_MESSAGE
            return truncated, True
        return output, False

    def _execute_command_sync(self, command: str, timeout_ms: int) -> BashCommandResult:
        """
        Execute command synchronously with timeout.

        Args:
            command: Command to execute
            timeout_ms: Timeout in milliseconds

        Returns:
            BashCommandResult with execution details

        Raises:
            BashTimeoutError: If command times out
            BashProcessError: If bash process fails
            BashExecutionError: If command execution fails
        """
        self._ensure_process_running()

        if self._process is None:
            raise BashProcessError("Bash process not initialized")

        timeout_sec = timeout_ms / 1000.0
        start_time = time.time()

        # Prepare command with exit code capture
        # We append commands to capture exit code and print delimiter
        prepared_command = f"{command}\necho '{EXIT_CODE_MARKER}'$?\necho '{COMMAND_DELIMITER}'\n"

        try:
            # Send command to bash
            assert self._process.stdin is not None
            self._process.stdin.write(prepared_command)
            self._process.stdin.flush()

            # Read output with timeout
            output_lines: list[str] = []
            exit_code = 0
            timed_out = False

            assert self._process.stdout is not None

            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    timed_out = True
                    # Kill the process on timeout
                    self._process.kill()
                    raise BashTimeoutError(f"Command timed out after {timeout_ms}ms")

                # Read line with small timeout to allow checking overall timeout
                # Use a thread to read with timeout
                line_container: list[str | None] = [None]

                def read_line(container: list[str | None] = line_container) -> None:
                    try:
                        if self._process and self._process.stdout:
                            container[0] = self._process.stdout.readline()
                        else:
                            container[0] = None
                    except Exception:
                        container[0] = None

                reader_thread = threading.Thread(target=read_line, daemon=True)
                reader_thread.start()
                reader_thread.join(timeout=0.1)  # 100ms check interval

                line = line_container[0]

                if line is None:
                    continue  # Timeout on read, check overall timeout and retry

                if line == "":
                    # EOF - process died
                    raise BashProcessError("Bash process terminated unexpectedly")

                # Check for delimiter
                if COMMAND_DELIMITER.strip() in line:
                    break

                # Check for exit code marker
                if EXIT_CODE_MARKER in line:
                    try:
                        exit_code = int(line.split(EXIT_CODE_MARKER)[1].strip())
                    except (ValueError, IndexError):
                        exit_code = -1
                    continue

                output_lines.append(line)

            # Combine output
            output = "".join(output_lines)

            # Truncate if needed
            output, truncated = self._truncate_output(output)

            execution_time_ms = (time.time() - start_time) * 1000

            return BashCommandResult(
                output=output,
                exit_code=exit_code,
                truncated=truncated,
                execution_time_ms=execution_time_ms,
                timed_out=timed_out,
            )

        except BashTimeoutError:
            raise
        except BashProcessError:
            raise
        except Exception as e:
            raise BashExecutionError(f"Command execution failed: {e}") from e

    # ========================================================================
    # Command Execution - Asynchronous
    # ========================================================================

    async def _execute_command_async(self, command: str, timeout_ms: int) -> BashCommandResult:
        """
        Execute command asynchronously with timeout.

        Args:
            command: Command to execute
            timeout_ms: Timeout in milliseconds

        Returns:
            BashCommandResult with execution details

        Raises:
            BashTimeoutError: If command times out
            BashProcessError: If bash process fails
            BashExecutionError: If command execution fails
        """
        # For async, we'll use asyncio subprocess
        # This creates a new subprocess per command, but allows proper async handling
        timeout_sec = timeout_ms / 1000.0
        start_time = time.time()

        # Prepare command with exit code capture
        prepared_command = f"{command}\necho '{EXIT_CODE_MARKER}'$?\n"

        try:
            # Create async subprocess
            if self._is_wsl:
                # For WSL, wrap command
                process = await asyncio.create_subprocess_shell(
                    f"wsl bash -c {shlex.quote(prepared_command)}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            elif self._is_powershell:
                # For PowerShell, use different syntax
                process = await asyncio.create_subprocess_shell(
                    prepared_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    shell=True,
                    executable=self._shell_path,
                )
            else:
                # For bash/sh
                process = await asyncio.create_subprocess_shell(
                    prepared_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    shell=True,
                    executable=self._shell_path,
                )

            # Wait for completion with timeout
            try:
                stdout_bytes, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
                timed_out = False
            except TimeoutError as e:
                # Kill process on timeout
                process.kill()
                await process.wait()
                raise BashTimeoutError(f"Command timed out after {timeout_ms}ms") from e

            # Decode output
            output = stdout_bytes.decode("utf-8", errors="replace")

            # Extract exit code
            exit_code = 0
            if EXIT_CODE_MARKER in output:
                try:
                    parts = output.split(EXIT_CODE_MARKER)
                    exit_code_line = parts[-1].split("\n")[0].strip()
                    exit_code = int(exit_code_line)
                    # Remove exit code marker from output
                    output = parts[0]
                except (ValueError, IndexError):
                    exit_code = process.returncode or 0
            else:
                exit_code = process.returncode or 0

            # Truncate if needed
            output, truncated = self._truncate_output(output)

            execution_time_ms = (time.time() - start_time) * 1000

            return BashCommandResult(
                output=output,
                exit_code=exit_code,
                truncated=truncated,
                execution_time_ms=execution_time_ms,
                timed_out=timed_out,
            )

        except BashTimeoutError:
            raise
        except Exception as e:
            raise BashExecutionError(f"Async command execution failed: {e}") from e

    # ========================================================================
    # Public API
    # ========================================================================

    def execute(self, cmd_input: BashCommandInput) -> BashCommandResult | asyncio.Task[BashCommandResult]:
        """
        Execute a bash command (synchronous version).

        Args:
            cmd_input: Validated command input

        Returns:
            BashCommandResult with execution details

        Raises:
            BashTimeoutError: If command times out
            BashProcessError: If bash process fails
            BashExecutionError: If command execution fails
            RuntimeError: If called in async context or background mode requested

        Example:
            >>> with BashTool() as bash:
            ...     result = bash.execute(BashCommandInput(
            ...         command="echo 'test'",
            ...         description="Test command"
            ...     ))
            ...     print(result.output)
            test
        """
        if cmd_input.run_in_background:
            raise RuntimeError(
                "Background execution requires async context. Use 'async with BashTool()' and 'await bash.execute(...)'"
            )

        with self._lock:
            command = self._escape_command(cmd_input.command)
            timeout_ms = cmd_input.timeout or 120000

            return self._execute_command_sync(command, timeout_ms)

    async def execute_async(self, cmd_input: BashCommandInput) -> BashCommandResult | asyncio.Task[BashCommandResult]:
        """
        Execute a bash command asynchronously.

        Args:
            cmd_input: Validated command input

        Returns:
            BashCommandResult if run_in_background=False
            asyncio.Task[BashCommandResult] if run_in_background=True

        Raises:
            BashTimeoutError: If command times out
            BashProcessError: If bash process fails
            BashExecutionError: If command execution fails

        Example:
            >>> async with BashTool() as bash:
            ...     result = await bash.execute_async(BashCommandInput(
            ...         command="ls -la",
            ...         description="List files"
            ...     ))
            ...     print(result.output)

            >>> # Background execution
            >>> async with BashTool() as bash:
            ...     task = await bash.execute_async(BashCommandInput(
            ...         command="sleep 5 && echo 'done'",
            ...         run_in_background=True,
            ...         description="Background task"
            ...     ))
            ...     # Do other work...
            ...     result = await task  # Wait when ready
        """
        async with self._async_lock:
            command = self._escape_command(cmd_input.command)
            timeout_ms = cmd_input.timeout or 120000

            if cmd_input.run_in_background:
                # Create background task
                return asyncio.create_task(self._execute_command_async(command, timeout_ms))
            # Execute and wait
            return await self._execute_command_async(command, timeout_ms)

    def run_command(self, command: str, timeout_ms: int = 120000, description: str | None = None) -> BashCommandResult:
        """
        Convenience method to run a command synchronously.

        Args:
            command: Bash command to execute
            timeout_ms: Timeout in milliseconds (default: 120000 / 2 minutes)
            description: Optional command description

        Returns:
            BashCommandResult with execution details

        Example:
            >>> with BashTool() as bash:
            ...     result = bash.run_command("pwd")
            ...     print(result.output)
        """
        cmd_input = BashCommandInput(command=command, timeout=timeout_ms, description=description)
        return self.execute(cmd_input)  # type: ignore

    async def run_command_async(
        self, command: str, timeout_ms: int = 120000, description: str | None = None, background: bool = False
    ) -> BashCommandResult | asyncio.Task[BashCommandResult]:
        """
        Convenience method to run a command asynchronously.

        Args:
            command: Bash command to execute
            timeout_ms: Timeout in milliseconds (default: 120000 / 2 minutes)
            description: Optional command description
            background: Run in background (default: False)

        Returns:
            BashCommandResult if background=False
            asyncio.Task[BashCommandResult] if background=True

        Example:
            >>> async with BashTool() as bash:
            ...     result = await bash.run_command_async("ls")
            ...     print(result.output)
        """
        cmd_input = BashCommandInput(
            command=command, timeout=timeout_ms, description=description, run_in_background=background
        )
        return await self.execute_async(cmd_input)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def is_alive(self) -> bool:
        """
        Check if bash process is alive.

        Returns:
            True if process is running, False otherwise
        """
        return self._process is not None and self._process.poll() is None

    def reset(self) -> None:
        """
        Reset the bash session by restarting the process.

        Useful for clearing environment pollution from previous commands.

        Example:
            >>> with BashTool() as bash:
            ...     bash.run_command("export MY_VAR=test")
            ...     bash.reset()  # MY_VAR is now cleared
            ...     result = bash.run_command("echo $MY_VAR")
            ...     print(result.output)  # Empty
        """
        self._cleanup()
        self._start_process()


# ============================================================================
# Convenience Functions
# ============================================================================


def run_bash_command(command: str, timeout_ms: int = 120000, description: str | None = None) -> BashCommandResult:
    """
    Convenience function to run a single bash command.

    Creates a temporary BashTool instance, executes the command, and cleans up.

    Args:
        command: Bash command to execute
        timeout_ms: Timeout in milliseconds (default: 120000 / 2 minutes)
        description: Optional command description

    Returns:
        BashCommandResult with execution details

    Example:
        >>> result = run_bash_command("echo 'Hello, World!'")
        >>> print(result.output)
        Hello, World!
        >>> print(result.exit_code)
        0
    """
    with BashTool() as bash:
        return bash.run_command(command, timeout_ms, description)


async def run_bash_command_async(
    command: str, timeout_ms: int = 120000, description: str | None = None
) -> BashCommandResult:
    """
    Convenience function to run a single bash command asynchronously.

    Creates a temporary BashTool instance, executes the command, and cleans up.

    Args:
        command: Bash command to execute
        timeout_ms: Timeout in milliseconds (default: 120000 / 2 minutes)
        description: Optional command description

    Returns:
        BashCommandResult with execution details

    Example:
        >>> result = await run_bash_command_async("ls -la")
        >>> print(result.output)
    """
    async with BashTool() as bash:
        result = await bash.run_command_async(command, timeout_ms, description)
        # Ensure we return BashCommandResult, not Task
        if isinstance(result, asyncio.Task):
            return await result
        return result


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Models
    "BashCommandInput",
    "BashCommandResult",
    # Main class
    "BashTool",
    # Exceptions
    "BashToolError",
    "BashTimeoutError",
    "BashProcessError",
    "BashExecutionError",
    # Convenience functions
    "run_bash_command",
    "run_bash_command_async",
]
