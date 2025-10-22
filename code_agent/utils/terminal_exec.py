"""
Terminal command execution utilities.

Provides synchronous and asynchronous command execution, streaming I/O,
and simple interactive session support without external dependencies.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    timed_out: bool = False
    error: str | None = None
    start_ts: float = field(default_factory=time.time)
    end_ts: float = 0.0


async def _stream_reader(stream: asyncio.StreamReader, cb: Callable[[str], Awaitable[None]] | None) -> str:
    buf_parts: list[str] = []
    try:
        while True:
            chunk = await stream.read(1024)
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            buf_parts.append(text)
            if cb:
                await cb(text)
    except Exception:
        # Swallow read errors; reported at higher level
        pass
    return "".join(buf_parts)


async def run_command_async(
    cmd: str | Sequence[str],
    *,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    shell: bool = True,
    stream: bool = False,
    on_stdout: Callable[[str], Awaitable[None]] | None = None,
    on_stderr: Callable[[str], Awaitable[None]] | None = None,
) -> CommandResult:
    """Run a command asynchronously with optional streaming callbacks.

    On Windows, PTY is not used; standard pipes are employed.
    """
    result = CommandResult()
    result.start_ts = time.time()

    try:
        if shell and isinstance(cmd, (list, tuple)):
            # Join to a single string in shell mode
            command_str = " ".join(cmd)
            create = asyncio.create_subprocess_shell
            args = command_str
        elif shell and isinstance(cmd, str):
            create = asyncio.create_subprocess_shell
            args = cmd
        else:
            create = asyncio.create_subprocess_exec  # type: ignore[assignment]
            # Split naive; callers should prefer list form for exec
            args = cmd.split() if isinstance(cmd, str) else tuple(cmd)  # type: ignore[assignment]

        proc = await create(  # type: ignore[misc]
            *([args] if not isinstance(args, (list, tuple)) else args),  # type: ignore[arg-type]
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            if stream:
                stdout_task = asyncio.create_task(_stream_reader(proc.stdout, on_stdout))  # type: ignore[arg-type]
                stderr_task = asyncio.create_task(_stream_reader(proc.stderr, on_stderr))  # type: ignore[arg-type]
                await asyncio.wait([stdout_task, stderr_task], timeout=timeout)
                # If timed out, terminate
                if (stdout_task.done() and stderr_task.done()) is False and timeout is not None:
                    result.timed_out = True
                    with contextlib.suppress(ProcessLookupError):
                        proc.terminate()
                result.stdout = stdout_task.result() if stdout_task.done() else ""
                result.stderr = stderr_task.result() if stderr_task.done() else ""
            else:
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except TimeoutError:
                    result.timed_out = True
                    with contextlib.suppress(ProcessLookupError):
                        proc.terminate()
                    stdout_bytes, stderr_bytes = await proc.communicate()
                result.stdout = (stdout_bytes or b"").decode(errors="replace")
                result.stderr = (stderr_bytes or b"").decode(errors="replace")

            result.exit_code = await proc.wait()
        finally:
            result.end_ts = time.time()
            result.duration_s = result.end_ts - result.start_ts

    except FileNotFoundError as e:
        result.error = str(e)
        result.exit_code = 127
        result.end_ts = time.time()
        result.duration_s = result.end_ts - result.start_ts
    except PermissionError as e:
        result.error = str(e)
        result.exit_code = 126
        result.end_ts = time.time()
        result.duration_s = result.end_ts - result.start_ts
    except Exception as e:
        result.error = str(e)
        result.exit_code = -1
        result.end_ts = time.time()
        result.duration_s = result.end_ts - result.start_ts

    return result


def run_command(
    cmd: str | Sequence[str],
    *,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    shell: bool = True,
) -> CommandResult:
    """Synchronous wrapper around run_command_async."""
    try:
        return asyncio.run(
            run_command_async(
                cmd,
                timeout=timeout,
                cwd=cwd,
                env=env,
                shell=shell,
                stream=False,
            )
        )
    except RuntimeError:
        # Event loop already running; degrade to simple blocking subprocess
        import subprocess

        start = time.time()
        try:
            completed = subprocess.run(
                cmd if isinstance(cmd, (list, tuple)) else cmd,
                cwd=str(cwd) if cwd else None,
                env={**os.environ, **(env or {})},
                shell=shell,
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return CommandResult(
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                exit_code=completed.returncode,
                duration_s=time.time() - start,
                timed_out=False,
                error=None,
                start_ts=start,
                end_ts=time.time(),
            )
        except subprocess.TimeoutExpired as e:
            return CommandResult(
                stdout=e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr=e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""),
                exit_code=None,
                duration_s=time.time() - start,
                timed_out=True,
                error=str(e),
                start_ts=start,
                end_ts=time.time(),
            )
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr="",
                exit_code=-1,
                duration_s=time.time() - start,
                timed_out=False,
                error=str(e),
                start_ts=start,
                end_ts=time.time(),
            )


class InteractiveTerminalSession:
    """Minimal interactive terminal session backed by a subprocess."""

    def __init__(self, command: str | Sequence[str]) -> None:
        self.command = command
        self.process: asyncio.subprocess.Process | None = None
        self.stdout_buf: list[str] = []
        self.stderr_buf: list[str] = []
        self._stdout_task: asyncio.Task[Any] | None = None
        self._stderr_task: asyncio.Task[Any] | None = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(
        self, *, cwd: str | Path | None = None, env: dict[str, str] | None = None, shell: bool = True
    ) -> None:
        if self.process is not None:
            return
        if shell and isinstance(self.command, (list, tuple)):
            args = " ".join(self.command)
            create = asyncio.create_subprocess_shell
        elif shell and isinstance(self.command, str):
            args = self.command
            create = asyncio.create_subprocess_shell
        else:
            create = asyncio.create_subprocess_exec  # type: ignore[assignment]
            args = self.command  # type: ignore[assignment]

        self.process = await create(
            args if isinstance(args, str) else args[0],
            *(args[1:] if isinstance(args, (list, tuple)) else ()),
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **(env or {})},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def _pump(reader: asyncio.StreamReader, buf: list[str]) -> None:
            while True:
                chunk = await reader.read(1024)
                if not chunk:
                    break
                buf.append(chunk.decode(errors="replace"))

        if self.process.stdout:
            self._stdout_task = asyncio.create_task(_pump(self.process.stdout, self.stdout_buf))
        if self.process.stderr:
            self._stderr_task = asyncio.create_task(_pump(self.process.stderr, self.stderr_buf))

    async def send(self, data: str) -> None:
        if not self.process or not self.process.stdin:
            return
        self.process.stdin.write(data.encode())
        await self.process.stdin.drain()

    async def terminate(self) -> None:
        if self.process is None:
            return
        with contextlib.suppress(ProcessLookupError):
            self.process.terminate()
        await self._await_close()

    async def kill(self) -> None:
        if self.process is None:
            return
        with contextlib.suppress(ProcessLookupError):
            self.process.kill()
        await self._await_close()

    async def _await_close(self) -> None:
        if self.process:
            await self.process.wait()
        if self._stdout_task:
            with contextlib.suppress(Exception):
                await self._stdout_task
        if self._stderr_task:
            with contextlib.suppress(Exception):
                await self._stderr_task


__all__ = [
    "CommandResult",
    "run_command_async",
    "run_command",
    "InteractiveTerminalSession",
]
