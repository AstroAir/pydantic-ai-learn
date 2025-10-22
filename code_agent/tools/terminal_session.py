"""
Real-Time Terminal Session

Interactive terminal session management with streaming I/O, process lifecycle
control, and session persistence.

Features:
- Real-time streaming stdout/stderr
- Interactive stdin handling
- Process lifecycle management (start, stop, kill, restart)
- Session state tracking
- Multi-session support
- Integration with terminal sandbox

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils.terminal_exec import InteractiveTerminalSession
from .terminal_sandbox import TerminalSandbox

# ============================================================================
# Session Types
# ============================================================================


class SessionState(str, Enum):
    """Terminal session states."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SessionInfo:
    """Information about a terminal session."""

    session_id: str
    """Unique session identifier"""

    state: SessionState
    """Current session state"""

    created_at: float
    """Creation timestamp"""

    last_activity: float
    """Last activity timestamp"""

    command_count: int = 0
    """Number of commands executed"""

    error_count: int = 0
    """Number of errors encountered"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "command_count": self.command_count,
            "error_count": self.error_count,
            "metadata": self.metadata,
        }


# ============================================================================
# Real-Time Terminal Session
# ============================================================================


class RealTimeTerminalSession:
    """
    Real-time interactive terminal session with streaming I/O.

    Provides secure command execution with real-time output streaming,
    interactive input handling, and process lifecycle management.
    """

    def __init__(
        self,
        sandbox: TerminalSandbox | None = None,
        session_id: str | None = None,
        working_directory: Path | None = None,
        environment: dict[str, str] | None = None,
    ):
        """
        Initialize terminal session.

        Args:
            sandbox: Terminal sandbox for security (creates default if None)
            session_id: Session identifier (generates UUID if None)
            working_directory: Working directory for commands
            environment: Environment variables
        """
        self.sandbox = sandbox or TerminalSandbox()
        self.session_id = session_id or str(uuid.uuid4())
        self.working_directory = working_directory
        self.environment = environment or {}

        self.info = SessionInfo(
            session_id=self.session_id,
            state=SessionState.IDLE,
            created_at=time.time(),
            last_activity=time.time(),
        )

        self._interactive_session: InteractiveTerminalSession | None = None
        self._output_callbacks: list[Callable[[str, str], Awaitable[None]]] = []
        self._error_callbacks: list[Callable[[Exception], Awaitable[None]]] = []

    @property
    def is_running(self) -> bool:
        """Check if session is running."""
        return self.info.state == SessionState.RUNNING

    @property
    def is_interactive(self) -> bool:
        """Check if interactive session is active."""
        return self._interactive_session is not None and self._interactive_session.is_running

    def add_output_callback(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        """
        Add callback for output events.

        Args:
            callback: Async callback function(stream_type, data)
                     stream_type is 'stdout' or 'stderr'
        """
        self._output_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], Awaitable[None]]) -> None:
        """
        Add callback for error events.

        Args:
            callback: Async callback function(error)
        """
        self._error_callbacks.append(callback)

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        stream: bool = True,
    ) -> tuple[str, str, int]:
        """
        Execute command with real-time streaming.

        Args:
            command: Command to execute
            timeout: Execution timeout
            stream: Enable streaming output

        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        self.info.last_activity = time.time()
        self.info.command_count += 1

        # Validate command first
        validation_result = self.sandbox.validate_command(command)
        if not validation_result.is_valid:
            self.info.error_count += 1
            error_msg = "; ".join(validation_result.errors)
            await self._notify_error(ValueError(f"Command validation failed: {error_msg}"))
            return "", error_msg, 403

        # Create output callbacks for streaming
        async def on_stdout(data: str) -> None:
            await self._notify_output("stdout", data)

        async def on_stderr(data: str) -> None:
            await self._notify_output("stderr", data)

        # Execute command
        try:
            result = await self.sandbox.execute_async(
                command,
                timeout=timeout,
                cwd=self.working_directory,
                env=self.environment,
                stream=stream,
                on_stdout=on_stdout if stream else None,
                on_stderr=on_stderr if stream else None,
            )

            if result.exit_code != 0 or result.error:
                self.info.error_count += 1

            return result.stdout, result.stderr, result.exit_code or 0

        except Exception as e:
            self.info.error_count += 1
            await self._notify_error(e)
            return "", str(e), -1

    async def start_interactive(
        self,
        shell_command: str = "python",
        *,
        auto_restart: bool = False,
    ) -> bool:
        """
        Start interactive terminal session.

        Args:
            shell_command: Shell/interpreter to start (e.g., 'python', 'bash')
            auto_restart: Automatically restart on exit

        Returns:
            True if started successfully
        """
        if self.is_interactive:
            return True

        # Validate shell command
        validation_result = self.sandbox.validate_command(shell_command)
        if not validation_result.is_valid:
            error_msg = "; ".join(validation_result.errors)
            await self._notify_error(ValueError(f"Shell command validation failed: {error_msg}"))
            return False

        try:
            self.info.state = SessionState.STARTING
            self._interactive_session = InteractiveTerminalSession(shell_command)

            await self._interactive_session.start(
                cwd=self.working_directory,
                env=self.environment,
            )

            self.info.state = SessionState.RUNNING
            self.info.last_activity = time.time()

            # Start output monitoring
            asyncio.create_task(self._monitor_interactive_output())

            return True

        except Exception as e:
            self.info.state = SessionState.ERROR
            await self._notify_error(e)
            return False

    async def send_input(self, data: str) -> bool:
        """
        Send input to interactive session.

        Args:
            data: Input data to send

        Returns:
            True if sent successfully
        """
        if not self.is_interactive or not self._interactive_session:
            return False

        try:
            await self._interactive_session.send(data)
            self.info.last_activity = time.time()
            return True

        except Exception as e:
            await self._notify_error(e)
            return False

    async def stop_interactive(self, force: bool = False) -> bool:
        """
        Stop interactive session.

        Args:
            force: Force kill instead of graceful termination

        Returns:
            True if stopped successfully
        """
        if not self._interactive_session:
            return True

        try:
            self.info.state = SessionState.STOPPING

            if force:
                await self._interactive_session.kill()
            else:
                await self._interactive_session.terminate()

            self._interactive_session = None
            self.info.state = SessionState.STOPPED
            return True

        except Exception as e:
            self.info.state = SessionState.ERROR
            await self._notify_error(e)
            return False

    async def restart_interactive(self, shell_command: str | None = None) -> bool:
        """
        Restart interactive session.

        Args:
            shell_command: New shell command (uses previous if None)

        Returns:
            True if restarted successfully
        """
        # Stop current session
        await self.stop_interactive(force=True)

        # Start new session
        if shell_command:
            return await self.start_interactive(shell_command)
        # Use default
        return await self.start_interactive()

    async def _monitor_interactive_output(self) -> None:
        """Monitor interactive session output and trigger callbacks."""
        if not self._interactive_session:
            return

        try:
            while self.is_interactive and self._interactive_session:
                # Check for new stdout
                if self._interactive_session.stdout_buf:
                    stdout_data = "".join(self._interactive_session.stdout_buf)
                    self._interactive_session.stdout_buf.clear()
                    await self._notify_output("stdout", stdout_data)

                # Check for new stderr
                if self._interactive_session.stderr_buf:
                    stderr_data = "".join(self._interactive_session.stderr_buf)
                    self._interactive_session.stderr_buf.clear()
                    await self._notify_output("stderr", stderr_data)

                # Small delay to prevent busy loop
                await asyncio.sleep(0.1)

        except Exception as e:
            await self._notify_error(e)

    async def _notify_output(self, stream_type: str, data: str) -> None:
        """Notify output callbacks."""
        for callback in self._output_callbacks:
            with contextlib.suppress(Exception):
                await callback(stream_type, data)

    async def _notify_error(self, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            with contextlib.suppress(Exception):
                await callback(error)

    def get_info(self) -> SessionInfo:
        """Get session information."""
        return self.info

    async def cleanup(self) -> None:
        """Clean up session resources."""
        if self.is_interactive:
            await self.stop_interactive(force=True)


# ============================================================================
# Multi-Session Manager
# ============================================================================


class TerminalSessionManager:
    """
    Manages multiple terminal sessions.

    Provides session creation, lookup, cleanup, and lifecycle management
    for multiple concurrent terminal sessions.
    """

    def __init__(self, sandbox: TerminalSandbox | None = None, max_sessions: int = 10):
        """
        Initialize session manager.

        Args:
            sandbox: Shared terminal sandbox for all sessions
            max_sessions: Maximum number of concurrent sessions
        """
        self.sandbox = sandbox or TerminalSandbox()
        self.max_sessions = max_sessions
        self._sessions: dict[str, RealTimeTerminalSession] = {}
        self._session_timeout_minutes = 30

    def create_session(
        self,
        session_id: str | None = None,
        working_directory: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> RealTimeTerminalSession:
        """
        Create a new terminal session.

        Args:
            session_id: Session identifier (generates UUID if None)
            working_directory: Working directory
            environment: Environment variables

        Returns:
            New terminal session

        Raises:
            ValueError: If max sessions exceeded
        """
        # Check session limit
        if len(self._sessions) >= self.max_sessions:
            # Try to clean up inactive sessions
            self._cleanup_inactive_sessions()

            if len(self._sessions) >= self.max_sessions:
                raise ValueError(f"Maximum number of sessions ({self.max_sessions}) exceeded")

        # Create session
        session = RealTimeTerminalSession(
            sandbox=self.sandbox,
            session_id=session_id,
            working_directory=working_directory,
            environment=environment,
        )

        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> RealTimeTerminalSession | None:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Terminal session or None if not found
        """
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[SessionInfo]:
        """
        List all sessions.

        Returns:
            List of session information
        """
        return [session.get_info() for session in self._sessions.values()]

    async def close_session(self, session_id: str, force: bool = False) -> bool:
        """
        Close a session.

        Args:
            session_id: Session identifier
            force: Force close

        Returns:
            True if closed successfully
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        await session.cleanup()
        del self._sessions[session_id]
        return True

    async def close_all_sessions(self, force: bool = False) -> None:
        """
        Close all sessions.

        Args:
            force: Force close all sessions
        """
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id, force=force)

    def _cleanup_inactive_sessions(self) -> None:
        """Clean up inactive sessions based on timeout."""
        current_time = time.time()
        timeout_seconds = self._session_timeout_minutes * 60

        inactive_sessions = [
            session_id
            for session_id, session in self._sessions.items()
            if current_time - session.info.last_activity > timeout_seconds
        ]

        for session_id in inactive_sessions:
            # Use asyncio to clean up
            asyncio.create_task(self.close_session(session_id, force=True))

    def get_stats(self) -> dict[str, Any]:
        """Get session manager statistics."""
        return {
            "total_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "active_sessions": sum(1 for s in self._sessions.values() if s.is_running),
            "interactive_sessions": sum(1 for s in self._sessions.values() if s.is_interactive),
        }


__all__ = [
    "SessionState",
    "SessionInfo",
    "RealTimeTerminalSession",
    "TerminalSessionManager",
]
