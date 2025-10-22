"""
Terminal Session Manager

Provides session persistence, auto-save, and recovery functionality.

Features:
- Session persistence (JSON and pickle formats)
- Auto-save on interval or event
- Session recovery after crashes
- Session history and replay
- Export to multiple formats

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

# ============================================================================
# Session Data Models
# ============================================================================


@dataclass
class SessionEntry:
    """Single session entry (prompt and response)."""

    timestamp: str
    prompt: str
    response: str
    status: str  # success, failed, interrupted
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionEntry:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SessionData:
    """Complete session data."""

    session_id: str
    start_time: str
    end_time: str | None = None
    entries: list[SessionEntry] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "entries": [e.to_dict() for e in self.entries],
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionData:
        """Create from dictionary."""
        entries = [SessionEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            session_id=data["session_id"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            entries=entries,
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Session Manager
# ============================================================================


class TerminalSessionManager:
    """
    Manages terminal session persistence and recovery.

    Features:
    - Auto-save sessions
    - Session recovery
    - Multiple export formats
    - Session history
    """

    def __init__(
        self,
        console: Console,
        session_dir: Path | None = None,
        auto_save: bool = True,
        auto_save_interval: int = 60,  # seconds
        max_sessions: int = 100,
    ):
        """
        Initialize session manager.

        Args:
            console: Rich console for output
            session_dir: Directory for session files
            auto_save: Enable auto-save
            auto_save_interval: Auto-save interval in seconds
            max_sessions: Maximum number of sessions to keep
        """
        self.console = console
        self.session_dir = session_dir or (Path.home() / ".code_agent" / "sessions")
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval
        self.max_sessions = max_sessions

        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Current session
        self.current_session: SessionData | None = None
        self.last_save_time = time.time()

        # Initialize new session
        self._init_new_session()

    def _init_new_session(self) -> None:
        """Initialize a new session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = SessionData(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            metadata={"version": "2.1", "platform": "code_agent_terminal"},
        )

    def add_entry(
        self,
        prompt: str,
        response: str,
        status: str = "success",
        execution_time: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add entry to current session.

        Args:
            prompt: User prompt
            response: Agent response
            status: Entry status
            execution_time: Execution time in seconds
            metadata: Additional metadata
        """
        if not self.current_session:
            self._init_new_session()

        entry = SessionEntry(
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            response=response,
            status=status,
            execution_time=execution_time,
            metadata=metadata or {},
        )

        assert self.current_session is not None
        self.current_session.entries.append(entry)

        # Auto-save if enabled
        if self.auto_save:
            current_time = time.time()
            if current_time - self.last_save_time >= self.auto_save_interval:
                self.save_session()
                self.last_save_time = current_time

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        """Update session metrics."""
        if self.current_session:
            self.current_session.metrics.update(metrics)

    def save_session(self, format: str = "json") -> Path | None:  # noqa: A002
        """
        Save current session to file.

        Args:
            format: Save format ('json' or 'pickle')

        Returns:
            Path to saved file or None on error
        """
        if not self.current_session:
            return None

        try:
            # Update end time
            self.current_session.end_time = datetime.now().isoformat()

            # Generate filename
            filename = f"session_{self.current_session.session_id}.{format}"
            filepath = self.session_dir / filename

            # Save based on format
            if format == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(self.current_session.to_dict(), f, indent=2)
            elif format == "pickle":
                with open(filepath, "wb") as f:
                    pickle.dump(self.current_session, f)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return filepath

        except Exception as e:
            self.console.print(f"[red]❌ Failed to save session: {e}[/red]")
            return None

    def load_session(self, session_id: str, format: str = "json") -> bool:  # noqa: A002
        """
        Load session from file.

        Args:
            session_id: Session ID to load
            format: File format

        Returns:
            True if loaded successfully
        """
        try:
            filename = f"session_{session_id}.{format}"
            filepath = self.session_dir / filename

            if not filepath.exists():
                self.console.print(f"[red]❌ Session not found: {session_id}[/red]")
                return False

            # Load based on format
            if format == "json":
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                    self.current_session = SessionData.from_dict(data)
            elif format == "pickle":
                with open(filepath, "rb") as f:
                    self.current_session = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.console.print(f"[green]✅ Session loaded: {session_id}[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Failed to load session: {e}[/red]")
            return False

    def list_sessions(self) -> list[str]:
        """List available sessions."""
        sessions = []
        for filepath in self.session_dir.glob("session_*.json"):
            session_id = filepath.stem.replace("session_", "")
            sessions.append(session_id)
        return sorted(sessions, reverse=True)

    def export_markdown(self, filepath: Path | None = None) -> Path | None:
        """
        Export current session to markdown.

        Args:
            filepath: Output file path

        Returns:
            Path to exported file or None on error
        """
        if not self.current_session:
            return None

        try:
            if not filepath:
                filepath = self.session_dir / f"session_{self.current_session.session_id}.md"

            with open(filepath, "w", encoding="utf-8") as f:
                # Header
                f.write("# Code Agent Session\n\n")
                f.write(f"**Session ID**: {self.current_session.session_id}\n")
                f.write(f"**Start Time**: {self.current_session.start_time}\n")
                if self.current_session.end_time:
                    f.write(f"**End Time**: {self.current_session.end_time}\n")
                f.write("\n")

                # Metrics
                if self.current_session.metrics:
                    f.write("## Metrics\n\n")
                    for key, value in self.current_session.metrics.items():
                        f.write(f"- **{key}**: {value}\n")
                    f.write("\n")

                # Entries
                f.write("## Conversation\n\n")
                for i, entry in enumerate(self.current_session.entries, 1):
                    f.write(f"### Entry {i}\n\n")
                    f.write(f"**Time**: {entry.timestamp}\n")
                    f.write(f"**Status**: {entry.status}\n")
                    if entry.execution_time > 0:
                        f.write(f"**Execution Time**: {entry.execution_time:.2f}s\n")
                    f.write("\n")

                    f.write("**Prompt**:\n")
                    f.write(f"```\n{entry.prompt}\n```\n\n")

                    f.write("**Response**:\n")
                    f.write(f"{entry.response}\n\n")
                    f.write("---\n\n")

            return filepath

        except Exception as e:
            self.console.print(f"[red]❌ Failed to export markdown: {e}[/red]")
            return None

    def cleanup_old_sessions(self) -> int:
        """
        Clean up old sessions beyond max_sessions limit.

        Returns:
            Number of sessions deleted
        """
        sessions = self.list_sessions()
        if len(sessions) <= self.max_sessions:
            return 0

        # Delete oldest sessions
        to_delete = sessions[self.max_sessions :]
        deleted = 0

        for session_id in to_delete:
            try:
                # Delete both json and pickle if they exist
                for ext in ["json", "pickle", "md"]:
                    filepath = self.session_dir / f"session_{session_id}.{ext}"
                    if filepath.exists():
                        filepath.unlink()
                deleted += 1
            except Exception:
                pass

        return deleted


__all__ = [
    "TerminalSessionManager",
    "SessionData",
    "SessionEntry",
]
