"""
Enhanced Streaming Components for Terminal UI

Provides advanced streaming visualization with:
- Character-by-character streaming with buffering
- Multiple progress indicators
- Adaptive rendering based on terminal size
- Stream rate limiting and smoothing
- Better error handling during streaming

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import shutil
import time
from collections import deque
from collections.abc import AsyncIterable, Callable

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

# ============================================================================
# Terminal Size Detection
# ============================================================================


def get_terminal_size() -> tuple[int, int]:
    """
    Get terminal size (columns, lines).

    Returns:
        Tuple of (columns, lines)
    """
    try:
        size = shutil.get_terminal_size()
        return size.columns, size.lines
    except Exception:
        return 80, 24  # Default fallback


def is_terminal_wide() -> bool:
    """Check if terminal is wide enough for advanced features."""
    columns, _ = get_terminal_size()
    return columns >= 100


# ============================================================================
# Stream Buffer
# ============================================================================


class StreamBuffer:
    """
    Buffer for streaming text with rate limiting and smoothing.

    Prevents overwhelming the terminal with too many updates.
    """

    def __init__(
        self,
        min_update_interval: float = 0.05,  # 50ms
        max_buffer_size: int = 1000,
    ):
        """
        Initialize stream buffer.

        Args:
            min_update_interval: Minimum time between updates in seconds
            max_buffer_size: Maximum buffer size before forcing flush
        """
        self.min_update_interval = min_update_interval
        self.max_buffer_size = max_buffer_size
        self.buffer: deque[str] = deque(maxlen=max_buffer_size)
        self.last_update_time = 0.0
        self.total_chars = 0

    def add(self, chunk: str) -> bool:
        """
        Add chunk to buffer.

        Args:
            chunk: Text chunk to add

        Returns:
            True if buffer should be flushed
        """
        self.buffer.append(chunk)
        self.total_chars += len(chunk)

        current_time = time.time()
        time_since_update = current_time - self.last_update_time

        # Flush if enough time passed or buffer is full
        should_flush = time_since_update >= self.min_update_interval or len(self.buffer) >= self.max_buffer_size

        if should_flush:
            self.last_update_time = current_time

        return should_flush

    def flush(self) -> str:
        """
        Flush buffer and return accumulated text.

        Returns:
            Accumulated text
        """
        text = "".join(self.buffer)
        self.buffer.clear()
        return text

    def get_stats(self) -> dict[str, int]:
        """Get buffer statistics."""
        return {
            "buffer_size": len(self.buffer),
            "total_chars": self.total_chars,
        }


# ============================================================================
# Enhanced Streaming Display
# ============================================================================


class EnhancedStreamingDisplay:
    """
    Enhanced streaming display with better visualization.

    Features:
    - Smooth character-by-character streaming
    - Progress indicators
    - Adaptive rendering
    - Rate limiting
    """

    def __init__(
        self, console: Console, show_progress: bool = True, show_stats: bool = False, adaptive_rendering: bool = True
    ):
        """
        Initialize streaming display.

        Args:
            console: Rich console
            show_progress: Show progress indicator
            show_stats: Show streaming statistics
            adaptive_rendering: Adapt to terminal size
        """
        self.console = console
        self.show_progress = show_progress
        self.show_stats = show_stats
        self.adaptive_rendering = adaptive_rendering

        self.buffer = StreamBuffer()
        self.start_time = 0.0
        self.chars_per_second = 0.0

    async def stream_text(
        self, text_stream: AsyncIterable[str], title: str = "Response", on_chunk: Callable[[str], None] | None = None
    ) -> str:
        """
        Stream text with enhanced visualization.

        Args:
            text_stream: Async iterator of text chunks
            title: Display title
            on_chunk: Optional callback for each chunk

        Returns:
            Complete streamed text
        """
        self.start_time = time.time()
        accumulated_text = ""

        # Check terminal size
        is_wide = is_terminal_wide() if self.adaptive_rendering else True

        # Create progress indicator
        if self.show_progress and is_wide:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console,
            )
            task = progress.add_task(f"[cyan]Streaming {title}...", total=None)
        else:
            progress = None
            task = None

        try:
            # Use Live display for smooth updates
            with Live(
                self._create_panel(accumulated_text, title, streaming=True),
                console=self.console,
                refresh_per_second=20,  # 50ms refresh
                transient=False,
            ) as live:
                if progress and task is not None:
                    progress.start()

                async for chunk in text_stream:
                    accumulated_text += chunk

                    # Call chunk callback
                    if on_chunk:
                        on_chunk(chunk)

                    # Add to buffer
                    should_update = self.buffer.add(chunk)

                    # Update display if needed
                    if should_update:
                        live.update(self._create_panel(accumulated_text, title, streaming=True))

                        if progress and task is not None:
                            progress.update(task, advance=len(chunk))

                # Final update
                live.update(self._create_panel(accumulated_text, title, streaming=False))

                if progress:
                    progress.stop()

        except Exception as e:
            self.console.print(f"[red]❌ Streaming error: {e}[/red]")
            raise

        # Calculate stats
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.chars_per_second = len(accumulated_text) / elapsed

        # Show stats if enabled
        if self.show_stats and is_wide:
            self._display_stats(len(accumulated_text), elapsed)

        return accumulated_text

    def _create_panel(self, text: str, title: str, streaming: bool) -> Panel:
        """Create display panel."""
        panel_title = f"[bold cyan]{title}[/bold cyan]"
        if streaming:
            panel_title += " [dim](streaming...)[/dim]"

        # Truncate text if terminal is narrow
        display_text = text
        if self.adaptive_rendering:
            columns, _ = get_terminal_size()
            if columns < 80 and len(text) > 500:
                display_text = text[:500] + "\n[dim]...(truncated for display)[/dim]"

        return Panel(
            Text(display_text),
            title=panel_title,
            border_style="cyan" if streaming else "green",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _display_stats(self, total_chars: int, elapsed: float) -> None:
        """Display streaming statistics."""
        stats_text = Text()
        stats_text.append("Streaming Stats: ", style="bold cyan")
        stats_text.append(f"{total_chars} chars in {elapsed:.2f}s ", style="dim")
        stats_text.append(f"({self.chars_per_second:.0f} chars/s)", style="dim green")

        self.console.print(stats_text)


# ============================================================================
# Multi-Progress Tracker
# ============================================================================


class MultiProgressTracker:
    """
    Track multiple concurrent operations with progress bars.

    Useful for tracking tool executions, file operations, etc.
    """

    def __init__(self, console: Console):
        """Initialize tracker."""
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self.tasks: dict[str, TaskID] = {}

    def __enter__(self) -> MultiProgressTracker:
        """Enter context."""
        self.progress.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context."""
        self.progress.stop()

    def add_task(self, name: str, total: int | None = None) -> str:
        """
        Add a task to track.

        Args:
            name: Task name
            total: Total units (None for indeterminate)

        Returns:
            Task ID
        """
        task_id = self.progress.add_task(name, total=total)
        self.tasks[name] = task_id
        return name

    def update_task(self, name: str, advance: int = 1, description: str | None = None) -> None:
        """Update task progress."""
        if name in self.tasks:
            task_id = self.tasks[name]
            if description:
                self.progress.update(task_id, advance=advance, description=description)
            else:
                self.progress.update(task_id, advance=advance)

    def complete_task(self, name: str) -> None:
        """Mark task as complete."""
        if name in self.tasks:
            task_id = self.tasks[name]
            self.progress.update(task_id, completed=True)


__all__ = [
    "EnhancedStreamingDisplay",
    "StreamBuffer",
    "MultiProgressTracker",
    "get_terminal_size",
    "is_terminal_wide",
]
