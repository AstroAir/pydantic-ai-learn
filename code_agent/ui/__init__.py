"""
UI Module - Terminal User Interface Components

Provides terminal-based user interface for the code agent.

Exports:
    - CodeAgentTerminal: Main terminal UI class
    - launch_terminal: Launch terminal UI
    - launch_terminal_async: Launch terminal UI asynchronously
    - AdvancedInputHandler: Advanced input handling with history and completion
    - TerminalSessionManager: Session persistence and management
    - EnhancedStreamingDisplay: Enhanced streaming visualization
"""

from __future__ import annotations

from .input_handler import (
    PROMPT_TOOLKIT_AVAILABLE,
    AdvancedInputHandler,
    CommandCompleter,
    InputValidator,
)
from .runner import run_terminal
from .session_manager import (
    SessionData,
    SessionEntry,
    TerminalSessionManager,
)
from .streaming import (
    EnhancedStreamingDisplay,
    MultiProgressTracker,
    StreamBuffer,
    get_terminal_size,
    is_terminal_wide,
)
from .terminal import (
    CodeAgentTerminal,
    TerminalUIComponents,
    TerminalUIState,
    launch_terminal,
    launch_terminal_async,
)

__all__ = [
    # Core terminal
    "CodeAgentTerminal",
    "TerminalUIState",
    "TerminalUIComponents",
    "launch_terminal",
    "launch_terminal_async",
    "run_terminal",
    # Input handling
    "AdvancedInputHandler",
    "CommandCompleter",
    "InputValidator",
    "PROMPT_TOOLKIT_AVAILABLE",
    # Session management
    "TerminalSessionManager",
    "SessionData",
    "SessionEntry",
    # Streaming
    "EnhancedStreamingDisplay",
    "StreamBuffer",
    "MultiProgressTracker",
    "get_terminal_size",
    "is_terminal_wide",
]
