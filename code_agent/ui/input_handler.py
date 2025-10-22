"""
Advanced Input Handler for Terminal UI

Provides sophisticated input handling with:
- Command history with persistence
- Tab completion
- Multi-line input support
- Input validation
- Readline-like editing (Ctrl+A, Ctrl+E, etc.)
- Custom keyboard shortcuts for terminal operations
- Graceful fallback to basic input

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Prompt

# Try to import prompt_toolkit for advanced features
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.validation import ValidationError, Validator

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Import shortcut registry
from .shortcut_registry import ShortcutCategory, ShortcutRegistry

# ============================================================================
# Command Completer
# ============================================================================


class CommandCompleter:
    """Command completion for terminal input."""

    def __init__(self, commands: list[str] | None = None):
        """
        Initialize completer.

        Args:
            commands: List of commands to complete
        """
        self.commands = commands or [
            "exit",
            "quit",
            "q",
            "clear",
            "help",
            "metrics",
            "errors",
            "workflow",
            "history",
            "export",
            "save",
            "load",
            "reset",
            "status",
            "config",
        ]

        self.completer: WordCompleter | None
        if PROMPT_TOOLKIT_AVAILABLE:
            self.completer = WordCompleter(self.commands, ignore_case=True, sentence=True)
        else:
            self.completer = None

    def add_command(self, command: str) -> None:
        """Add a command to completion list."""
        if command not in self.commands:
            self.commands.append(command)
            if PROMPT_TOOLKIT_AVAILABLE:
                self.completer = WordCompleter(self.commands, ignore_case=True, sentence=True)

    def remove_command(self, command: str) -> None:
        """Remove a command from completion list."""
        if command in self.commands:
            self.commands.remove(command)
            if PROMPT_TOOLKIT_AVAILABLE:
                self.completer = WordCompleter(self.commands, ignore_case=True, sentence=True)


# ============================================================================
# Input Validator
# ============================================================================


class InputValidator:
    """Validates terminal input."""

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 10000,
        allow_empty: bool = False,
        custom_validator: Callable[[str], tuple[bool, str]] | None = None,
    ):
        """
        Initialize validator.

        Args:
            min_length: Minimum input length
            max_length: Maximum input length
            allow_empty: Allow empty input
            custom_validator: Custom validation function (returns (is_valid, error_message))
        """
        self.min_length = min_length
        self.max_length = max_length
        self.allow_empty = allow_empty
        self.custom_validator = custom_validator

    def validate(self, text: str) -> tuple[bool, str]:
        """
        Validate input text.

        Args:
            text: Input text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check empty
        if not text.strip():
            if self.allow_empty:
                return True, ""
            return False, "Input cannot be empty"

        # Check length
        if len(text) < self.min_length:
            return False, f"Input too short (minimum {self.min_length} characters)"

        if len(text) > self.max_length:
            return False, f"Input too long (maximum {self.max_length} characters)"

        # Custom validation
        if self.custom_validator:
            return self.custom_validator(text)

        return True, ""


# ============================================================================
# Prompt Toolkit Validator Adapter
# ============================================================================

if PROMPT_TOOLKIT_AVAILABLE:

    class PromptToolkitValidator(Validator):
        """Adapter for InputValidator to work with prompt_toolkit."""

        def __init__(self, input_validator: InputValidator):
            """Initialize with InputValidator."""
            self.input_validator = input_validator

        def validate(self, document: Any) -> None:
            """Validate document."""
            text = document.text
            is_valid, error_msg = self.input_validator.validate(text)

            if not is_valid:
                raise ValidationError(message=error_msg, cursor_position=len(text))


# ============================================================================
# Advanced Input Handler
# ============================================================================


class AdvancedInputHandler:
    """
    Advanced input handler with history, completion, and validation.

    Features:
    - Command history with persistence
    - Tab completion
    - Multi-line input support
    - Input validation
    - Readline-like editing
    - Custom keyboard shortcuts
    - Graceful fallback to basic input
    """

    def __init__(
        self,
        console: Console,
        history_file: Path | None = None,
        enable_completion: bool = True,
        enable_validation: bool = True,
        enable_multiline: bool = False,
        enable_custom_bindings: bool = True,
        validator: InputValidator | None = None,
        completer: CommandCompleter | None = None,
        on_help: Callable[[], None] | None = None,
        on_metrics: Callable[[], None] | None = None,
        on_errors: Callable[[], None] | None = None,
        on_save_session: Callable[[], None] | None = None,
        on_load_session: Callable[[], None] | None = None,
        on_workflow_status: Callable[[], None] | None = None,
        on_export_session: Callable[[], None] | None = None,
        on_show_shortcuts: Callable[[], None] | None = None,
    ):
        """
        Initialize input handler.

        Args:
            console: Rich console for output
            history_file: Path to history file
            enable_completion: Enable tab completion
            enable_validation: Enable input validation
            enable_multiline: Enable multi-line input
            enable_custom_bindings: Enable custom keyboard shortcuts
            validator: Input validator
            completer: Command completer
            on_help: Callback for help shortcut (Ctrl+X H)
            on_metrics: Callback for metrics shortcut (Ctrl+X M)
            on_errors: Callback for errors shortcut (Ctrl+X E)
            on_save_session: Callback for save session shortcut (Ctrl+X S)
            on_load_session: Callback for load session shortcut (Ctrl+X L)
            on_workflow_status: Callback for workflow status shortcut (Ctrl+X W)
            on_export_session: Callback for export session shortcut (Ctrl+X X)
            on_show_shortcuts: Callback for show shortcuts shortcut (Ctrl+X ?)
        """
        self.console = console
        self.history_file = history_file or Path.home() / ".code_agent_history"
        self.enable_completion = enable_completion and PROMPT_TOOLKIT_AVAILABLE
        self.enable_validation = enable_validation
        self.enable_multiline = enable_multiline
        self.enable_custom_bindings = enable_custom_bindings and PROMPT_TOOLKIT_AVAILABLE

        # Create validator and completer
        self.validator = validator or InputValidator(allow_empty=True)
        self.completer = completer or CommandCompleter()

        # Callbacks for custom shortcuts
        self.on_help = on_help
        self.on_metrics = on_metrics
        self.on_errors = on_errors
        self.on_save_session = on_save_session
        self.on_load_session = on_load_session
        self.on_workflow_status = on_workflow_status
        self.on_export_session = on_export_session
        self.on_show_shortcuts = on_show_shortcuts

        # Initialize shortcut registry
        self.shortcut_registry = ShortcutRegistry()

        # Initialize prompt session if available
        self.session: Any | None = None
        if PROMPT_TOOLKIT_AVAILABLE:
            self._init_prompt_session()

        # Fallback mode
        self.use_fallback = not PROMPT_TOOLKIT_AVAILABLE

    def _create_custom_key_bindings(self) -> Any:
        """
        Create custom key bindings for terminal operations.

        Returns:
            KeyBindings instance with custom shortcuts
        """
        if not PROMPT_TOOLKIT_AVAILABLE:
            return None

        bindings = KeyBindings()

        # Register all shortcuts in the registry
        self._register_shortcuts()

        # ===================================================================
        # HELP AND INFORMATION SHORTCUTS
        # ===================================================================

        # Ctrl+X H - Show help
        @bindings.add("c-x", "h")
        def _show_help(event: Any) -> None:
            """Show help when Ctrl+X H is pressed."""
            if self.on_help:
                event.app.current_buffer.text = ""
                self._show_feedback("Ctrl+X H")
                self.on_help()

        # Ctrl+X M - Show metrics
        @bindings.add("c-x", "m")
        def _show_metrics(event: Any) -> None:
            """Show metrics when Ctrl+X M is pressed."""
            if self.on_metrics:
                event.app.current_buffer.text = ""
                self._show_feedback("Ctrl+X M")
                self.on_metrics()

        # Ctrl+X E - Show errors
        @bindings.add("c-x", "e")
        def _show_errors(event: Any) -> None:
            """Show errors when Ctrl+X E is pressed."""
            if self.on_errors:
                event.app.current_buffer.text = ""
                self._show_feedback("Ctrl+X E")
                self.on_errors()

        # Ctrl+X ? - Show shortcuts
        @bindings.add("c-x", "?")
        def _show_shortcuts(event: Any) -> None:
            """Show keyboard shortcuts when Ctrl+X ? is pressed."""
            if self.on_show_shortcuts:
                event.app.current_buffer.text = ""
                self._show_feedback("Ctrl+X ?")
                self.on_show_shortcuts()
            else:
                # Fallback: show shortcuts from registry
                event.app.current_buffer.text = ""
                self.console.print("\n[dim]Keyboard shortcut: Ctrl+X ?[/dim]")
                self.console.print(self.shortcut_registry.generate_help_text())

        # ===================================================================
        # TERMINAL CONTROL SHORTCUTS
        # ===================================================================

        # Ctrl+X C - Clear screen
        @bindings.add("c-x", "c")
        def _clear_screen(event: Any) -> None:
            """Clear screen when Ctrl+X C is pressed."""
            event.app.current_buffer.text = ""
            self.console.clear()
            self._show_feedback("Ctrl+X C", "Screen cleared")

        # ===================================================================
        # SESSION MANAGEMENT SHORTCUTS
        # ===================================================================

        # Ctrl+X S - Save session
        @bindings.add("c-x", "s")
        def _save_session(event: Any) -> None:
            """Save session when Ctrl+X S is pressed."""
            if self.on_save_session:
                event.app.current_buffer.text = ""
                self._show_feedback("Ctrl+X S")
                self.on_save_session()

        # Ctrl+X L - Load session
        @bindings.add("c-x", "l")
        def _load_session(event: Any) -> None:
            """Load session when Ctrl+X L is pressed."""
            if self.on_load_session:
                event.app.current_buffer.text = ""
                self._show_feedback("Ctrl+X L")
                self.on_load_session()

        # Ctrl+X X - Export session
        @bindings.add("c-x", "x")
        def _export_session(event: Any) -> None:
            """Export session when Ctrl+X X is pressed."""
            if self.on_export_session:
                event.app.current_buffer.text = ""
                self._show_feedback("Ctrl+X X")
                self.on_export_session()

        # ===================================================================
        # WORKFLOW SHORTCUTS
        # ===================================================================

        # Ctrl+X W - Show workflow status
        @bindings.add("c-x", "w")
        def _workflow_status(event: Any) -> None:
            """Show workflow status when Ctrl+X W is pressed."""
            if self.on_workflow_status:
                event.app.current_buffer.text = ""
                self._show_feedback("Ctrl+X W")
                self.on_workflow_status()

        return bindings

    def _register_shortcuts(self) -> None:
        """Register all shortcuts in the registry for documentation."""
        # Help and Information
        self.shortcut_registry.register(
            ("c-x", "h"),
            "Show help message",
            category=ShortcutCategory.HELP,
            filter_condition=lambda: self.on_help is not None,
        )
        self.shortcut_registry.register(
            ("c-x", "m"),
            "Show performance metrics",
            category=ShortcutCategory.HELP,
            filter_condition=lambda: self.on_metrics is not None,
        )
        self.shortcut_registry.register(
            ("c-x", "e"),
            "Show error summary",
            category=ShortcutCategory.HELP,
            filter_condition=lambda: self.on_errors is not None,
        )
        self.shortcut_registry.register(("c-x", "?"), "Show all keyboard shortcuts", category=ShortcutCategory.HELP)

        # Terminal Control
        self.shortcut_registry.register(("c-x", "c"), "Clear screen", category=ShortcutCategory.TERMINAL_CONTROL)

        # Session Management
        self.shortcut_registry.register(
            ("c-x", "s"),
            "Save current session",
            category=ShortcutCategory.SESSION_MANAGEMENT,
            filter_condition=lambda: self.on_save_session is not None,
        )
        self.shortcut_registry.register(
            ("c-x", "l"),
            "Load a session",
            category=ShortcutCategory.SESSION_MANAGEMENT,
            filter_condition=lambda: self.on_load_session is not None,
        )
        self.shortcut_registry.register(
            ("c-x", "x"),
            "Export session to file",
            category=ShortcutCategory.SESSION_MANAGEMENT,
            filter_condition=lambda: self.on_export_session is not None,
        )

        # Workflow
        self.shortcut_registry.register(
            ("c-x", "w"),
            "Show workflow status",
            category=ShortcutCategory.WORKFLOW,
            filter_condition=lambda: self.on_workflow_status is not None,
        )

        # Standard shortcuts (provided by prompt_toolkit)
        self.shortcut_registry.register(("c-a",), "Move to beginning of line", category=ShortcutCategory.NAVIGATION)
        self.shortcut_registry.register(("c-e",), "Move to end of line", category=ShortcutCategory.NAVIGATION)
        self.shortcut_registry.register(
            ("c-k",), "Delete from cursor to end of line", category=ShortcutCategory.EDITING
        )
        self.shortcut_registry.register(
            ("c-u",), "Delete from cursor to beginning of line", category=ShortcutCategory.EDITING
        )
        self.shortcut_registry.register(("c-w",), "Delete word backward", category=ShortcutCategory.EDITING)
        self.shortcut_registry.register(("c-l",), "Clear screen", category=ShortcutCategory.TERMINAL_CONTROL)
        self.shortcut_registry.register(("c-c",), "Cancel current input", category=ShortcutCategory.TERMINAL_CONTROL)
        self.shortcut_registry.register(
            ("c-d",), "Exit terminal (on empty line)", category=ShortcutCategory.TERMINAL_CONTROL
        )

    def _show_feedback(self, shortcut: str, message: str | None = None) -> None:
        """
        Show visual feedback for shortcut activation.

        Args:
            shortcut: Shortcut that was activated
            message: Optional custom message
        """
        if message:
            self.console.print(f"\n[dim]Keyboard shortcut: {shortcut} - {message}[/dim]")
        else:
            self.console.print(f"\n[dim]Keyboard shortcut: {shortcut}[/dim]")

    def _init_prompt_session(self) -> None:
        """Initialize prompt_toolkit session."""
        if not PROMPT_TOOLKIT_AVAILABLE:
            return

        # Create history file directory if needed
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # Create session
        kwargs: dict[str, Any] = {
            "history": FileHistory(str(self.history_file)),
            "enable_history_search": True,
            "multiline": self.enable_multiline,
        }

        if self.enable_completion and self.completer.completer:
            kwargs["completer"] = self.completer.completer
            kwargs["complete_while_typing"] = True

        if self.enable_validation:
            kwargs["validator"] = PromptToolkitValidator(self.validator)
            kwargs["validate_while_typing"] = False

        # Add custom key bindings if enabled
        if self.enable_custom_bindings:
            custom_bindings = self._create_custom_key_bindings()
            if custom_bindings:
                kwargs["key_bindings"] = custom_bindings

        self.session = PromptSession(**kwargs)

    def get_input(self, prompt: str = "You", default: str = "", multiline: bool | None = None) -> str:
        """
        Get input from user.

        Args:
            prompt: Prompt text
            default: Default value
            multiline: Override multiline setting

        Returns:
            User input string
        """
        # Use multiline override if provided
        use_multiline = multiline if multiline is not None else self.enable_multiline

        # Use prompt_toolkit if available
        if not self.use_fallback and self.session:
            try:
                # Update session multiline if needed
                if use_multiline != self.enable_multiline:
                    self.session.multiline = use_multiline

                # Format prompt
                formatted_prompt = HTML(f"<ansibrightcyan><b>{prompt}</b></ansibrightcyan> ")

                # Get input
                result = self.session.prompt(formatted_prompt, default=default)

                # Restore original multiline setting
                if use_multiline != self.enable_multiline:
                    self.session.multiline = self.enable_multiline

                return str(result)

            except (KeyboardInterrupt, EOFError):
                raise
            except Exception as e:
                # Fall back to basic input on error
                self.console.print(f"[yellow]⚠️  Advanced input failed: {e}[/yellow]")
                self.console.print("[yellow]Falling back to basic input[/yellow]")
                self.use_fallback = True

        # Fallback to Rich prompt
        return str(
            Prompt.ask(f"[bold cyan]{prompt}[/bold cyan]", console=self.console, default=default if default else "")
        )

    def get_multiline_input(self, prompt: str = "You") -> str:
        """
        Get multi-line input from user.

        Args:
            prompt: Prompt text

        Returns:
            Multi-line user input
        """
        return self.get_input(prompt=prompt, multiline=True)

    def clear_history(self) -> None:
        """Clear input history."""
        if self.history_file.exists():
            try:
                self.history_file.unlink()
                self.console.print("[green]✅ History cleared[/green]")
            except Exception as e:
                self.console.print(f"[red]❌ Failed to clear history: {e}[/red]")

    def get_shortcuts_help(self, category: ShortcutCategory | None = None) -> str:
        """
        Get help text for keyboard shortcuts.

        Args:
            category: Optional category filter

        Returns:
            Formatted help text
        """
        return self.shortcut_registry.generate_help_text(category)

    def get_enabled_shortcuts(self) -> list[Any]:
        """Get list of enabled shortcuts."""
        return self.shortcut_registry.get_enabled()


__all__ = [
    "AdvancedInputHandler",
    "CommandCompleter",
    "InputValidator",
    "PROMPT_TOOLKIT_AVAILABLE",
]
