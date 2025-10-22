"""
Code Agent Terminal UI - Interactive Terminal Interface

A sophisticated terminal-based user interface inspired by Claude Code,
providing rich interactive experience with real-time streaming, syntax
highlighting, and comprehensive information display.

Features:
- Real-time streaming responses
- Syntax highlighting for code blocks
- Markdown rendering
- Interactive command history with persistence
- Auto-completion and input validation
- Performance metrics display
- Error recovery visualization
- Session management with auto-save
- Multi-line input support
- Advanced input handling (readline-like)
- Adaptive rendering based on terminal size

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..config.logging import LogFormat, LogLevel
from ..core.agent import CodeAgent
from ..utils.terminal_exec import (
    CommandResult,
    InteractiveTerminalSession,
    run_command,
)

# Import enhanced components
from .input_handler import PROMPT_TOOLKIT_AVAILABLE, AdvancedInputHandler
from .session_manager import TerminalSessionManager
from .streaming import EnhancedStreamingDisplay

PYDANTIC_AI_AVAILABLE = True


# ============================================================================
# Terminal UI State
# ============================================================================


class TerminalUIState:
    """State management for terminal UI."""

    def __init__(self) -> None:
        """Initialize UI state."""
        self.current_status = "idle"
        self.current_prompt = ""
        self.response_buffer = ""
        self.command_history: list[str] = []
        self.history_index = -1
        self.session_start = datetime.now()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.multiline_mode = False
        self.last_execution_time = 0.0


# ============================================================================
# Terminal UI Components
# ============================================================================


class TerminalUIComponents:
    """UI component builders for terminal interface."""

    def __init__(self, console: Console) -> None:
        """Initialize UI components."""
        self.console = console

    def create_header(self, status: str = "idle") -> Panel:
        """Create header panel with status."""
        status_colors = {
            "idle": "green",
            "thinking": "yellow",
            "executing": "blue",
            "streaming": "cyan",
            "error": "red",
        }

        status_text = Text()
        status_text.append("Code Agent Terminal", style="bold magenta")
        status_text.append(" | Status: ", style="dim")
        status_text.append(status.upper(), style=f"bold {status_colors.get(status, 'white')}")

        return Panel(status_text, box=box.DOUBLE, border_style="magenta", padding=(0, 1))

    def create_metrics_panel(
        self, total_requests: int, successful: int, failed: int, input_tokens: int = 0, output_tokens: int = 0
    ) -> Panel:
        """Create metrics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", str(total_requests))
        table.add_row("Successful", str(successful))
        table.add_row("Failed", str(failed))

        if input_tokens > 0 or output_tokens > 0:
            table.add_row("Input Tokens", f"{input_tokens:,}")
            table.add_row("Output Tokens", f"{output_tokens:,}")
            table.add_row("Total Tokens", f"{input_tokens + output_tokens:,}")

        success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
        table.add_row("Success Rate", f"{success_rate:.1f}%")

        return Panel(table, title="[bold cyan]Metrics[/bold cyan]", border_style="cyan", box=box.ROUNDED)

    def create_error_panel(self, error_summary: dict[str, Any]) -> Panel:
        """Create error panel."""
        if error_summary.get("total_errors", 0) == 0:
            return Panel(
                "[green]No errors[/green]", title="[bold red]Errors[/bold red]", border_style="green", box=box.ROUNDED
            )

        table = Table(show_header=True, box=None)
        table.add_column("Type", style="red")
        table.add_column("Message", style="yellow")
        table.add_column("Category", style="cyan")

        for error in error_summary.get("recent_errors", [])[:3]:
            table.add_row(
                error.get("type", "Unknown"), error.get("message", "")[:50] + "...", error.get("category", "unknown")
            )

        return Panel(
            table,
            title=f"[bold red]Errors ({error_summary['total_errors']})[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        )

    def create_tool_execution_panel(self, tool_name: str, status: str = "running") -> Panel:
        """Create tool execution panel."""
        status_icons = {"running": "⏳", "success": "✅", "failed": "❌"}

        icon = status_icons.get(status, "❓")
        text = Text()
        text.append(f"{icon} ", style="bold")
        text.append(tool_name, style="cyan")
        text.append(f" [{status}]", style="dim")

        return Panel(text, title="[bold blue]Tool Execution[/bold blue]", border_style="blue", box=box.ROUNDED)

    def render_markdown(self, content: str) -> Markdown:
        """Render markdown content."""
        return Markdown(content)

    def render_code(self, code: str, language: str = "python") -> Syntax:
        """Render code with syntax highlighting."""
        return Syntax(code, language, theme="monokai", line_numbers=True, word_wrap=True)

    def create_response_panel(self, content: str, streaming: bool = False) -> Panel:
        """Create response panel."""
        title = "[bold green]Response[/bold green]"
        if streaming:
            title = "[bold cyan]Response (streaming...)[/bold cyan]"

        # Try to render as markdown
        rendered: Markdown | Text
        try:
            rendered = self.render_markdown(content)
        except Exception:
            rendered = Text(content)

        return Panel(
            rendered, title=title, border_style="green" if not streaming else "cyan", box=box.ROUNDED, padding=(1, 2)
        )

    def create_prompt_panel(self, prompt: str) -> Panel:
        """Create prompt panel."""
        return Panel(
            Text(prompt, style="bold white"),
            title="[bold yellow]Your Prompt[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def create_stdout_panel(self, content: str) -> Panel:
        """Create stdout panel for process output."""
        return Panel(
            Text(content or "", style="white"),
            title="[bold blue]STDOUT[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 1),
        )

    def create_stderr_panel(self, content: str) -> Panel:
        """Create stderr panel for process errors."""
        return Panel(
            Text(content or "", style="yellow"),
            title="[bold red]STDERR[/bold red]",
            border_style="red",
            box=box.ROUNDED,
            padding=(1, 1),
        )

    def create_process_summary_panel(self, result: CommandResult, *, cwd: str | None = None) -> Panel:
        """Create a summary panel for a finished process."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Exit Code", str(result.exit_code if result.exit_code is not None else ""))
        table.add_row("Duration", f"{result.duration_s:.2f}s")
        table.add_row("Timed Out", "Yes" if result.timed_out else "No")
        if cwd:
            table.add_row("CWD", cwd)
        if result.error:
            table.add_row("Error", result.error)
        return Panel(
            table, title="[bold magenta]Process Summary[/bold magenta]", border_style="magenta", box=box.ROUNDED
        )


# ============================================================================
# Main Terminal UI Class
# ============================================================================


class CodeAgentTerminal:
    """
    Interactive terminal UI for Code Agent.

    Provides a rich, interactive command-line interface with:
    - Real-time streaming responses
    - Syntax highlighting
    - Markdown rendering
    - Performance metrics
    - Error visualization
    - Session management
    """

    def __init__(
        self,
        model: str = "openai:gpt-4",
        log_level: LogLevel = LogLevel.INFO,
        enable_workflow: bool = True,
        enable_advanced_input: bool = True,
        enable_session_manager: bool = True,
        enable_enhanced_streaming: bool = True,
    ):
        """
        Initialize terminal UI.

        Args:
            model: Model to use for the agent
            log_level: Logging level
            enable_workflow: Enable workflow orchestration
            enable_advanced_input: Enable advanced input handling (prompt_toolkit)
            enable_session_manager: Enable session persistence
            enable_enhanced_streaming: Enable enhanced streaming display
        """
        self.console = Console()
        self.components = TerminalUIComponents(self.console)
        self.state = TerminalUIState()

        # Create agent
        self.agent = CodeAgent(
            model=model, log_level=log_level, log_format=LogFormat.HUMAN, enable_workflow=enable_workflow
        )

        # Session file (legacy)
        self.session_file = Path(".code_agent_session.txt")

        # Enhanced features
        self.enable_advanced_input = enable_advanced_input and PROMPT_TOOLKIT_AVAILABLE
        self.enable_session_manager = enable_session_manager
        self.enable_enhanced_streaming = enable_enhanced_streaming

        # Initialize advanced input handler
        if self.enable_advanced_input:
            self.input_handler = AdvancedInputHandler(
                console=self.console,
                enable_completion=True,
                enable_validation=True,
                enable_multiline=False,
                enable_custom_bindings=True,
                on_help=self.display_help,
                on_metrics=self.display_metrics,
                on_errors=self.display_errors,
                on_save_session=self._handle_save_session_shortcut,
                on_load_session=self._handle_load_session_shortcut,
                on_workflow_status=self.display_workflow_status,
                on_export_session=self._handle_export_session_shortcut,
                on_show_shortcuts=self.display_shortcuts,
            )
        else:
            self.input_handler = None  # type: ignore[assignment]

        # Initialize session manager
        if self.enable_session_manager:
            self.session_manager = TerminalSessionManager(console=self.console, auto_save=True, auto_save_interval=60)
        else:
            self.session_manager = None  # type: ignore[assignment]

        # Initialize enhanced streaming display
        if self.enable_enhanced_streaming:
            self.streaming_display = EnhancedStreamingDisplay(
                console=self.console, show_progress=True, show_stats=False, adaptive_rendering=True
            )
        else:
            self.streaming_display = None  # type: ignore[assignment]

        # OS command/session runtime state
        self.cwd: Path = Path.cwd()
        self.env_overrides: dict[str, str] = {}
        self.sessions: dict[str, InteractiveTerminalSession] = {}
        self._next_session_id: int = 1
        self.default_timeout: float = 30.0  # Default timeout for shell commands

    def display_welcome(self) -> None:
        """Display welcome message."""
        # Build feature list
        features = []
        if self.enable_advanced_input:
            features.append("  • Advanced input with history & completion ✨")
        if self.enable_session_manager:
            features.append("  • Auto-save sessions 💾")
        if self.enable_enhanced_streaming:
            features.append("  • Enhanced streaming display 🚀")

        welcome_items = [
            Text("Welcome to Code Agent Terminal!", style="bold magenta", justify="center"),
            Text(""),
            Text("An interactive AI-powered code analysis assistant", style="dim", justify="center"),
            Text(""),
        ]

        if features:
            welcome_items.extend(
                [
                    Text("Enhanced Features:", style="bold green"),
                    *[Text(f, style="dim green") for f in features],
                    Text(""),
                ]
            )

        welcome_items.extend(
            [
                Text("Commands:", style="bold cyan"),
                Text("  • Type your question or request", style="dim"),
                Text("  • 'exit' or 'quit' to exit", style="dim"),
                Text("  • 'clear' to clear screen", style="dim"),
                Text("  • 'metrics' to show detailed metrics", style="dim"),
                Text("  • 'errors' to show error history", style="dim"),
                Text("  • '!<cmd>' to run a shell command", style="dim"),
                Text("  • ':shell' to start an interactive shell session", style="dim"),
                Text("  • ':cwd [path]' and ':timeout [secs]' to adjust execution", style="dim"),
                Text("  • 'help' for more commands", style="dim"),
            ]
        )

        welcome = Panel(Group(*welcome_items), box=box.DOUBLE, border_style="magenta", padding=(1, 2))
        self.console.print(welcome)
        self.console.print()

    def display_help(self) -> None:
        """Display help message."""
        help_table = Table(show_header=True, box=box.ROUNDED, border_style="cyan")
        help_table.add_column("Command", style="cyan", no_wrap=True)
        help_table.add_column("Description", style="white")

        commands = [
            ("exit, quit", "Exit the terminal"),
            ("clear", "Clear the screen"),
            ("metrics", "Show detailed performance metrics"),
            ("errors", "Show error history and diagnostics"),
            ("workflow", "Show workflow status (if enabled)"),
            ("history", "Show command history"),
            ("export [file]", "Export session to file (default: session.md)"),
            ("!<cmd>", "Run a shell command and show stdout/stderr"),
            (":shell [cmd]", "Start an interactive shell session (default shell if omitted)"),
            (":procs", "List interactive sessions"),
            (":attach <id>", "Attach to a session for I/O"),
            (":kill <id>", "Terminate a session"),
            (":timeout [secs]", "Get or set default command timeout"),
            (":cwd [path]", "Get or set working directory"),
            (":env KEY=VAL | --list", "Set or list environment overrides"),
            ("help", "Show this help message"),
            ("shortcuts", "Show keyboard shortcuts"),
        ]

        # Add enhanced commands if available
        if self.enable_session_manager:
            commands.extend(
                [
                    ("save", "Save current session"),
                    ("load <id>", "Load a previous session"),
                    ("sessions", "List available sessions"),
                ]
            )

        if self.enable_advanced_input:
            commands.append(("multiline", "Toggle multi-line input mode"))

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        # Add keyboard shortcuts hint
        if self.enable_advanced_input:
            self.console.print(
                Panel(help_table, title="[bold cyan]Available Commands[/bold cyan]", border_style="cyan")
            )
            self.console.print("\n[dim]💡 Tip: Press [bold]Ctrl+X ?[/bold] to see all keyboard shortcuts[/dim]\n")
        else:
            self.console.print(
                Panel(help_table, title="[bold cyan]Available Commands[/bold cyan]", border_style="cyan")
            )

    def display_metrics(self) -> None:
        """Display detailed metrics."""
        # Get agent metrics
        agent_metrics = self.agent.state.logger.get_metrics_summary()

        # Create comprehensive metrics table
        table = Table(show_header=True, box=box.ROUNDED, border_style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        # Session metrics
        session_duration = datetime.now() - self.state.session_start
        table.add_row("Session Duration", str(session_duration).split(".")[0])
        table.add_row("Total Requests", str(self.state.total_requests))
        table.add_row("Successful", str(self.state.successful_requests))
        table.add_row("Failed", str(self.state.failed_requests))

        # Agent metrics
        table.add_row("", "")  # Separator
        table.add_row("[bold]Agent Metrics[/bold]", "")
        table.add_row("Total Operations", str(agent_metrics.get("total_operations", 0)))
        table.add_row("Successful Ops", str(agent_metrics.get("successful", 0)))
        table.add_row("Failed Ops", str(agent_metrics.get("failed", 0)))

        avg_duration = agent_metrics.get("average_duration", 0)
        table.add_row("Avg Duration", f"{avg_duration:.2f}s")

        total_tokens = agent_metrics.get("total_input_tokens", 0) + agent_metrics.get("total_output_tokens", 0)
        table.add_row("Total Tokens", f"{total_tokens:,}")

        self.console.print(Panel(table, title="[bold green]Detailed Metrics[/bold green]", border_style="green"))

    def display_errors(self) -> None:
        """Display error history."""
        error_summary = self.agent.state.get_error_summary()

        if error_summary["total_errors"] == 0:
            self.console.print(
                Panel(
                    "[green]✅ No errors recorded[/green]",
                    title="[bold green]Error History[/bold green]",
                    border_style="green",
                )
            )
            return

        # Create error table
        table = Table(show_header=True, box=box.ROUNDED, border_style="red")
        table.add_column("Time", style="dim")
        table.add_column("Type", style="red")
        table.add_column("Message", style="yellow")
        table.add_column("Category", style="cyan")
        table.add_column("Severity", style="magenta")

        for error in error_summary.get("recent_errors", [])[:10]:
            table.add_row(
                error.get("timestamp", "")[:19],
                error.get("type", "Unknown"),
                error.get("message", "")[:60] + ("..." if len(error.get("message", "")) > 60 else ""),
                error.get("category", "unknown"),
                error.get("severity", "unknown"),
            )

        # Category breakdown
        by_category = error_summary.get("by_category", {})
        category_text = Text()
        category_text.append("\nBy Category: ", style="bold")
        for cat, count in by_category.items():
            category_text.append(f"{cat}={count} ", style="cyan")

        self.console.print(
            Panel(
                Group(table, category_text),
                title=f"[bold red]Error History ({error_summary['total_errors']} total)[/bold red]",
                border_style="red",
            )
        )

    def display_workflow_status(self) -> None:
        """Display workflow status."""
        if not self.agent.state.workflow_orchestrator:
            self.console.print(
                Panel(
                    "[yellow]Workflow orchestration is not enabled[/yellow]",
                    title="[bold blue]Workflow Status[/bold blue]",
                    border_style="yellow",
                )
            )
            return

        workflow = self.agent.state.workflow_orchestrator
        status = workflow.get_workflow_status()

        # Create status table
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Current State", status.get("current_state", "unknown"))
        table.add_row("Total Checkpoints", str(status.get("total_checkpoints", 0)))
        table.add_row("Fix Attempts", str(status.get("total_fix_attempts", 0)))
        table.add_row("Successful Fixes", str(status.get("successful_fixes", 0)))

        self.console.print(Panel(table, title="[bold blue]Workflow Status[/bold blue]", border_style="blue"))

    def display_history(self) -> None:
        """Display command history."""
        if not self.state.command_history:
            self.console.print("[dim]No command history[/dim]")
            return

        table = Table(show_header=True, box=box.ROUNDED, border_style="cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Command", style="white")

        for i, cmd in enumerate(self.state.command_history[-20:], 1):
            table.add_row(str(i), cmd[:100] + ("..." if len(cmd) > 100 else ""))

        self.console.print(Panel(table, title="[bold cyan]Command History[/bold cyan]", border_style="cyan"))

    def export_session(self, filename: str = "session.md") -> None:
        """Export session to markdown file."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# Code Agent Session\n\n")
                f.write(f"**Date**: {self.state.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Metrics\n\n")
                f.write(f"- Total Requests: {self.state.total_requests}\n")
                f.write(f"- Successful: {self.state.successful_requests}\n")
                f.write(f"- Failed: {self.state.failed_requests}\n\n")
                f.write("## Command History\n\n")
                for i, cmd in enumerate(self.state.command_history, 1):
                    f.write(f"{i}. {cmd}\n")

            self.console.print(f"[green]✅ Session exported to {filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Failed to export session: {e}[/red]")

    def display_shortcuts(self) -> None:
        """Display keyboard shortcuts."""
        if not self.enable_advanced_input or not self.input_handler:
            self.console.print("[yellow]Keyboard shortcuts are only available in Advanced Input Mode[/yellow]")
            self.console.print("[dim]Install prompt_toolkit to enable: pip install prompt_toolkit[/dim]")
            return

        # Get shortcuts help text
        help_text = self.input_handler.get_shortcuts_help()

        # Display in a panel
        self.console.print(
            Panel(
                Text(help_text, style="white"),
                title="[bold cyan]Keyboard Shortcuts[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

        # Show additional tips
        self.console.print("\n[dim]💡 Tips:[/dim]")
        self.console.print("[dim]  • All Ctrl+X shortcuts require pressing Ctrl+X first, then the second key[/dim]")
        self.console.print("[dim]  • Standard shortcuts (Ctrl+A, Ctrl+E, etc.) work immediately[/dim]")
        self.console.print("[dim]  • Use ↑/↓ arrows to navigate command history[/dim]")
        self.console.print("[dim]  • Press Tab for auto-completion[/dim]\n")

    def _handle_save_session_shortcut(self) -> None:
        """Handle save session keyboard shortcut."""
        if self.session_manager:
            filepath = self.session_manager.save_session()
            if filepath:
                self.console.print(f"[green]✅ Session saved to {filepath}[/green]")
        else:
            self.console.print("[yellow]Session management is not enabled[/yellow]")

    def _handle_load_session_shortcut(self) -> None:
        """Handle load session keyboard shortcut."""
        if self.session_manager:
            # Show available sessions
            sessions = self.session_manager.list_sessions()
            if sessions:
                self.console.print("[bold cyan]Available Sessions:[/bold cyan]")
                for i, session_id in enumerate(sessions[:10], 1):
                    self.console.print(f"  {i}. {session_id}")
                self.console.print("\n[dim]Use 'load <session_id>' command to load a session[/dim]")
            else:
                self.console.print("[dim]No saved sessions available[/dim]")
        else:
            self.console.print("[yellow]Session management is not enabled[/yellow]")

    def _handle_export_session_shortcut(self) -> None:
        """Handle export session keyboard shortcut."""
        # Use default filename
        self.export_session("session.md")

    def _run_shell_command(self, cmd: str) -> None:
        """Execute a one-shot shell command and display stdout/stderr panels."""
        self.console.print(
            Panel(Text(cmd, style="white"), title="[bold yellow]Command[/bold yellow]", border_style="yellow")
        )
        # Execute synchronously to simplify UI; leverage default timeout
        try:
            result = run_command(cmd, timeout=self.default_timeout, shell=True, cwd=self.cwd, env=self.env_overrides)
        except Exception as e:
            self.console.print(f"[red]Execution failed: {e}[/red]")
            return
        # Render outputs
        self.console.print(self.components.create_stdout_panel(result.stdout))
        if result.stderr:
            self.console.print(self.components.create_stderr_panel(result.stderr))
        self.console.print(self.components.create_process_summary_panel(result, cwd=str(self.cwd)))

    def handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Args:
            command: Command to handle

        Returns:
            True if should continue, False if should exit
        """
        cmd_lower = command.lower().strip()

        # Exit commands
        if cmd_lower in ("exit", "quit", "q"):
            # Save session before exit
            if self.session_manager:
                self.session_manager.save_session()
                self.console.print("[green]✅ Session saved[/green]")
            self.console.print("[yellow]Goodbye! 👋[/yellow]")
            return False

        # Clear screen
        if cmd_lower == "clear":
            self.console.clear()
            self.display_welcome()
            return True

        # Help
        if cmd_lower == "help":
            self.display_help()
            return True

        # Shortcuts
        if cmd_lower == "shortcuts":
            self.display_shortcuts()
            return True

        # Metrics
        if cmd_lower == "metrics":
            self.display_metrics()
            return True

        # Errors
        if cmd_lower == "errors":
            self.display_errors()
            return True

        # Workflow
        if cmd_lower == "workflow":
            self.display_workflow_status()
            return True

        # History
        if cmd_lower == "history":
            self.display_history()
            return True

        # Export
        if cmd_lower.startswith("export"):
            parts = cmd_lower.split()
            filename = parts[1] if len(parts) > 1 else "session.md"
            self.export_session(filename)
            return True

        # Session management commands
        if self.session_manager:
            if cmd_lower == "save":
                filepath = self.session_manager.save_session()
                if filepath:
                    self.console.print(f"[green]✅ Session saved to {filepath}[/green]")
                return True

            if cmd_lower.startswith("load"):
                parts = cmd_lower.split()
                if len(parts) > 1:
                    session_id = parts[1]
                    self.session_manager.load_session(session_id)
                else:
                    self.console.print("[yellow]Usage: load <session_id>[/yellow]")
                return True

            if cmd_lower == "sessions":
                sessions = self.session_manager.list_sessions()
                if sessions:
                    self.console.print("[bold cyan]Available Sessions:[/bold cyan]")
                    for session_id in sessions[:10]:
                        self.console.print(f"  • {session_id}")
                else:
                    self.console.print("[dim]No saved sessions[/dim]")
                return True

        # One-shot shell command
        if command.strip().startswith("!"):
            one = command.strip()[1:].strip()
            if not one:
                self.console.print("[yellow]Usage: !<command>[/yellow]")
                return True
            self._run_shell_command(one)
            return True

        # Timeout get/set
        if cmd_lower.startswith(":timeout"):
            parts = command.strip().split()
            if len(parts) == 1:
                self.console.print(f"[dim]Default timeout: {self.default_timeout}s[/dim]")
            else:
                try:
                    self.default_timeout = float(parts[1])
                    self.console.print(f"[green]Timeout set to {self.default_timeout:.2f}s[/green]")
                except Exception:
                    self.console.print("[red]Invalid timeout value[/red]")
            return True

        # CWD get/set
        if cmd_lower.startswith(":cwd"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) == 1:
                self.console.print(f"[dim]CWD: {self.cwd}[/dim]")
            else:
                new_path = Path(parts[1]).expanduser().resolve()
                if new_path.exists() and new_path.is_dir():
                    self.cwd = new_path
                    self.console.print(f"[green]CWD set to {self.cwd}[/green]")
                else:
                    self.console.print("[red]Invalid directory[/red]")
            return True

        # ENV set/list
        if cmd_lower.startswith(":env"):
            args = command.strip().split()[1:]
            if not args or args == ["--list"]:
                if not self.env_overrides:
                    self.console.print("[dim](no env overrides)[/dim]")
                else:
                    for k, v in self.env_overrides.items():
                        self.console.print(f"[cyan]{k}[/cyan]=[white]{v}[/white]")
            else:
                for item in args:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        self.env_overrides[k] = v
                self.console.print("[green]Environment overrides updated[/green]")
            return True

        # Interactive sessions: list/start/attach/kill
        if cmd_lower.startswith(":procs"):
            if not self.sessions:
                self.console.print("[dim](no interactive sessions)\n[/dim]")
                return True
            table = Table(show_header=True, box=box.ROUNDED, border_style="blue")
            table.add_column("ID", style="cyan")
            table.add_column("Command", style="white")
            table.add_column("Running", style="green")
            for sid, sess in self.sessions.items():
                table.add_row(sid, str(sess.command), "yes" if sess.is_running else "no")
            self.console.print(Panel(table, title="[bold blue]Sessions[/bold blue]", border_style="blue"))
            return True

        if cmd_lower.startswith(":shell"):
            parts = command.strip().split(maxsplit=1)
            launch_cmd = (
                parts[1]
                if len(parts) > 1
                else ("pwsh" if sys.platform.startswith("win") else os.environ.get("SHELL", "/bin/sh"))
            )
            sid = str(self._next_session_id)
            self._next_session_id += 1
            sess = InteractiveTerminalSession(launch_cmd)
            self.sessions[sid] = sess
            try:
                asyncio.run(sess.start(cwd=self.cwd, env=self.env_overrides, shell=True))
                self.console.print(f"[green]Started session {sid}: {launch_cmd}[/green]")
            except Exception as e:
                self.console.print(f"[red]Failed to start session: {e}[/red]")
            return True

        if cmd_lower.startswith(":kill"):
            parts = command.strip().split()
            if len(parts) < 2:
                self.console.print("[yellow]Usage: :kill <id>[/yellow]")
                return True
            sid = parts[1]
            sess_kill: InteractiveTerminalSession | None = self.sessions.get(sid)
            if not sess_kill:
                self.console.print("[red]Session not found[/red]")
                return True
            try:
                asyncio.run(sess_kill.terminate())
                self.console.print(f"[green]Terminated session {sid}[/green]")
            except Exception as e:
                self.console.print(f"[red]Failed to terminate: {e}[/red]")
            return True

        if cmd_lower.startswith(":attach"):
            parts = command.strip().split()
            if len(parts) < 2:
                self.console.print("[yellow]Usage: :attach <id>[/yellow]")
                return True
            sid = parts[1]
            sess_attach: InteractiveTerminalSession | None = self.sessions.get(sid)
            if not sess_attach or not sess_attach.is_running:
                self.console.print("[red]Session not running or not found[/red]")
                return True
            self.console.print(f"[cyan]Attached to session {sid}. Type 'exit' to detach.[/cyan]")
            while True:
                try:
                    line = Prompt.ask("[bold yellow]session>[/bold yellow]")
                except (KeyboardInterrupt, EOFError):
                    break
                if line.strip().lower() in ("exit", ":q", ":quit"):
                    break
                try:
                    asyncio.run(sess.send(line + "\n"))
                except Exception as e:
                    self.console.print(f"[red]Send failed: {e}[/red]")
                    break
            return True

        # Multi-line mode toggle
        if cmd_lower == "multiline" and self.input_handler:
            self.state.multiline_mode = not self.state.multiline_mode
            status = "enabled" if self.state.multiline_mode else "disabled"
            self.console.print(f"[cyan]Multi-line mode {status}[/cyan]")
            return True

        return True

    async def process_streaming_response(self, prompt: str) -> None:
        """
        Process streaming response from agent.

        Args:
            prompt: User prompt
        """
        self.state.current_status = "streaming"
        self.state.response_buffer = ""
        start_time = time.time()

        try:
            # Display prompt
            self.console.print(self.components.create_prompt_panel(prompt))
            self.console.print()

            # Use enhanced streaming if available
            if self.streaming_display:
                self.state.response_buffer = await self.streaming_display.stream_text(
                    self.agent.run_stream(prompt), title="Response"
                )
            else:
                # Fallback to basic streaming
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("[cyan]Streaming response...", total=None)

                    # Stream response
                    response_panel = None
                    async for chunk in self.agent.run_stream(prompt):
                        self.state.response_buffer += chunk

                        # Update display
                        if response_panel:
                            self.console.clear()

                        response_panel = self.components.create_response_panel(
                            self.state.response_buffer, streaming=True
                        )
                        self.console.print(response_panel)

                    progress.update(task, completed=True, description="[green]✅ Complete")

                # Final display
                self.console.clear()
                self.console.print(self.components.create_prompt_panel(prompt))
                self.console.print()
                self.console.print(self.components.create_response_panel(self.state.response_buffer, streaming=False))

            self.state.successful_requests += 1
            self.state.last_execution_time = time.time() - start_time

            # Add to session manager
            if self.session_manager:
                self.session_manager.add_entry(
                    prompt=prompt,
                    response=self.state.response_buffer,
                    status="success",
                    execution_time=self.state.last_execution_time,
                )

        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
            self.state.failed_requests += 1

            # Add interrupted entry to session
            if self.session_manager:
                self.session_manager.add_entry(
                    prompt=prompt,
                    response=self.state.response_buffer,
                    status="interrupted",
                    execution_time=time.time() - start_time,
                )
        except Exception as e:
            self.console.print(f"\n[red]❌ Error: {e}[/red]")
            self.state.failed_requests += 1

            # Add failed entry to session
            if self.session_manager:
                self.session_manager.add_entry(
                    prompt=prompt, response=str(e), status="failed", execution_time=time.time() - start_time
                )
        finally:
            self.state.current_status = "idle"

    def process_sync_response(self, prompt: str) -> None:
        """
        Process synchronous response from agent.

        Args:
            prompt: User prompt
        """
        self.state.current_status = "thinking"
        start_time = time.time()

        try:
            # Display prompt
            self.console.print(self.components.create_prompt_panel(prompt))
            self.console.print()

            # Show progress
            with self.console.status("[cyan]Thinking...", spinner="dots"):
                result = self.agent.run_sync(prompt)

            # Display response
            response_text = str(result.output) if hasattr(result, "output") else str(result)
            self.console.print(self.components.create_response_panel(response_text))

            self.state.successful_requests += 1
            self.state.last_execution_time = time.time() - start_time

            # Add to session manager
            if self.session_manager:
                self.session_manager.add_entry(
                    prompt=prompt,
                    response=response_text,
                    status="success",
                    execution_time=self.state.last_execution_time,
                )

        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
            self.state.failed_requests += 1

            # Add interrupted entry to session
            if self.session_manager:
                self.session_manager.add_entry(
                    prompt=prompt, response="", status="interrupted", execution_time=time.time() - start_time
                )
        except Exception as e:
            self.console.print(f"\n[red]❌ Error: {e}[/red]")

            # Show error details if available
            error_summary = self.agent.state.get_error_summary()
            if error_summary.get("total_errors", 0) > 0:
                self.console.print(self.components.create_error_panel(error_summary))

            self.state.failed_requests += 1

            # Add failed entry to session
            if self.session_manager:
                self.session_manager.add_entry(
                    prompt=prompt, response=str(e), status="failed", execution_time=time.time() - start_time
                )
        finally:
            self.state.current_status = "idle"

    def run(self, use_streaming: bool = False) -> None:
        """
        Run the interactive terminal.

        Args:
            use_streaming: Use streaming mode for responses
        """
        # Clear screen and show welcome
        self.console.clear()
        self.display_welcome()

        # Main loop
        while True:
            try:
                # Show status bar
                self.console.print()
                self.console.rule(style="dim")

                # Get user input using advanced handler if available
                if self.input_handler:
                    try:
                        prompt_text = self.input_handler.get_input(prompt="You", multiline=self.state.multiline_mode)
                    except (KeyboardInterrupt, EOFError):
                        raise
                    except Exception as e:
                        # Fallback to basic input on error
                        self.console.print(f"[yellow]⚠️  Input error: {e}[/yellow]")
                        prompt_text = Prompt.ask("[bold cyan]You[/bold cyan]", console=self.console)
                else:
                    # Use basic Rich prompt
                    prompt_text = Prompt.ask("[bold cyan]You[/bold cyan]", console=self.console)

                # Skip empty input
                if not prompt_text.strip():
                    continue

                # Add to history
                self.state.command_history.append(prompt_text)
                self.state.total_requests += 1

                # Handle special commands
                if not self.handle_command(prompt_text):
                    break

                # Skip if it was a command
                lower = prompt_text.lower().strip()
                if lower in (
                    "help",
                    "shortcuts",
                    "metrics",
                    "errors",
                    "workflow",
                    "history",
                    "clear",
                    "save",
                    "sessions",
                    "multiline",
                ) or lower.startswith(("export", "load", "!", ":")):
                    continue

                # Process request
                self.console.print()

                if use_streaming:
                    # Use async streaming
                    asyncio.run(self.process_streaming_response(prompt_text))
                else:
                    # Use sync mode
                    self.process_sync_response(prompt_text)

                # Show mini metrics
                self.console.print()
                metrics_text = (
                    f"[dim]Requests: {self.state.total_requests} | "
                    f"Success: {self.state.successful_requests} | "
                    f"Failed: {self.state.failed_requests}"
                )
                if self.state.last_execution_time > 0:
                    metrics_text += f" | Last: {self.state.last_execution_time:.2f}s"
                metrics_text += "[/dim]"
                self.console.print(metrics_text)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' or 'quit' to exit[/yellow]")
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {e}[/red]")
                continue

        # Cleanup - save session
        if self.session_manager:
            filepath = self.session_manager.save_session()
            if filepath:
                self.console.print(f"[green]Session saved to {filepath}[/green]")

        self.console.print("\n[green]Session ended. Goodbye! 👋[/green]")

    async def run_async(self) -> None:
        """Run the interactive terminal in async mode with streaming."""
        # Clear screen and show welcome
        self.console.clear()
        self.display_welcome()

        # Main loop
        while True:
            try:
                # Show status bar
                self.console.print()
                self.console.rule(style="dim")

                # Get user input using advanced handler if available
                if self.input_handler:
                    try:
                        prompt_text = self.input_handler.get_input(prompt="You", multiline=self.state.multiline_mode)
                    except (KeyboardInterrupt, EOFError):
                        raise
                    except Exception as e:
                        # Fallback to basic input on error
                        self.console.print(f"[yellow]⚠️  Input error: {e}[/yellow]")
                        prompt_text = Prompt.ask("[bold cyan]You[/bold cyan]", console=self.console)
                else:
                    # Use basic Rich prompt (sync, but that's ok for terminal input)
                    prompt_text = Prompt.ask("[bold cyan]You[/bold cyan]", console=self.console)

                # Skip empty input
                if not prompt_text.strip():
                    continue

                # Add to history
                self.state.command_history.append(prompt_text)
                self.state.total_requests += 1

                # Handle special commands
                if not self.handle_command(prompt_text):
                    break

                # Skip if it was a command
                lower = prompt_text.lower().strip()
                if lower in (
                    "help",
                    "shortcuts",
                    "metrics",
                    "errors",
                    "workflow",
                    "history",
                    "clear",
                    "save",
                    "sessions",
                    "multiline",
                ) or lower.startswith(("export", "load", "!", ":")):
                    continue

                # Process request with streaming
                self.console.print()
                await self.process_streaming_response(prompt_text)

                # Show mini metrics
                self.console.print()
                metrics_text = (
                    f"[dim]Requests: {self.state.total_requests} | "
                    f"Success: {self.state.successful_requests} | "
                    f"Failed: {self.state.failed_requests}"
                )
                if self.state.last_execution_time > 0:
                    metrics_text += f" | Last: {self.state.last_execution_time:.2f}s"
                metrics_text += "[/dim]"
                self.console.print(metrics_text)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' or 'quit' to exit[/yellow]")
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {e}[/red]")
                continue

        # Cleanup - save session
        if self.session_manager:
            filepath = self.session_manager.save_session()
            if filepath:
                self.console.print(f"[green]Session saved to {filepath}[/green]")

        self.console.print("\n[green]Session ended. Goodbye! 👋[/green]")


# ============================================================================
# Convenience Functions
# ============================================================================


def launch_terminal(
    model: str = "openai:gpt-4",
    log_level: LogLevel = LogLevel.INFO,
    enable_workflow: bool = True,
    use_streaming: bool = False,
    enable_advanced_input: bool = True,
    enable_session_manager: bool = True,
    enable_enhanced_streaming: bool = True,
) -> None:
    """
    Launch the Code Agent terminal UI.

    Args:
        model: Model to use for the agent
        log_level: Logging level
        enable_workflow: Enable workflow orchestration
        use_streaming: Use streaming mode for responses
        enable_advanced_input: Enable advanced input handling (requires prompt_toolkit)
        enable_session_manager: Enable session persistence and auto-save
        enable_enhanced_streaming: Enable enhanced streaming display

    Example:
        >>> from code_agent.terminal_ui import launch_terminal
        >>> launch_terminal(use_streaming=True)

        >>> # With all enhancements
        >>> launch_terminal(
        ...     use_streaming=True,
        ...     enable_advanced_input=True,
        ...     enable_session_manager=True,
        ...     enable_enhanced_streaming=True
        ... )
    """
    terminal = CodeAgentTerminal(
        model=model,
        log_level=log_level,
        enable_workflow=enable_workflow,
        enable_advanced_input=enable_advanced_input,
        enable_session_manager=enable_session_manager,
        enable_enhanced_streaming=enable_enhanced_streaming,
    )
    terminal.run(use_streaming=use_streaming)


async def launch_terminal_async(
    model: str = "openai:gpt-4",
    log_level: LogLevel = LogLevel.INFO,
    enable_workflow: bool = True,
    enable_advanced_input: bool = True,
    enable_session_manager: bool = True,
    enable_enhanced_streaming: bool = True,
) -> None:
    """
    Launch the Code Agent terminal UI in async mode with streaming.

    Args:
        model: Model to use for the agent
        log_level: Logging level
        enable_workflow: Enable workflow orchestration
        enable_advanced_input: Enable advanced input handling (requires prompt_toolkit)
        enable_session_manager: Enable session persistence and auto-save
        enable_enhanced_streaming: Enable enhanced streaming display

    Example:
        >>> import asyncio
        >>> from code_agent.terminal_ui import launch_terminal_async
        >>> asyncio.run(launch_terminal_async())
    """
    terminal = CodeAgentTerminal(
        model=model,
        log_level=log_level,
        enable_workflow=enable_workflow,
        enable_advanced_input=enable_advanced_input,
        enable_session_manager=enable_session_manager,
        enable_enhanced_streaming=enable_enhanced_streaming,
    )
    await terminal.run_async()


__all__ = [
    "CodeAgentTerminal",
    "TerminalUIState",
    "TerminalUIComponents",
    "launch_terminal",
    "launch_terminal_async",
]
