"""
Terminal input adapters for CodeAgentTerminal.

Provides an optional prompt_toolkit-powered input with history and completion,
falling back to Rich Prompt.ask when unavailable or disabled.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from collections.abc import Iterable

try:  # Optional dependency
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory

    _PTK_AVAILABLE = True
except Exception:  # pragma: no cover
    _PTK_AVAILABLE = False

from rich.prompt import Prompt

from .config import TerminalConfig


class TerminalInput:
    """Unified input provider with optional advanced features."""

    def __init__(self, config: TerminalConfig, commands: Iterable[str] | None = None) -> None:
        self.config = config
        self.commands = list(commands or [])
        self._history: list[str] = []
        self._ptk_session: PromptSession[str] | None = None
        if _PTK_AVAILABLE and self.config.use_prompt_toolkit:
            try:
                completer = WordCompleter(self.commands, ignore_case=True)
                self._ptk_session = PromptSession(
                    history=InMemoryHistory(),
                    completer=completer,
                )
            except Exception:
                self._ptk_session = None

    def get_input(self, label: str = "You") -> str:
        """Get a single line of input using configured provider."""
        prompt_text = f"[bold cyan]{label}[/bold cyan]"

        if self._ptk_session is not None:
            try:
                text: str = self._ptk_session.prompt(
                    message="",
                )
                if text:
                    self._history.append(text)
                return text
            except (KeyboardInterrupt, EOFError):
                return ""
            except Exception:
                # Fallback to Rich
                pass

        # Rich fallback
        try:
            text_result: str = Prompt.ask(prompt_text)
            if text_result:
                self._history.append(text_result)
            return text_result
        except (KeyboardInterrupt, EOFError):
            return ""

    @property
    def history(self) -> list[str]:
        return self._history
