"""
Terminal UI Configuration

Configuration objects and platform defaults for Code Agent terminal UI.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TerminalConfig:
    """
    Configuration for CodeAgentTerminal behavior.

    All fields are optional and default to safe, backward-compatible values.
    """

    # Input
    use_prompt_toolkit: bool = False
    multiline_input: bool = False

    # Process execution
    enable_shell_commands: bool = True
    enable_interactive_sessions: bool = True
    default_timeout: float | None = 120.0
    default_shell: str | None = None  # auto-detect if None

    # Rendering/UX
    scrollback_limit: int | None = 5000
    refresh_per_second: int = 10

    # Session
    session_file: Path | None = None


def detect_default_shell() -> str:
    """Best-effort default shell detection per platform.

    Windows:
      - Prefer 'pwsh' (PowerShell 7), then 'powershell', else 'cmd'
    POSIX:
      - Respect $SHELL if set, else '/bin/bash' or '/bin/sh'
    """
    if os.name == "nt":
        for candidate in ("pwsh", "powershell", "cmd"):
            if shutil.which(candidate):
                return candidate
        return "cmd"

    # POSIX
    env_shell = os.environ.get("SHELL")
    if env_shell:
        return env_shell
    for candidate in ("/bin/bash", "/bin/zsh", "/bin/sh"):
        if Path(candidate).exists():
            return candidate
    return "/bin/sh"


__all__ = [
    "TerminalConfig",
    "detect_default_shell",
]
