#!/usr/bin/env python3
"""
Code Agent Terminal Launcher

Standalone script to launch the Code Agent terminal UI.

Usage:
    python -m code_agent.run_terminal
    python -m code_agent.run_terminal --streaming
    python -m code_agent.run_terminal --model openai:gpt-4 --log-level DEBUG

Author: The Augster
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from code_agent.config.logging import LogLevel
from code_agent.ui.terminal import launch_terminal


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Code Agent Terminal - Interactive AI-powered code analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with default settings
  python -m code_agent.run_terminal

  # Launch with streaming mode
  python -m code_agent.run_terminal --streaming

  # Launch with custom model and debug logging
  python -m code_agent.run_terminal --model openai:gpt-4 --log-level DEBUG

  # Launch without workflow orchestration
  python -m code_agent.run_terminal --no-workflow

Commands in terminal:
  exit, quit     - Exit the terminal
  clear          - Clear the screen
  metrics        - Show detailed performance metrics
  errors         - Show error history and diagnostics
  workflow       - Show workflow status
  history        - Show command history
  export [file]  - Export session to file
  help           - Show help message
        """,
    )

    parser.add_argument("--model", type=str, default="openai:gpt-4", help="Model to use (default: openai:gpt-4)")

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode for responses")

    parser.add_argument("--no-workflow", action="store_true", help="Disable workflow orchestration")

    parser.add_argument(
        "--no-advanced-input", action="store_true", help="Disable advanced input handling (prompt_toolkit)"
    )

    parser.add_argument("--no-session-manager", action="store_true", help="Disable session persistence and auto-save")

    parser.add_argument("--no-enhanced-streaming", action="store_true", help="Disable enhanced streaming display")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Convert log level string to enum
    log_level_map = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL,
    }
    log_level = log_level_map[args.log_level]

    # Launch terminal
    try:
        launch_terminal(
            model=args.model,
            log_level=log_level,
            enable_workflow=not args.no_workflow,
            use_streaming=args.streaming,
            enable_advanced_input=not args.no_advanced_input,
            enable_session_manager=not args.no_session_manager,
            enable_enhanced_streaming=not args.no_enhanced_streaming,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye! 👋")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)


def run_terminal(
    model: str = "openai:gpt-4",
    log_level: LogLevel = LogLevel.INFO,
    enable_workflow: bool = True,
    use_streaming: bool = True,
    enable_advanced_input: bool = True,
    enable_session_manager: bool = True,
    enable_enhanced_streaming: bool = True,
) -> None:
    """
    Launch the Code Agent terminal UI.

    Args:
        model: Model to use (default: openai:gpt-4)
        log_level: Logging level (default: INFO)
        enable_workflow: Enable workflow orchestration (default: True)
        use_streaming: Enable streaming mode (default: True)
        enable_advanced_input: Enable advanced input handling (default: True)
        enable_session_manager: Enable session persistence (default: True)
        enable_enhanced_streaming: Enable enhanced streaming display (default: True)
    """
    launch_terminal(
        model=model,
        log_level=log_level,
        enable_workflow=enable_workflow,
        use_streaming=use_streaming,
        enable_advanced_input=enable_advanced_input,
        enable_session_manager=enable_session_manager,
        enable_enhanced_streaming=enable_enhanced_streaming,
    )


if __name__ == "__main__":
    main()
