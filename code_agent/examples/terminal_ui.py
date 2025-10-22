"""
Code Agent Terminal UI Examples

Comprehensive examples demonstrating the interactive terminal UI features.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio

from code_agent import (
    CodeAgentTerminal,
    LogLevel,
    launch_terminal,
    launch_terminal_async,
)

# ============================================================================
# Example 1: Basic Terminal Launch
# ============================================================================


def example_1_basic_terminal() -> None:
    """
    Example 1: Launch basic terminal with default settings.

    This is the simplest way to start the interactive terminal.
    """
    print("=" * 80)
    print("Example 1: Basic Terminal Launch")
    print("=" * 80)
    print()
    print("Launching Code Agent Terminal with default settings...")
    print("Commands available: help, metrics, errors, workflow, history, export, exit")
    print()

    # Launch terminal
    launch_terminal()


# ============================================================================
# Example 2: Terminal with Streaming
# ============================================================================


def example_2_streaming_terminal() -> None:
    """
    Example 2: Launch terminal with streaming mode enabled.

    Streaming mode shows responses as they are generated in real-time.
    """
    print("=" * 80)
    print("Example 2: Terminal with Streaming")
    print("=" * 80)
    print()
    print("Launching Code Agent Terminal with streaming enabled...")
    print("Responses will stream in real-time!")
    print()

    # Launch terminal with streaming
    launch_terminal(use_streaming=True)


# ============================================================================
# Example 3: Terminal with Custom Configuration
# ============================================================================


def example_3_custom_configuration() -> None:
    """
    Example 3: Launch terminal with custom configuration.

    Demonstrates how to configure the terminal with specific settings.
    """
    print("=" * 80)
    print("Example 3: Custom Configuration")
    print("=" * 80)
    print()
    print("Launching Code Agent Terminal with custom settings:")
    print("  - Model: openai:gpt-4")
    print("  - Log Level: DEBUG")
    print("  - Workflow: Enabled")
    print("  - Streaming: Enabled")
    print()

    # Launch terminal with custom configuration
    launch_terminal(model="openai:gpt-4", log_level=LogLevel.DEBUG, enable_workflow=True, use_streaming=True)


# ============================================================================
# Example 4: Async Terminal
# ============================================================================


async def example_4_async_terminal() -> None:
    """
    Example 4: Launch terminal in async mode.

    Async mode provides better performance for streaming responses.
    """
    print("=" * 80)
    print("Example 4: Async Terminal")
    print("=" * 80)
    print()
    print("Launching Code Agent Terminal in async mode...")
    print("This provides optimal performance for streaming!")
    print()

    # Launch terminal in async mode
    await launch_terminal_async(model="openai:gpt-4", log_level=LogLevel.INFO, enable_workflow=True)


# ============================================================================
# Example 5: Programmatic Terminal Control
# ============================================================================


def example_5_programmatic_control() -> None:
    """
    Example 5: Programmatic terminal control.

    Demonstrates how to create and control the terminal programmatically.
    """
    print("=" * 80)
    print("Example 5: Programmatic Terminal Control")
    print("=" * 80)
    print()

    # Create terminal instance
    terminal = CodeAgentTerminal(model="openai:gpt-4", log_level=LogLevel.INFO, enable_workflow=True)

    # Display welcome
    terminal.display_welcome()

    # Show help
    print("\nDisplaying help...")
    terminal.display_help()

    # Show metrics
    print("\nDisplaying metrics...")
    terminal.display_metrics()

    # Show errors
    print("\nDisplaying errors...")
    terminal.display_errors()

    # Show workflow status
    print("\nDisplaying workflow status...")
    terminal.display_workflow_status()

    # Now run the interactive loop
    print("\nStarting interactive mode...")
    terminal.run(use_streaming=False)


# ============================================================================
# Example 6: Terminal with Session Export
# ============================================================================


def example_6_session_export() -> None:
    """
    Example 6: Terminal with session export.

    Demonstrates how to export session history to a file.
    """
    print("=" * 80)
    print("Example 6: Session Export")
    print("=" * 80)
    print()
    print("Launching Code Agent Terminal...")
    print("Use 'export session.md' to save your session!")
    print()

    # Create terminal
    terminal = CodeAgentTerminal()

    # Display welcome
    terminal.display_welcome()

    # Show export instructions
    terminal.console.print("[bold cyan]💡 Tip:[/bold cyan] Use 'export [filename]' to save your session")
    terminal.console.print("[dim]Example: export my_session.md[/dim]")
    terminal.console.print()

    # Run terminal
    terminal.run()


# ============================================================================
# Example 7: Terminal UI Components Demo
# ============================================================================


def example_7_ui_components_demo() -> None:
    """
    Example 7: Demonstrate UI components.

    Shows all the different UI components available.
    """
    print("=" * 80)
    print("Example 7: UI Components Demo")
    print("=" * 80)
    print()

    from rich.console import Console

    from code_agent.terminal_ui import TerminalUIComponents

    console = Console()
    components = TerminalUIComponents(console)

    # Header
    console.print(components.create_header("idle"))
    console.print()

    # Metrics panel
    console.print(
        components.create_metrics_panel(
            total_requests=10, successful=8, failed=2, input_tokens=1000, output_tokens=2000
        )
    )
    console.print()

    # Error panel (no errors)
    console.print(components.create_error_panel({"total_errors": 0}))
    console.print()

    # Tool execution panel
    console.print(components.create_tool_execution_panel("analyze_code", "success"))
    console.print()

    # Response panel
    console.print(
        components.create_response_panel(
            "This is a sample response with **markdown** formatting!\n\n"
            "- Item 1\n"
            "- Item 2\n"
            "- Item 3\n\n"
            "```python\n"
            "def hello():\n"
            "    print('Hello, World!')\n"
            "```"
        )
    )
    console.print()

    # Prompt panel
    console.print(components.create_prompt_panel("Analyze the code in my_module.py"))


# ============================================================================
# Main Entry Point
# ============================================================================

# ============================================================================
# Example 8: Shell Commands and Interactive Sessions
# ============================================================================


def example_8_shell_and_sessions() -> None:
    """
    Example 8: Demonstrate shell commands and interactive sessions.

    Shows how to run shell one-liners and manage interactive processes.
    """
    print("=" * 80)
    print("Example 8: Shell & Sessions")
    print("=" * 80)
    print()
    print("Tips:")
    print("  - Run a one-shot command:  !echo hello")
    print("  - Start interactive shell: :shell   (or :shell pwsh / :shell bash)")
    print("  - List sessions:            :procs")
    print("  - Attach to session:        :attach 1")
    print("  - Kill a session:           :kill 1")
    print("  - Set timeout:              :timeout 10")
    print("  - Change working dir:       :cwd D:/path/to/dir")
    print("  - Set env var:              :env FOO=bar  (use :env --list to show)")
    print()
    print("Launching terminal. Try commands above inside the terminal.")
    launch_terminal()


def main() -> None:
    """Main entry point for examples."""
    examples = {
        "1": ("Basic Terminal Launch", example_1_basic_terminal),
        "2": ("Terminal with Streaming", example_2_streaming_terminal),
        "3": ("Custom Configuration", example_3_custom_configuration),
        "4": ("Async Terminal", lambda: asyncio.run(example_4_async_terminal())),
        "5": ("Programmatic Control", example_5_programmatic_control),
        "6": ("Session Export", example_6_session_export),
        "7": ("UI Components Demo", example_7_ui_components_demo),
        "8": ("Shell & Sessions", example_8_shell_and_sessions),
    }

    print("\n" + "=" * 80)
    print("Code Agent Terminal UI Examples")
    print("=" * 80)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("\n  0. Run all examples")
    print("  q. Quit")

    choice = input("\nSelect an example (1-7, 0, or q): ").strip()

    if choice == "q":
        print("Goodbye!")
        return

    if choice == "0":
        for key, (name, func) in examples.items():
            print(f"\n\nRunning Example {key}: {name}")
            input("Press Enter to continue...")
            func()
    elif choice in examples:
        name, func = examples[choice]
        print(f"\n\nRunning Example {choice}: {name}")
        func()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
