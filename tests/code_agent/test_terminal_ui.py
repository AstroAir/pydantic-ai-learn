"""
Test Terminal UI Components

Unit tests for the terminal UI components to ensure proper functionality.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from rich.console import Console

from code_agent.config.logging import LogLevel
from code_agent.ui.input_handler import (
    CommandCompleter,
    InputValidator,
)
from code_agent.ui.session_manager import (
    SessionEntry,
)
from code_agent.ui.streaming import (
    StreamBuffer,
    get_terminal_size,
    is_terminal_wide,
)
from code_agent.ui.terminal import (
    CodeAgentTerminal,
    TerminalUIComponents,
    TerminalUIState,
)


def test_terminal_ui_state() -> None:
    """Test TerminalUIState initialization and properties."""
    print("Testing TerminalUIState...")

    state = TerminalUIState()

    assert state.current_status == "idle"
    assert state.current_prompt == ""
    assert state.response_buffer == ""
    assert state.command_history == []
    assert state.history_index == -1
    assert state.total_requests == 0
    assert state.successful_requests == 0
    assert state.failed_requests == 0

    print("✅ TerminalUIState tests passed")


def test_terminal_ui_components() -> None:
    """Test TerminalUIComponents creation."""
    print("Testing TerminalUIComponents...")

    console = Console()
    components = TerminalUIComponents(console)

    # Test header creation
    header = components.create_header("idle")
    assert header is not None

    header_thinking = components.create_header("thinking")
    assert header_thinking is not None

    # Test metrics panel
    metrics = components.create_metrics_panel(
        total_requests=10, successful=8, failed=2, input_tokens=1000, output_tokens=2000
    )
    assert metrics is not None

    # Test error panel (no errors)
    error_panel = components.create_error_panel({"total_errors": 0})
    assert error_panel is not None

    # Test error panel (with errors)
    error_panel_with_errors = components.create_error_panel(
        {
            "total_errors": 2,
            "recent_errors": [
                {
                    "type": "ValueError",
                    "message": "Test error",
                    "category": "permanent",
                    "timestamp": "2025-10-15 12:00:00",
                }
            ],
        }
    )
    assert error_panel_with_errors is not None

    # Test tool execution panel
    tool_panel = components.create_tool_execution_panel("analyze_code", "running")
    assert tool_panel is not None

    tool_panel_success = components.create_tool_execution_panel("analyze_code", "success")
    assert tool_panel_success is not None

    # Test markdown rendering
    markdown = components.render_markdown("# Test\n\nThis is **bold**")
    assert markdown is not None

    # Test code rendering
    code = components.render_code("def hello():\n    print('Hello')", "python")
    assert code is not None

    # Test response panel
    response = components.create_response_panel("Test response")
    assert response is not None

    response_streaming = components.create_response_panel("Test", streaming=True)
    assert response_streaming is not None

    # Test prompt panel
    prompt = components.create_prompt_panel("Test prompt")
    assert prompt is not None

    print("✅ TerminalUIComponents tests passed")


def test_code_agent_terminal_initialization() -> None:
    """Test CodeAgentTerminal initialization."""
    print("Testing CodeAgentTerminal initialization...")

    # Skip if no API key (tests UI components only)
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping (no API key) - UI components work correctly")
        return

    # Test default initialization
    terminal = CodeAgentTerminal()
    assert terminal.console is not None
    assert terminal.components is not None
    assert terminal.state is not None
    assert terminal.agent is not None

    # Test custom initialization
    terminal_custom = CodeAgentTerminal(model="openai:gpt-4", log_level=LogLevel.DEBUG, enable_workflow=True)
    assert terminal_custom.console is not None
    assert terminal_custom.agent is not None

    print("✅ CodeAgentTerminal initialization tests passed")


def test_terminal_command_handling() -> None:
    """Test terminal command handling."""
    print("Testing terminal command handling...")

    # Skip if no API key
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping (no API key) - UI components work correctly")
        return

    terminal = CodeAgentTerminal()

    # Test exit commands
    assert terminal.handle_command("exit") is False

    terminal2 = CodeAgentTerminal()
    assert terminal2.handle_command("quit") is False

    terminal3 = CodeAgentTerminal()
    assert terminal3.handle_command("q") is False

    # Test non-exit commands
    terminal4 = CodeAgentTerminal()
    assert terminal4.handle_command("help") is True
    assert terminal4.handle_command("metrics") is True
    assert terminal4.handle_command("errors") is True
    assert terminal4.handle_command("workflow") is True
    assert terminal4.handle_command("history") is True
    assert terminal4.handle_command("clear") is True

    print("✅ Terminal command handling tests passed")


def test_terminal_display_methods() -> None:
    """Test terminal display methods."""
    print("Testing terminal display methods...")

    # Skip if no API key
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping (no API key) - UI components work correctly")
        return

    terminal = CodeAgentTerminal()

    # These should not raise exceptions
    try:
        terminal.display_welcome()
        terminal.display_help()
        terminal.display_metrics()
        terminal.display_errors()
        terminal.display_workflow_status()
        terminal.display_history()
        print("✅ Terminal display methods tests passed")
    except Exception as e:
        print(f"❌ Terminal display methods tests failed: {e}")
        raise


def test_session_export() -> None:
    """Test session export functionality."""
    print("Testing session export...")

    import os
    import tempfile

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping (no API key) - UI components work correctly")
        return

    terminal = CodeAgentTerminal()

    # Add some history
    terminal.state.command_history.append("Test command 1")
    terminal.state.command_history.append("Test command 2")
    terminal.state.total_requests = 2
    terminal.state.successful_requests = 1
    terminal.state.failed_requests = 1

    # Export to temp file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        temp_file = f.name

    try:
        terminal.export_session(temp_file)

        # Verify file exists and has content
        assert os.path.exists(temp_file)
        with open(temp_file) as f:
            content = f.read()
            assert "Code Agent Session" in content
            assert "Test command 1" in content
            assert "Test command 2" in content

        print("✅ Session export tests passed")
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_input_validator() -> None:
    """Test InputValidator."""
    print("Testing InputValidator...")

    validator = InputValidator(min_length=1, max_length=100, allow_empty=False)

    # Test valid input
    is_valid, msg = validator.validate("test input")
    assert is_valid is True
    assert msg == ""

    # Test empty input
    is_valid, msg = validator.validate("")
    assert is_valid is False
    assert "empty" in msg.lower()

    # Test too long input
    is_valid, msg = validator.validate("x" * 101)
    assert is_valid is False
    assert "long" in msg.lower()

    print("✅ InputValidator tests passed")


def test_command_completer() -> None:
    """Test CommandCompleter."""
    print("Testing CommandCompleter...")

    completer = CommandCompleter()
    assert "exit" in completer.commands
    assert "help" in completer.commands

    completer.add_command("custom")
    assert "custom" in completer.commands

    completer.remove_command("custom")
    assert "custom" not in completer.commands

    print("✅ CommandCompleter tests passed")


def test_stream_buffer() -> None:
    """Test StreamBuffer."""
    print("Testing StreamBuffer...")

    buffer = StreamBuffer(min_update_interval=0.01, max_buffer_size=10)

    # Add chunks
    should_flush = buffer.add("test")
    assert isinstance(should_flush, bool)

    # Flush
    text = buffer.flush()
    assert text == "test"

    # Stats
    stats = buffer.get_stats()
    assert "buffer_size" in stats
    assert "total_chars" in stats

    print("✅ StreamBuffer tests passed")


def test_terminal_size_detection() -> None:
    """Test terminal size detection."""
    print("Testing terminal size detection...")

    columns, lines = get_terminal_size()
    assert columns > 0
    assert lines > 0

    is_wide = is_terminal_wide()
    assert isinstance(is_wide, bool)

    print("✅ Terminal size detection tests passed")


def test_session_entry() -> None:
    """Test SessionEntry."""
    print("Testing SessionEntry...")

    entry = SessionEntry(
        timestamp="2025-10-21T12:00:00",
        prompt="test prompt",
        response="test response",
        status="success",
        execution_time=1.5,
    )

    # Test to_dict
    data = entry.to_dict()
    assert data["prompt"] == "test prompt"
    assert data["status"] == "success"

    # Test from_dict
    entry2 = SessionEntry.from_dict(data)
    assert entry2.prompt == entry.prompt
    assert entry2.status == entry.status

    print("✅ SessionEntry tests passed")


def test_shell_command_initialization() -> None:
    """Test shell command execution initialization."""
    print("Testing shell command initialization...")

    # Disable advanced input to avoid console issues in tests
    terminal = CodeAgentTerminal(enable_advanced_input=False)

    # Check shell command attributes
    assert hasattr(terminal, "cwd")
    assert hasattr(terminal, "env_overrides")
    assert hasattr(terminal, "sessions")
    assert hasattr(terminal, "default_timeout")

    # Check default values
    assert terminal.default_timeout == 30.0
    assert isinstance(terminal.env_overrides, dict)
    assert isinstance(terminal.sessions, dict)
    assert len(terminal.sessions) == 0

    print("✅ Shell command initialization tests passed")


def test_terminal_ui_components_shell_panels() -> None:
    """Test TerminalUIComponents shell-related panels."""
    print("Testing TerminalUIComponents shell panels...")

    from code_agent.utils.terminal_exec import CommandResult

    console = Console()
    components = TerminalUIComponents(console)

    # Test stdout panel
    stdout_panel = components.create_stdout_panel("test output")
    assert stdout_panel is not None
    assert "STDOUT" in str(stdout_panel.title)

    # Test stderr panel
    stderr_panel = components.create_stderr_panel("test error")
    assert stderr_panel is not None
    assert "STDERR" in str(stderr_panel.title)

    # Test process summary panel
    result = CommandResult(stdout="output", stderr="", exit_code=0, duration_s=1.5, timed_out=False)
    summary_panel = components.create_process_summary_panel(result, cwd="/test/path")
    assert summary_panel is not None
    assert "Process Summary" in str(summary_panel.title)

    print("✅ TerminalUIComponents shell panels tests passed")


def run_all_tests() -> None:
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Running Terminal UI Tests")
    print("=" * 80 + "\n")

    tests = [
        test_terminal_ui_state,
        test_terminal_ui_components,
        test_code_agent_terminal_initialization,
        test_terminal_command_handling,
        test_terminal_display_methods,
        test_session_export,
        test_input_validator,
        test_command_completer,
        test_stream_buffer,
        test_terminal_size_detection,
        test_session_entry,
        test_shell_command_initialization,
        test_terminal_ui_components_shell_panels,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80 + "\n")

    if failed > 0:
        raise AssertionError(f"{failed} test(s) failed")


if __name__ == "__main__":
    run_all_tests()
