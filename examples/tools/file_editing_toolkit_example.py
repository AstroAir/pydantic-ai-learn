"""
Example: Using File Editing Toolkit with PydanticAI

This example demonstrates how to integrate the file editing toolkit with
a PydanticAI agent, including all four tools: Edit, MultiEdit, Write, and NotebookEdit.

The tools are registered with proper type hints and docstrings that PydanticAI
uses to generate the tool schema for the LLM.

Run with: python examples/tools/file_editing_toolkit_example.py
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic_ai import Agent, RunContext

from tools.file_editing_toolkit import (
    EditInput,
    FileEditState,
    MultiEditInput,
    NotebookEditInput,
    WriteInput,
    edit_file,
    multi_edit_file,
    notebook_edit,
    write_file,
)

# ============================================================================
# Create Agent with File Editing Tools
# ============================================================================

# Create the agent with FileEditState as dependency type
file_editor_agent = Agent(
    "openai:gpt-4",
    deps_type=FileEditState,
    system_prompt=(
        "You are a helpful file editing assistant. You can read, edit, and write files. "
        "Always read a file before editing it to understand its current content. "
        "Use absolute paths for all file operations."
    ),
)


# ============================================================================
# Register Tools with PydanticAI
# ============================================================================


@file_editor_agent.tool
def edit(
    ctx: RunContext[FileEditState], file_path: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    """
    Perform exact string replacement in a file.

    The file must have been read previously. Validates that old_string exists
    and is unique (unless replace_all=True).

    Args:
        file_path: Absolute path to the file to modify
        old_string: The exact text to replace (must exist in file)
        new_string: The replacement text (must differ from old_string)
        replace_all: Whether to replace all occurrences (default: False)

    Returns:
        Success message with edit details
    """
    return edit_file(
        EditInput(file_path=file_path, old_string=old_string, new_string=new_string, replace_all=replace_all), ctx.deps
    )


@file_editor_agent.tool
def multi_edit(ctx: RunContext[FileEditState], file_path: str, edits: list[dict[str, str | bool]]) -> str:
    """
    Perform multiple sequential edits atomically on a single file.

    All edits are validated before any are applied. If any edit fails,
    none are applied. Edits are applied sequentially.

    Args:
        file_path: Absolute path to the file to modify
        edits: Array of edit operations, each with 'old_string', 'new_string', and optional 'replace_all'

    Returns:
        Success message with edit count

    Example:
        edits = [
            {"old_string": "DEBUG = False", "new_string": "DEBUG = True"},
            {"old_string": "PORT = 8000", "new_string": "PORT = 3000"}
        ]
    """
    from file_editing_toolkit import SingleEdit

    # Convert dict edits to SingleEdit objects
    single_edits = [
        SingleEdit(
            old_string=str(e["old_string"]),
            new_string=str(e["new_string"]),
            replace_all=bool(e.get("replace_all", False)),
        )
        for e in edits
    ]

    return multi_edit_file(MultiEditInput(file_path=file_path, edits=single_edits), ctx.deps)


@file_editor_agent.tool
def write(ctx: RunContext[FileEditState], file_path: str, content: str) -> str:
    """
    Write or overwrite a file on the filesystem.

    For existing files, validates that the file was read first.
    For new files, creates the file directly.

    Args:
        file_path: Absolute path to the file to write
        content: The complete content to write to the file

    Returns:
        Success message
    """
    return write_file(WriteInput(file_path=file_path, content=content), ctx.deps)


@file_editor_agent.tool
def notebook_edit_cell(
    ctx: RunContext[FileEditState],
    notebook_path: str,
    cell_id: str | None,
    new_source: str = "",
    cell_type: str | None = None,
    edit_mode: str = "replace",
) -> str:
    """
    Edit Jupyter notebook cells (replace, insert, or delete).

    Edit modes:
    - replace: Replace the content of the cell with cell_id
    - insert: Insert a new cell after cell_id (or at beginning if cell_id is None)
    - delete: Delete the cell with cell_id

    Args:
        notebook_path: Absolute path to the .ipynb file
        cell_id: ID of the target cell (required for replace/delete, optional for insert)
        new_source: The new source content for the cell (required for replace/insert)
        cell_type: Cell type: 'code' or 'markdown' (required for insert)
        edit_mode: Edit operation: 'replace', 'insert', or 'delete' (default: 'replace')

    Returns:
        Success message with operation details
    """
    return notebook_edit(
        NotebookEditInput(
            notebook_path=notebook_path,
            cell_id=cell_id,
            new_source=new_source,
            cell_type=cell_type,  # type: ignore
            edit_mode=edit_mode,  # type: ignore
        ),
        ctx.deps,
    )


@file_editor_agent.tool_plain
def mark_file_as_read(file_path: str) -> str:
    """
    Mark a file as having been read (for testing purposes).

    In production, this would be called automatically when a file is read
    by a read_file tool.

    Args:
        file_path: Absolute path to the file

    Returns:
        Confirmation message
    """
    # This is a helper for testing - in production you'd have a read_file tool
    # that automatically marks files as read
    return f"Marked {file_path} as read (testing helper)"


# ============================================================================
# Example Usage
# ============================================================================


def example_basic_edit() -> None:
    """Example 1: Basic file editing."""
    print("\n" + "=" * 60)
    print("Example 1: Basic File Edit")
    print("=" * 60)

    # Create state tracker
    state = FileEditState()

    # Simulate reading a file (in production, use a read_file tool)
    test_file = "/tmp/test_file.py"
    state.mark_as_read(test_file)

    # Create test file
    from pathlib import Path

    Path(test_file).write_text("def old_function():\n    pass\n")

    # Use the edit tool
    result = edit_file(
        EditInput(
            file_path=test_file, old_string="def old_function():", new_string="def new_function():", replace_all=False
        ),
        state,
    )

    print(result)
    print(f"\nUpdated content:\n{Path(test_file).read_text()}")


def example_multi_edit() -> None:
    """Example 2: Multiple edits atomically."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Edit (Atomic)")
    print("=" * 60)

    from file_editing_toolkit import SingleEdit

    state = FileEditState()
    test_file = "/tmp/config.py"
    state.mark_as_read(test_file)

    # Create test file
    from pathlib import Path

    Path(test_file).write_text("DEBUG = False\nPORT = 8000\nHOST = 'localhost'\n")

    # Apply multiple edits
    result = multi_edit_file(
        MultiEditInput(
            file_path=test_file,
            edits=[
                SingleEdit(old_string="DEBUG = False", new_string="DEBUG = True"),
                SingleEdit(old_string="PORT = 8000", new_string="PORT = 3000"),
                SingleEdit(old_string="HOST = 'localhost'", new_string="HOST = '0.0.0.0'"),
            ],
        ),
        state,
    )

    print(result)
    print(f"\nUpdated content:\n{Path(test_file).read_text()}")


def example_write_file() -> None:
    """Example 3: Write new file."""
    print("\n" + "=" * 60)
    print("Example 3: Write New File")
    print("=" * 60)

    state = FileEditState()
    test_file = "/tmp/new_script.py"

    # Write new file
    result = write_file(
        WriteInput(file_path=test_file, content="#!/usr/bin/env python3\n\nprint('Hello, World!')\n"), state
    )

    print(result)
    print(f"\nFile content:\n{Path(test_file).read_text()}")


if __name__ == "__main__":
    # Run examples
    example_basic_edit()
    example_multi_edit()
    example_write_file()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
