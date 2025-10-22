"""
File Editing Toolkit for PydanticAI

A comprehensive, production-grade Python implementation of file editing tools
designed for seamless integration with PydanticAI agents.

Tools:
1. Edit: Perform exact string replacements with strict validation
2. MultiEdit: Apply multiple edits atomically to a single file
3. Write: Write or overwrite files with read verification
4. NotebookEdit: Edit Jupyter notebook cells (replace, insert, delete)

Features:
- Modern Python 3.12+ with latest type hints (using | for unions)
- Pydantic v2 validation for robust input handling
- State tracking to ensure files are read before editing
- Atomic operations for MultiEdit (all succeed or all fail)
- Comprehensive error handling with informative messages
- Security-focused design (path validation, absolute paths only)

Security Considerations:
- All paths are normalized and validated to prevent traversal attacks
- Absolute paths required to prevent ambiguity
- File read tracking prevents blind modifications
- Proper error handling prevents information leakage

Example Usage:
    ```python
    from tools.file_editing_toolkit import FileEditState, edit_file, EditInput
    from pydantic_ai import Agent, RunContext

    # Create state tracker
    state = FileEditState()

    # Create agent with file editing tools
    agent = Agent('openai:gpt-4', deps_type=FileEditState)

    # Register tools
    @agent.tool
    def edit(ctx: RunContext[FileEditState], file_path: str,
             old_string: str, new_string: str, replace_all: bool = False) -> str:
        return edit_file(EditInput(
            file_path=file_path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all
        ), ctx.deps)

    # Use the agent
    result = agent.run_sync('Edit the file...', deps=state)
    ```

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# Custom Exceptions
# ============================================================================


class FileEditError(Exception):
    """Base exception for file editing tool errors."""

    pass


class EditValidationError(FileEditError):
    """Raised when edit validation fails."""

    pass


class MultiEditError(FileEditError):
    """Raised when multi-edit operation fails."""

    pass


class WriteError(FileEditError):
    """Raised when write operation fails."""

    pass


class NotebookEditError(FileEditError):
    """Raised when notebook edit operation fails."""

    pass


class FileNotReadError(EditValidationError):
    """Raised when attempting to edit a file that hasn't been read."""

    pass


class StringNotFoundError(EditValidationError):
    """Raised when old_string is not found in file."""

    pass


class StringNotUniqueError(EditValidationError):
    """Raised when old_string appears multiple times but replace_all=False."""

    pass


class IdenticalStringsError(EditValidationError):
    """Raised when old_string and new_string are identical."""

    pass


class CellNotFoundError(NotebookEditError):
    """Raised when specified cell_id is not found in notebook."""

    pass


# ============================================================================
# State Management
# ============================================================================


@dataclass
class FileEditState:
    """
    State tracker for file editing operations.

    Maintains a set of file paths that have been read during the conversation,
    enabling validation that files are read before being edited.

    Attributes:
        read_files: Set of absolute file paths that have been read
    """

    read_files: set[str] = field(default_factory=set)

    def mark_as_read(self, file_path: str) -> None:
        """Mark a file as having been read."""
        self.read_files.add(str(Path(file_path).resolve()))

    def was_read(self, file_path: str) -> bool:
        """Check if a file has been read."""
        return str(Path(file_path).resolve()) in self.read_files

    def clear(self) -> None:
        """Clear all read file tracking."""
        self.read_files.clear()


# ============================================================================
# Helper Functions
# ============================================================================


def validate_absolute_path(path: str) -> Path:
    """
    Validate and normalize a file path.

    Args:
        path: File path to validate

    Returns:
        Normalized absolute Path object

    Raises:
        ValueError: If path is not absolute or contains invalid characters
    """
    try:
        p = Path(path).expanduser()

        # Ensure it's absolute
        if not p.is_absolute():
            raise ValueError(f"Path must be absolute, got: {path}")

        # Resolve to normalize (removes .., ., etc.)
        return p.resolve()
    except Exception as e:
        raise ValueError(f"Invalid path '{path}': {e}") from e


def strip_line_numbers(content: str) -> str:
    """
    Strip line number prefixes from file content.

    Handles format: "  123\tcontent" where line numbers are right-aligned
    with spaces, followed by a tab character.

    Args:
        content: File content potentially with line numbers

    Returns:
        Content with line numbers stripped
    """
    lines = content.split("\n")
    stripped_lines = []

    # Pattern: optional spaces, digits, tab, then content
    line_num_pattern = re.compile(r"^\s*\d+\t")

    for line in lines:
        if line_num_pattern.match(line):
            # Remove everything up to and including the first tab
            stripped_line = line.split("\t", 1)[1] if "\t" in line else line
            stripped_lines.append(stripped_line)
        else:
            stripped_lines.append(line)

    return "\n".join(stripped_lines)


def read_file_content(file_path: Path) -> str:
    """
    Read file content with proper error handling.

    Args:
        file_path: Path to file to read

    Returns:
        File content as string

    Raises:
        FileEditError: If file cannot be read
    """
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileEditError(f"File not found: {file_path}") from e
    except PermissionError as e:
        raise FileEditError(f"Permission denied reading file: {file_path}") from e
    except Exception as e:
        raise FileEditError(f"Error reading file {file_path}: {e}") from e


def write_file_content(file_path: Path, content: str) -> None:
    """
    Write content to file with proper error handling.

    Args:
        file_path: Path to file to write
        content: Content to write

    Raises:
        FileEditError: If file cannot be written
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    except PermissionError as e:
        raise FileEditError(f"Permission denied writing file: {file_path}") from e
    except Exception as e:
        raise FileEditError(f"Error writing file {file_path}: {e}") from e


# ============================================================================
# Input Models
# ============================================================================


class EditInput(BaseModel):
    """
    Input schema for edit_file tool.

    Validates parameters for performing exact string replacements in files.
    """

    file_path: str = Field(..., description="Absolute path to the file to modify", min_length=1, max_length=2000)

    old_string: str = Field(..., description="The exact text to replace (must exist in file)", min_length=1)

    new_string: str = Field(..., description="The replacement text (must differ from old_string)")

    replace_all: bool = Field(
        default=False, description="Whether to replace all occurrences (default: False, requires unique match)"
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "file_path": "/home/user/project/main.py",
                    "old_string": "def old_function():",
                    "new_string": "def new_function():",
                    "replace_all": False,
                }
            ]
        },
    }

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate that path is absolute."""
        validate_absolute_path(v)  # Will raise if invalid
        return v


class SingleEdit(BaseModel):
    """Single edit operation for MultiEdit."""

    old_string: str = Field(..., description="The exact text to replace", min_length=1)

    new_string: str = Field(..., description="The replacement text")

    replace_all: bool = Field(default=False, description="Whether to replace all occurrences")


class MultiEditInput(BaseModel):
    """
    Input schema for multi_edit_file tool.

    Validates parameters for performing multiple sequential edits atomically.
    """

    file_path: str = Field(..., description="Absolute path to the file to modify", min_length=1, max_length=2000)

    edits: list[SingleEdit] = Field(..., description="Array of edit operations to apply sequentially", min_length=1)

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "file_path": "/home/user/project/config.py",
                    "edits": [
                        {"old_string": "DEBUG = False", "new_string": "DEBUG = True"},
                        {"old_string": "PORT = 8000", "new_string": "PORT = 3000"},
                    ],
                }
            ]
        },
    }

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate that path is absolute."""
        validate_absolute_path(v)
        return v


class WriteInput(BaseModel):
    """
    Input schema for write_file tool.

    Validates parameters for writing or overwriting files.
    """

    file_path: str = Field(..., description="Absolute path to the file to write", min_length=1, max_length=2000)

    content: str = Field(..., description="The complete content to write to the file")

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "file_path": "/home/user/project/new_file.py",
                    "content": "#!/usr/bin/env python3\n\nprint('Hello, World!')\n",
                }
            ]
        },
    }

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate that path is absolute."""
        validate_absolute_path(v)
        return v


class NotebookEditInput(BaseModel):
    """
    Input schema for notebook_edit tool.

    Validates parameters for editing Jupyter notebook cells.
    """

    notebook_path: str = Field(..., description="Absolute path to the .ipynb file", min_length=1, max_length=2000)

    cell_id: str | None = Field(
        default=None,
        description=(
            "ID of the target cell (for insert: new cell inserted after this; "
            "for replace/delete: this cell is modified/deleted; if None for insert: insert at beginning)"
        ),
        min_length=1,
        max_length=64,
    )

    new_source: str = Field(
        default="", description="The new source content for the cell (required for replace and insert modes)"
    )

    cell_type: Literal["code", "markdown"] | None = Field(
        default=None, description="Cell type (required for insert mode, defaults to current type for replace mode)"
    )

    edit_mode: Literal["replace", "insert", "delete"] = Field(
        default="replace",
        description=(
            "The type of edit operation: replace (modify existing cell), insert (add new cell), delete (remove cell)"
        ),
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "notebook_path": "/home/user/notebook.ipynb",
                    "cell_id": "abc123",
                    "new_source": "print('Updated code')",
                    "edit_mode": "replace",
                },
                {
                    "notebook_path": "/home/user/notebook.ipynb",
                    "cell_id": "abc123",
                    "new_source": "# New markdown cell",
                    "cell_type": "markdown",
                    "edit_mode": "insert",
                },
            ]
        },
    }

    @field_validator("notebook_path")
    @classmethod
    def validate_notebook_path(cls, v: str) -> str:
        """Validate that path is absolute and ends with .ipynb."""
        validate_absolute_path(v)
        if not v.endswith(".ipynb"):
            raise ValueError("Notebook path must end with .ipynb")
        return v

    @model_validator(mode="after")
    def validate_edit_mode_requirements(self) -> NotebookEditInput:
        """Validate requirements based on edit mode."""
        if self.edit_mode == "insert" and self.cell_type is None:
            raise ValueError("cell_type is required for insert mode")

        if self.edit_mode in ("replace", "insert") and not self.new_source:
            raise ValueError(f"new_source is required for {self.edit_mode} mode")

        if self.edit_mode == "delete" and self.cell_id is None:
            raise ValueError("cell_id is required for delete mode")

        return self


# ============================================================================
# Core Tool Functions
# ============================================================================


def edit_file(input_params: EditInput, state: FileEditState) -> str:
    """
    Perform exact string replacement in a file with strict validation.

    This tool validates that:
    - The file has been read previously in the conversation
    - The old_string exists in the file
    - The old_string is unique (unless replace_all=True)
    - The old_string and new_string are different

    Args:
        input_params: Validated edit input parameters
        state: File edit state tracker

    Returns:
        Success message with edit details

    Raises:
        FileNotReadError: If file hasn't been read
        StringNotFoundError: If old_string not found in file
        StringNotUniqueError: If old_string appears multiple times and replace_all=False
        IdenticalStringsError: If old_string and new_string are identical
        FileEditError: If file operation fails
    """
    file_path = validate_absolute_path(input_params.file_path)

    # Validation 1: File must have been read
    if not state.was_read(str(file_path)):
        raise FileNotReadError(
            f"File must be read before editing: {file_path}\n"
            f"Please read the file first to ensure you understand its current content."
        )

    # Validation 2: Strings must be different
    if input_params.old_string == input_params.new_string:
        raise IdenticalStringsError("old_string and new_string must be different")

    # Read current content
    content = read_file_content(file_path)

    # Strip line numbers if present
    content = strip_line_numbers(content)

    # Validation 3: old_string must exist
    if input_params.old_string not in content:
        raise StringNotFoundError(f"String not found in file: {input_params.old_string[:100]}...")

    # Validation 4: old_string must be unique (unless replace_all=True)
    occurrences = content.count(input_params.old_string)
    if occurrences > 1 and not input_params.replace_all:
        raise StringNotUniqueError(
            f"String appears {occurrences} times in file. "
            f"Set replace_all=True to replace all occurrences, or make old_string more specific."
        )

    # Perform replacement
    if input_params.replace_all:
        new_content = content.replace(input_params.old_string, input_params.new_string)
    else:
        # Replace only first occurrence
        new_content = content.replace(input_params.old_string, input_params.new_string, 1)

    # Write back to file
    write_file_content(file_path, new_content)

    return (
        f"Successfully edited {file_path}\n"
        f"Replaced {occurrences} occurrence(s) of:\n"
        f"  '{input_params.old_string[:50]}...'\n"
        f"With:\n"
        f"  '{input_params.new_string[:50]}...'"
    )


def multi_edit_file(input_params: MultiEditInput, state: FileEditState) -> str:
    """
    Perform multiple sequential edit operations atomically on a single file.

    All edits are validated before any are applied. If any edit fails validation,
    none are applied (atomic transaction). Edits are applied sequentially, so
    each edit operates on the result of the previous edit.

    Special behavior for new file creation:
    - First edit can have old_string="" to create a new file
    - Subsequent edits operate on the newly created content

    Args:
        input_params: Validated multi-edit input parameters
        state: File edit state tracker

    Returns:
        Success message with edit count

    Raises:
        MultiEditError: If any edit fails validation or execution
        FileEditError: If file operations fail
    """
    file_path = validate_absolute_path(input_params.file_path)

    # Check if this is a new file creation (first edit has empty old_string)
    is_new_file = len(input_params.edits) > 0 and input_params.edits[0].old_string == "" and not file_path.exists()

    if is_new_file:
        # For new file creation, start with empty content
        current_content = ""
        # Mark as read since we're creating it
        state.mark_as_read(str(file_path))
    else:
        # File must have been read for editing
        if not state.was_read(str(file_path)):
            raise FileNotReadError(
                f"File must be read before editing: {file_path}\n"
                f"Please read the file first to ensure you understand its current content."
            )

        # Read current content
        current_content = read_file_content(file_path)
        current_content = strip_line_numbers(current_content)

    # Store original content for rollback
    original_content = current_content

    try:
        # Apply edits sequentially
        for i, edit in enumerate(input_params.edits, 1):
            # Special case: empty old_string for file creation (only first edit)
            if edit.old_string == "":
                if i == 1 and is_new_file:
                    current_content = edit.new_string
                    continue
                raise MultiEditError(f"Edit {i}: Empty old_string only allowed for first edit when creating new file")

            # Validate strings are different
            if edit.old_string == edit.new_string:
                raise MultiEditError(f"Edit {i}: old_string and new_string must be different")

            # Validate old_string exists
            if edit.old_string not in current_content:
                raise MultiEditError(f"Edit {i}: String not found in current content: {edit.old_string[:100]}...")

            # Validate uniqueness if not replace_all
            occurrences = current_content.count(edit.old_string)
            if occurrences > 1 and not edit.replace_all:
                raise MultiEditError(
                    f"Edit {i}: String appears {occurrences} times. "
                    f"Set replace_all=True or make old_string more specific."
                )

            # Apply edit
            if edit.replace_all:
                current_content = current_content.replace(edit.old_string, edit.new_string)
            else:
                current_content = current_content.replace(edit.old_string, edit.new_string, 1)

        # All edits successful, write to file
        write_file_content(file_path, current_content)

        return f"Successfully applied {len(input_params.edits)} edit(s) to {file_path}\nAll edits completed atomically."

    except Exception as e:
        # Rollback: restore original content if file existed
        import contextlib

        if not is_new_file and original_content is not None:
            with contextlib.suppress(Exception):
                write_file_content(file_path, original_content)  # Best effort rollback

        # Re-raise with context
        if isinstance(e, MultiEditError):
            raise
        raise MultiEditError(f"Multi-edit failed: {e}") from e


def write_file(input_params: WriteInput, state: FileEditState) -> str:
    """
    Write or overwrite a file on the local filesystem.

    For existing files, validates that the file was read first to prevent
    accidental overwrites. For new files, creates the file directly.

    Args:
        input_params: Validated write input parameters
        state: File edit state tracker

    Returns:
        Success message

    Raises:
        FileNotReadError: If existing file hasn't been read
        WriteError: If write operation fails
    """
    file_path = validate_absolute_path(input_params.file_path)

    # If file exists, it must have been read first
    if file_path.exists() and not state.was_read(str(file_path)):
        raise FileNotReadError(
            f"Existing file must be read before overwriting: {file_path}\n"
            f"Please read the file first to ensure you understand what you're replacing."
        )

    try:
        # Write content to file
        write_file_content(file_path, input_params.content)

        # Mark as read since we just wrote it
        state.mark_as_read(str(file_path))

        action = "Overwritten" if file_path.exists() else "Created"
        return f"{action} file: {file_path}\nContent length: {len(input_params.content)} characters"

    except Exception as e:
        raise WriteError(f"Failed to write file {file_path}: {e}") from e


def notebook_edit(input_params: NotebookEditInput, state: FileEditState) -> str:
    """
    Edit Jupyter notebook (.ipynb) cells with support for replace, insert, and delete operations.

    Parses the notebook JSON, modifies the cells array based on edit_mode, and
    preserves all notebook metadata and structure.

    Edit modes:
    - replace: Replace the content of the cell with cell_id
    - insert: Insert a new cell after the cell with cell_id (or at beginning if cell_id is None)
    - delete: Delete the cell with cell_id

    Args:
        input_params: Validated notebook edit input parameters
        state: File edit state tracker

    Returns:
        Success message with operation details

    Raises:
        FileNotReadError: If notebook hasn't been read
        CellNotFoundError: If specified cell_id not found
        NotebookEditError: If notebook parsing or editing fails
    """
    notebook_path = validate_absolute_path(input_params.notebook_path)

    # Notebook must have been read
    if not state.was_read(str(notebook_path)):
        raise FileNotReadError(
            f"Notebook must be read before editing: {notebook_path}\n"
            f"Please read the notebook first to understand its structure."
        )

    try:
        # Read and parse notebook
        content = read_file_content(notebook_path)
        notebook = json.loads(content)

        # Validate notebook structure
        if "cells" not in notebook:
            raise NotebookEditError("Invalid notebook: missing 'cells' array")

        cells = notebook["cells"]

        # Handle different edit modes
        match input_params.edit_mode:
            case "replace":
                # Find and replace cell
                cell_found = False
                for cell in cells:
                    if cell.get("id") == input_params.cell_id:
                        cell["source"] = input_params.new_source
                        if input_params.cell_type:
                            cell["cell_type"] = input_params.cell_type
                        cell_found = True
                        break

                if not cell_found:
                    raise CellNotFoundError(f"Cell with id '{input_params.cell_id}' not found")

                result_msg = f"Replaced content of cell '{input_params.cell_id}'"

            case "insert":
                # Create new cell
                import hashlib

                cell_id_hash = hashlib.md5(input_params.new_source.encode()).hexdigest()[:16]

                new_cell: dict[str, Any] = {
                    "cell_type": input_params.cell_type,
                    "metadata": {},
                    "source": input_params.new_source,
                    "id": f"cell_{cell_id_hash}",
                }

                # Add cell-type specific fields
                if input_params.cell_type == "code":
                    new_cell["execution_count"] = None
                    new_cell["outputs"] = []

                # Insert cell
                if input_params.cell_id is None:
                    # Insert at beginning
                    cells.insert(0, new_cell)
                    result_msg = f"Inserted new {input_params.cell_type} cell at beginning"
                else:
                    # Find insertion point
                    insert_index = None
                    for i, cell in enumerate(cells):
                        if cell.get("id") == input_params.cell_id:
                            insert_index = i + 1
                            break

                    if insert_index is None:
                        raise CellNotFoundError(f"Cell with id '{input_params.cell_id}' not found")

                    cells.insert(insert_index, new_cell)
                    result_msg = f"Inserted new {input_params.cell_type} cell after '{input_params.cell_id}'"

            case "delete":
                # Find and delete cell
                cell_index = None
                for i, cell in enumerate(cells):
                    if cell.get("id") == input_params.cell_id:
                        cell_index = i
                        break

                if cell_index is None:
                    raise CellNotFoundError(f"Cell with id '{input_params.cell_id}' not found")

                cells.pop(cell_index)
                result_msg = f"Deleted cell '{input_params.cell_id}'"

        # Write back to file with proper formatting
        notebook_json = json.dumps(notebook, indent=1, ensure_ascii=False)
        write_file_content(notebook_path, notebook_json + "\n")

        return f"Successfully edited notebook: {notebook_path}\n{result_msg}"

    except json.JSONDecodeError as e:
        raise NotebookEditError(f"Invalid notebook JSON: {e}") from e
    except Exception as e:
        if isinstance(e, (FileNotReadError, CellNotFoundError, NotebookEditError)):
            raise
        raise NotebookEditError(f"Notebook edit failed: {e}") from e
