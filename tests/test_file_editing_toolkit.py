"""
Tests for File Editing Toolkit

Comprehensive tests for all four file editing tools:
- Edit
- MultiEdit
- Write
- NotebookEdit
"""

import json
import tempfile
from pathlib import Path

import pytest

from tools.file_editing_toolkit import (
    CellNotFoundError,
    EditInput,
    FileEditState,
    FileNotReadError,
    IdenticalStringsError,
    MultiEditError,
    MultiEditInput,
    NotebookEditInput,
    SingleEdit,
    StringNotFoundError,
    StringNotUniqueError,
    WriteInput,
    edit_file,
    multi_edit_file,
    notebook_edit,
    write_file,
)

# ============================================================================
# Test Edit Tool
# ============================================================================


def test_edit_file_basic():
    """Test basic file editing."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("def old_function():\n    pass\n")
        temp_path = f.name

    try:
        # Mark as read
        state.mark_as_read(temp_path)

        # Edit file
        result = edit_file(
            EditInput(
                file_path=temp_path,
                old_string="def old_function():",
                new_string="def new_function():",
                replace_all=False,
            ),
            state,
        )

        assert "Successfully edited" in result

        # Verify content
        content = Path(temp_path).read_text()
        assert "def new_function():" in content
        assert "def old_function():" not in content
    finally:
        Path(temp_path).unlink()


def test_edit_file_not_read():
    """Test that editing fails if file not read."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("content")
        temp_path = f.name

    try:
        with pytest.raises(FileNotReadError):
            edit_file(EditInput(file_path=temp_path, old_string="content", new_string="new content"), state)
    finally:
        Path(temp_path).unlink()


def test_edit_file_string_not_found():
    """Test that editing fails if old_string not found."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("content")
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        with pytest.raises(StringNotFoundError):
            edit_file(EditInput(file_path=temp_path, old_string="nonexistent", new_string="new"), state)
    finally:
        Path(temp_path).unlink()


def test_edit_file_not_unique():
    """Test that editing fails if old_string appears multiple times."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("test\ntest\n")
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        with pytest.raises(StringNotUniqueError):
            edit_file(EditInput(file_path=temp_path, old_string="test", new_string="new", replace_all=False), state)
    finally:
        Path(temp_path).unlink()


def test_edit_file_replace_all():
    """Test replace_all functionality."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("test\ntest\ntest\n")
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        result = edit_file(EditInput(file_path=temp_path, old_string="test", new_string="new", replace_all=True), state)

        assert "Successfully edited" in result
        content = Path(temp_path).read_text()
        assert content.count("new") == 3
        assert "test" not in content
    finally:
        Path(temp_path).unlink()


def test_edit_file_identical_strings():
    """Test that editing fails if strings are identical."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("content")
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        with pytest.raises(IdenticalStringsError):
            edit_file(EditInput(file_path=temp_path, old_string="content", new_string="content"), state)
    finally:
        Path(temp_path).unlink()


# ============================================================================
# Test MultiEdit Tool
# ============================================================================


def test_multi_edit_basic():
    """Test basic multi-edit functionality."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("DEBUG = False\nPORT = 8000\n")
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        result = multi_edit_file(
            MultiEditInput(
                file_path=temp_path,
                edits=[
                    SingleEdit(old_string="DEBUG = False", new_string="DEBUG = True"),
                    SingleEdit(old_string="PORT = 8000", new_string="PORT = 3000"),
                ],
            ),
            state,
        )

        assert "Successfully applied 2 edit(s)" in result

        content = Path(temp_path).read_text()
        assert "DEBUG = True" in content
        assert "PORT = 3000" in content
    finally:
        Path(temp_path).unlink()


def test_multi_edit_sequential():
    """Test that edits are applied sequentially."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("value = 1\n")
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        _result = multi_edit_file(
            MultiEditInput(
                file_path=temp_path,
                edits=[
                    SingleEdit(old_string="value = 1", new_string="value = 2"),
                    SingleEdit(old_string="value = 2", new_string="value = 3"),
                ],
            ),
            state,
        )

        content = Path(temp_path).read_text()
        assert "value = 3" in content
    finally:
        Path(temp_path).unlink()


def test_multi_edit_atomic_rollback():
    """Test that failed multi-edit doesn't partially apply."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        original_content = "DEBUG = False\nPORT = 8000\n"
        f.write(original_content)
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        # Second edit will fail (string not found)
        with pytest.raises(MultiEditError):
            multi_edit_file(
                MultiEditInput(
                    file_path=temp_path,
                    edits=[
                        SingleEdit(old_string="DEBUG = False", new_string="DEBUG = True"),
                        SingleEdit(old_string="NONEXISTENT", new_string="NEW"),
                    ],
                ),
                state,
            )

        # Content should be unchanged (rollback)
        content = Path(temp_path).read_text()
        assert content == original_content
    finally:
        Path(temp_path).unlink()


# ============================================================================
# Test Write Tool
# ============================================================================


def test_write_new_file():
    """Test writing a new file."""
    state = FileEditState()

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "new_file.py"

        result = write_file(WriteInput(file_path=str(temp_path), content="print('Hello, World!')\n"), state)

        assert "Created file" in result
        assert temp_path.exists()
        assert temp_path.read_text() == "print('Hello, World!')\n"


def test_write_existing_file_not_read():
    """Test that overwriting existing file fails if not read."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("original content")
        temp_path = f.name

    try:
        with pytest.raises(FileNotReadError):
            write_file(WriteInput(file_path=temp_path, content="new content"), state)
    finally:
        Path(temp_path).unlink()


def test_write_existing_file_after_read():
    """Test overwriting existing file after reading."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("original content")
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        result = write_file(WriteInput(file_path=temp_path, content="new content"), state)

        assert "Overwritten file" in result
        assert Path(temp_path).read_text() == "new content"
    finally:
        Path(temp_path).unlink()


# ============================================================================
# Test NotebookEdit Tool
# ============================================================================


def create_test_notebook():
    """Create a test notebook structure."""
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_1",
                "metadata": {},
                "outputs": [],
                "source": "print('Cell 1')",
            },
            {"cell_type": "markdown", "id": "cell_2", "metadata": {}, "source": "# Markdown Cell"},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_notebook_edit_replace():
    """Test replacing notebook cell content."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ipynb") as f:
        json.dump(create_test_notebook(), f)
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        result = notebook_edit(
            NotebookEditInput(
                notebook_path=temp_path, cell_id="cell_1", new_source="print('Updated')", edit_mode="replace"
            ),
            state,
        )

        assert "Replaced content" in result

        notebook = json.loads(Path(temp_path).read_text())
        assert notebook["cells"][0]["source"] == "print('Updated')"
    finally:
        Path(temp_path).unlink()


def test_notebook_edit_insert():
    """Test inserting new notebook cell."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ipynb") as f:
        json.dump(create_test_notebook(), f)
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        result = notebook_edit(
            NotebookEditInput(
                notebook_path=temp_path,
                cell_id="cell_1",
                new_source="print('New cell')",
                cell_type="code",
                edit_mode="insert",
            ),
            state,
        )

        assert "Inserted new" in result

        notebook = json.loads(Path(temp_path).read_text())
        assert len(notebook["cells"]) == 3
        assert notebook["cells"][1]["source"] == "print('New cell')"
    finally:
        Path(temp_path).unlink()


def test_notebook_edit_delete():
    """Test deleting notebook cell."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ipynb") as f:
        json.dump(create_test_notebook(), f)
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        result = notebook_edit(NotebookEditInput(notebook_path=temp_path, cell_id="cell_1", edit_mode="delete"), state)

        assert "Deleted cell" in result

        notebook = json.loads(Path(temp_path).read_text())
        assert len(notebook["cells"]) == 1
        assert notebook["cells"][0]["id"] == "cell_2"
    finally:
        Path(temp_path).unlink()


def test_notebook_edit_cell_not_found():
    """Test that editing fails if cell_id not found."""
    state = FileEditState()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ipynb") as f:
        json.dump(create_test_notebook(), f)
        temp_path = f.name

    try:
        state.mark_as_read(temp_path)

        with pytest.raises(CellNotFoundError):
            notebook_edit(
                NotebookEditInput(
                    notebook_path=temp_path, cell_id="nonexistent", new_source="content", edit_mode="replace"
                ),
                state,
            )
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
