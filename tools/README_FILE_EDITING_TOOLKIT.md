# File Editing Toolkit for PydanticAI

A comprehensive, production-grade Python implementation of file editing tools designed for seamless integration with PydanticAI agents.

## Features

- **Modern Python 3.12+** with latest type hints (using `|` for unions)
- **Pydantic v2 validation** for robust input handling
- **State tracking** to ensure files are read before editing
- **Atomic operations** for MultiEdit (all succeed or all fail)
- **Comprehensive error handling** with informative messages
- **Security-focused design** (path validation, absolute paths only)

## Tools Overview

### 1. Edit Tool
Perform exact string replacements in files with strict validation.

**Features:**
- Tracks whether files have been read before allowing edits
- Validates that `old_string` exists and is unique (unless `replace_all=True`)
- Ensures `old_string` and `new_string` are different
- Preserves exact whitespace and indentation
- Handles line number prefixes correctly

**Parameters:**
- `file_path` (str, required): Absolute path to the file to modify
- `old_string` (str, required): The exact text to replace
- `new_string` (str, required): The replacement text
- `replace_all` (bool, optional, default=False): Whether to replace all occurrences

**Example:**
```python
from file_editing_toolkit import edit_file, EditInput, FileEditState

state = FileEditState()
state.mark_as_read("/path/to/file.py")

result = edit_file(
    EditInput(
        file_path="/path/to/file.py",
        old_string="def old_function():",
        new_string="def new_function():",
        replace_all=False
    ),
    state
)
```

### 2. MultiEdit Tool
Perform multiple sequential edit operations on a single file atomically.

**Features:**
- Built on top of the Edit tool functionality
- Apply edits sequentially in the provided order
- Each edit operates on the result of the previous edit
- Atomic transactions: all edits succeed or none are applied
- Validate all edits before applying any
- Support file creation when first edit has empty `old_string`

**Parameters:**
- `file_path` (str, required): Absolute path to the file to modify
- `edits` (array, required, min 1 item): Array of edit operations

**Example:**
```python
from file_editing_toolkit import multi_edit_file, MultiEditInput, SingleEdit, FileEditState

state = FileEditState()
state.mark_as_read("/path/to/config.py")

result = multi_edit_file(
    MultiEditInput(
        file_path="/path/to/config.py",
        edits=[
            SingleEdit(old_string="DEBUG = False", new_string="DEBUG = True"),
            SingleEdit(old_string="PORT = 8000", new_string="PORT = 3000")
        ]
    ),
    state
)
```

### 3. Write Tool
Write or overwrite files on the local filesystem.

**Features:**
- Overwrite existing files completely
- For existing files, verifies the file was read first
- Create new files when path doesn't exist
- Never proactively creates documentation files

**Parameters:**
- `file_path` (str, required): Absolute path to the file to write
- `content` (str, required): The complete content to write

**Example:**
```python
from file_editing_toolkit import write_file, WriteInput, FileEditState

state = FileEditState()

result = write_file(
    WriteInput(
        file_path="/path/to/new_file.py",
        content="#!/usr/bin/env python3\n\nprint('Hello, World!')\n"
    ),
    state
)
```

### 4. NotebookEdit Tool
Edit Jupyter notebook (.ipynb) cells with support for replace, insert, and delete operations.

**Features:**
- Parse and modify Jupyter notebook JSON structure
- Support three edit modes: replace, insert, delete
- Handle both code and markdown cell types
- Use cell IDs for precise targeting
- Preserve notebook metadata and structure

**Parameters:**
- `notebook_path` (str, required): Absolute path to the .ipynb file
- `cell_id` (str, optional): ID of the target cell
- `new_source` (str, required for replace/insert): The new source content
- `cell_type` (enum: "code" | "markdown", optional): Cell type
- `edit_mode` (enum: "replace" | "insert" | "delete", default="replace"): Edit operation

**Example:**
```python
from file_editing_toolkit import notebook_edit, NotebookEditInput, FileEditState

state = FileEditState()
state.mark_as_read("/path/to/notebook.ipynb")

# Replace cell content
result = notebook_edit(
    NotebookEditInput(
        notebook_path="/path/to/notebook.ipynb",
        cell_id="abc123",
        new_source="print('Updated code')",
        edit_mode="replace"
    ),
    state
)

# Insert new cell
result = notebook_edit(
    NotebookEditInput(
        notebook_path="/path/to/notebook.ipynb",
        cell_id="abc123",
        new_source="# New markdown cell",
        cell_type="markdown",
        edit_mode="insert"
    ),
    state
)

# Delete cell
result = notebook_edit(
    NotebookEditInput(
        notebook_path="/path/to/notebook.ipynb",
        cell_id="abc123",
        edit_mode="delete"
    ),
    state
)
```

## PydanticAI Integration

### Registering Tools with an Agent

```python
from pydantic_ai import Agent, RunContext
from file_editing_toolkit import (
    FileEditState,
    EditInput,
    edit_file,
)

# Create agent with FileEditState as dependency
agent = Agent(
    'openai:gpt-4',
    deps_type=FileEditState,
    system_prompt="You are a helpful file editing assistant."
)

# Register the edit tool
@agent.tool
def edit(
    ctx: RunContext[FileEditState],
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False
) -> str:
    """
    Perform exact string replacement in a file.

    Args:
        file_path: Absolute path to the file to modify
        old_string: The exact text to replace
        new_string: The replacement text
        replace_all: Whether to replace all occurrences
    """
    return edit_file(
        EditInput(
            file_path=file_path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all
        ),
        ctx.deps
    )

# Use the agent
state = FileEditState()
result = agent.run_sync('Edit the file...', deps=state)
```

## State Management

The `FileEditState` class tracks which files have been read during the conversation:

```python
from file_editing_toolkit import FileEditState

# Create state tracker
state = FileEditState()

# Mark file as read
state.mark_as_read("/path/to/file.py")

# Check if file was read
if state.was_read("/path/to/file.py"):
    print("File has been read")

# Clear all tracking
state.clear()
```

## Error Handling

The toolkit provides comprehensive error handling with custom exceptions:

- `FileEditError`: Base exception for all file editing errors
- `EditValidationError`: Edit validation failures
- `FileNotReadError`: Attempting to edit unread file
- `StringNotFoundError`: old_string not found in file
- `StringNotUniqueError`: old_string appears multiple times
- `IdenticalStringsError`: old_string and new_string are identical
- `MultiEditError`: Multi-edit operation failures
- `WriteError`: Write operation failures
- `NotebookEditError`: Notebook editing failures
- `CellNotFoundError`: Specified cell_id not found

## Security Considerations

- **Absolute paths required**: Prevents ambiguity and path traversal attacks
- **Path validation**: All paths are normalized and validated
- **Read tracking**: Prevents blind modifications of files
- **Proper error handling**: Prevents information leakage

## Examples

See `file_editing_toolkit_example.py` for comprehensive usage examples including:
- Basic file editing
- Multi-edit operations
- Writing new files
- Notebook cell manipulation

## Requirements

- Python 3.12+
- pydantic >= 2.0
- pydantic-ai >= 1.0.15

## License

Part of the pydantic-ai-learn project.
