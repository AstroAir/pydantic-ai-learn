# Task Planning Toolkit for PydanticAI

A comprehensive, production-grade Python implementation of task planning tools designed for seamless integration with PydanticAI agents.

## 🛠️ Tools Overview

### 1. TodoWrite Tool (`todo_write_tool`)
Create and manage structured task lists with state tracking and business rule validation.

**Features:**
- Enforces business rules (single in_progress task, unique IDs)
- Persistent state management across sessions
- Comprehensive validation with meaningful error messages
- Structured task summaries with progress tracking

**Business Rules:**
- Only ONE task can be `in_progress` at any time
- Task IDs must be unique within a task list
- Tasks must have content (min length 1), status, and id

### 2. Task Tool (`task_tool`)
Launch specialized subagents to handle complex, multi-step tasks autonomously.

**Subagent Types:**
- `general-purpose`: Research, code search, multi-step tasks (Tools: all)
- `statusline-setup`: Configure Claude Code status line (Tools: Read, Edit)
- `output-style-setup`: Create Claude Code output style (Tools: Read, Write, Edit, Glob, Grep)

## ✨ Features

- **Modern Python 3.12+** with latest type hints (using `|` for unions)
- **Pydantic v2 validation** for robust input handling
- **State tracking** for task lists with business rule validation
- **Subagent launching** for complex multi-step operations
- **Comprehensive error handling** with informative messages
- **Async/await patterns** for concurrent execution
- **Thread-safe state management** with session isolation
- **35+ comprehensive tests** with full coverage

## 🚀 Quick Start

### Installation

The toolkit requires:
- Python 3.12+
- pydantic-ai>=1.0.15
- Pydantic v2

### Basic Usage

```python
from tools.task_planning_toolkit import (
    TaskListState,
    TodoWriteInput,
    TodoItem,
    todo_write,
    todo_write_tool,
    task_tool
)
from pydantic_ai import Agent

# Create state tracker
state = TaskListState()

# Create agent with task planning tools
agent = Agent('openai:gpt-4', deps_type=TaskListState)

# Register tools
agent.include_tools(todo_write_tool, task_tool)

# Create initial task list
todos = [
    TodoItem(
        content="Research project requirements",
        status="completed",
        id="research-001"
    ),
    TodoItem(
        content="Implement core functionality",
        status="in_progress",
        id="implement-002"
    ),
    TodoItem(
        content="Write comprehensive tests",
        status="pending",
        id="tests-003"
    )
]

# Update task list
result = todo_write(TodoWriteInput(todos=todos), state)
print(result)
# Output:
# Successfully updated task list with 3 task(s).
# - 1 completed
# - 1 in progress: Implement core functionality
# - 1 pending

# Use with agent
result = agent.run_sync('Update the task status', deps=state)
```

## 📖 Detailed Examples

### 1. TodoWrite Tool
Create and manage structured task lists for tracking progress during coding sessions.

**Features:**
- Track task states: `pending`, `in_progress`, `completed`
- Validate business rules (only one in_progress task, unique IDs)
- CRUD operations on task lists
- Clear success/error messages
- State persistence across tool calls

**Business Rules:**
- **Only ONE task can be `in_progress` at any time**
- Task IDs must be unique within a task list
- Each task must have: `content` (min length 1), `status`, and `id`
- Mark tasks `in_progress` BEFORE starting work
- Mark `completed` IMMEDIATELY after finishing
- Only mark completed when FULLY accomplished (no errors, tests passing)

**Parameters:**
- `todos` (array, required): List of todo items
  - Each item has:
    - `content` (string, required, min length 1): Task description
    - `status` (enum, required): `pending`, `in_progress`, or `completed`
    - `id` (string, required): Unique identifier

**Example:**
```python
from tools.task_planning_toolkit import todo_write, TodoWriteInput, TodoItem, TaskListState

state = TaskListState()

todos = [
    TodoItem(content="Research API design", status="completed", id="task-1"),
    TodoItem(content="Implement endpoints", status="in_progress", id="task-2"),
    TodoItem(content="Write tests", status="pending", id="task-3")
]

result = todo_write(TodoWriteInput(todos=todos), state)
print(result)
# Output:
# Successfully updated task list with 3 task(s).
# - 1 completed
# - 1 in progress: Implement endpoints
# - 1 pending
```

### 2. Task Tool (Subagent Launcher)
Launch specialized subagents to handle complex, multi-step tasks autonomously.

**Features:**
- Support for multiple subagent types with different capabilities
- Stateless invocations (one-shot communication)
- Async execution for concurrent agent launches
- Detailed task descriptions and prompts

**Subagent Types:**

| Type | Description | Available Tools |
|------|-------------|----------------|
| `general-purpose` | Research, code search, multi-step tasks | All tools |
| `statusline-setup` | Configure Claude Code status line | Read, Edit |
| `output-style-setup` | Create Claude Code output style | Read, Write, Edit, Glob, LS, Grep |

**Parameters:**
- `description` (string, required, 3-5 words): Short task description
- `prompt` (string, required, min length 10): Detailed task instructions
- `subagent_type` (enum, required): Agent type to use

**Usage Guidelines:**
- Launch multiple agents concurrently when possible
- Provide highly detailed, autonomous task descriptions
- Specify whether agent should write code or just research
- Agent returns single final message (not visible to user)
- Summarize agent results back to user

**Example:**
```python
from tools.task_planning_toolkit import task_launcher, TaskInput

result = await task_launcher(
    TaskInput(
        description="Research API patterns",
        prompt="Research best practices for REST API design, including versioning, authentication, and error handling. Provide a comprehensive summary.",
        subagent_type="general-purpose"
    )
)
print(result)
```

## PydanticAI Integration

### Registering Tools with an Agent

```python
from pydantic_ai import Agent, RunContext
from tools.task_planning_toolkit import (
    TaskListState,
    TodoItem,
    TodoWriteInput,
    TaskInput,
    todo_write,
    task_launcher,
)

# Create agent with TaskListState as dependency
agent = Agent(
    'openai:gpt-4',
    deps_type=TaskListState,
    system_prompt="You are a helpful task planning assistant."
)

# Register TodoWrite tool
@agent.tool
def manage_todos(
    ctx: RunContext[TaskListState],
    todos: list[dict[str, str]]
) -> str:
    """Create and manage a structured task list."""
    todo_items = [TodoItem(**t) for t in todos]
    return todo_write(TodoWriteInput(todos=todo_items), ctx.deps)

# Register Task launcher tool
@agent.tool
async def launch_subagent(
    ctx: RunContext[TaskListState],
    description: str,
    prompt: str,
    subagent_type: str
) -> str:
    """Launch specialized subagents for complex tasks."""
    return await task_launcher(
        TaskInput(
            description=description,
            prompt=prompt,
            subagent_type=subagent_type
        )
    )

# Use the agent
state = TaskListState()
result = agent.run_sync('Create a task list for implementing a REST API', deps=state)
```

## State Management

The `TaskListState` class tracks the task list across tool calls:

```python
from tools.task_planning_toolkit import TaskListState, TodoItem

# Create state tracker
state = TaskListState()

# Update tasks
tasks = [TodoItem(content="Task 1", status="pending", id="id-1")]
state.update_tasks(tasks)

# Get current tasks
current_tasks = state.get_tasks()

# Get in-progress task
in_progress = state.get_in_progress_task()

# Get task by ID
task = state.get_task_by_id("id-1")

# Clear all tasks
state.clear()
```

## Helper Functions

### Generate Task ID

```python
from tools.task_planning_toolkit import generate_task_id

# Generate unique task ID
task_id = generate_task_id()
```

### Validation Functions

```python
from tools.task_planning_toolkit import (
    validate_unique_ids,
    validate_single_in_progress,
    TodoItem
)

tasks = [
    TodoItem(content="Task 1", status="pending", id="id-1"),
    TodoItem(content="Task 2", status="in_progress", id="id-2")
]

# Validate unique IDs
validate_unique_ids(tasks)  # Raises DuplicateTaskIDError if duplicates found

# Validate single in_progress
validate_single_in_progress(tasks)  # Raises MultipleInProgressError if multiple in_progress
```

## Error Handling

The toolkit provides comprehensive error handling with custom exceptions:

- `TaskPlanningError`: Base exception for all task planning errors
- `TodoValidationError`: Todo validation failures
- `MultipleInProgressError`: Multiple tasks marked as in_progress
- `DuplicateTaskIDError`: Duplicate task IDs detected
- `SubagentError`: Subagent operation failures
- `InvalidSubagentTypeError`: Invalid subagent type specified

**Example:**
```python
from tools.task_planning_toolkit import (
    todo_write,
    TodoWriteInput,
    TodoItem,
    TaskListState,
    MultipleInProgressError,
    DuplicateTaskIDError
)

state = TaskListState()

try:
    todos = [
        TodoItem(content="Task 1", status="in_progress", id="id-1"),
        TodoItem(content="Task 2", status="in_progress", id="id-2")
    ]
    todo_write(TodoWriteInput(todos=todos), state)
except MultipleInProgressError as e:
    print(f"Validation error: {e}")
    # Output: Multiple tasks marked as in_progress: id-1, id-2. Only ONE task can be in_progress at any time.
```

## Usage Patterns

### Pattern 1: Simple Task Tracking

```python
# Create task list for a feature
todos = [
    TodoItem(content="Research requirements", status="completed", id="task-1"),
    TodoItem(content="Implement feature", status="in_progress", id="task-2"),
    TodoItem(content="Write tests", status="pending", id="task-3")
]

result = todo_write(TodoWriteInput(todos=todos), state)
```

### Pattern 2: Updating Task Status

```python
# Mark current task as completed, start next task
updated_todos = [
    TodoItem(content="Research requirements", status="completed", id="task-1"),
    TodoItem(content="Implement feature", status="completed", id="task-2"),
    TodoItem(content="Write tests", status="in_progress", id="task-3")
]

result = todo_write(TodoWriteInput(todos=updated_todos), state)
```

### Pattern 3: Launching Concurrent Subagents

```python
import asyncio

# Launch multiple subagents concurrently
tasks = [
    task_launcher(TaskInput(
        description="Research API patterns",
        prompt="Research REST API best practices",
        subagent_type="general-purpose"
    )),
    task_launcher(TaskInput(
        description="Setup status line",
        prompt="Configure status line display",
        subagent_type="statusline-setup"
    ))
]

results = await asyncio.gather(*tasks)
```

## Examples

See `task_planning_toolkit_example.py` for comprehensive usage examples including:
- Basic todo list management
- Updating task status
- Validation error handling
- Launching subagents
- Complete workflow examples

## Requirements

- Python 3.12+
- pydantic >= 2.0
- pydantic-ai >= 1.0.15

## License

Part of the pydantic-ai-learn project.
