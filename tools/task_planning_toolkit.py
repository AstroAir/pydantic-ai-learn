"""
Task Planning Toolkit for PydanticAI

A comprehensive, production-grade Python implementation of task planning tools
designed for seamless integration with PydanticAI agents.

Tools:
1. TodoWrite: Create and manage structured task lists with state tracking
2. Task: Launch specialized subagents for complex multi-step operations

Features:
- Modern Python 3.12+ with latest type hints (using | for unions)
- Pydantic v2 validation for robust input handling
- State tracking for task lists with validation rules
- Support for multiple subagent types with different capabilities
- Comprehensive error handling with informative messages
- Async/await patterns for concurrent subagent execution

Business Rules:
- Only ONE task can be in_progress at any time
- Task IDs must be unique within a task list
- Tasks must have: content (min length 1), status, and id
- Subagents are stateless (one-shot communication)

Example Usage:
    ```python
    from tools.task_planning_toolkit import TaskListState, todo_write, TodoWriteInput, TodoItem
    from pydantic_ai import Agent, RunContext

    # Create state tracker
    state = TaskListState()

    # Create agent with task planning tools
    agent = Agent('openai:gpt-4', deps_type=TaskListState)

    # Register tools
    @agent.tool
    def manage_todos(ctx: RunContext[TaskListState], todos: list[dict]) -> str:
        todo_items = [TodoItem(**t) for t in todos]
        return todo_write(TodoWriteInput(todos=todo_items), ctx.deps)

    # Use the agent
    result = agent.run_sync('Create a task list...', deps=state)
    ```

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# Custom Exceptions
# ============================================================================


class TaskPlanningError(Exception):
    """Base exception for task planning tool errors."""

    pass


class TodoValidationError(TaskPlanningError):
    """Raised when todo validation fails."""

    pass


class MultipleInProgressError(TodoValidationError):
    """Raised when multiple tasks are marked as in_progress."""

    pass


class DuplicateTaskIDError(TodoValidationError):
    """Raised when duplicate task IDs are detected."""

    pass


class SubagentError(TaskPlanningError):
    """Raised when subagent operation fails."""

    pass


class InvalidSubagentTypeError(SubagentError):
    """Raised when invalid subagent type is specified."""

    pass


# ============================================================================
# Type Definitions
# ============================================================================

TaskStatus = Literal["pending", "in_progress", "completed"]
SubagentType = Literal["general-purpose", "statusline-setup", "output-style-setup"]


# ============================================================================
# State Management
# ============================================================================


@dataclass
class TaskListState:
    """
    State tracker for task list management.

    Maintains the current task list with validation for business rules:
    - Only one task can be in_progress at a time
    - Task IDs must be unique

    Attributes:
        tasks: List of current tasks
    """

    tasks: list[TodoItem] = field(default_factory=list)

    def update_tasks(self, new_tasks: list[TodoItem]) -> None:
        """Update the task list with new tasks."""
        self.tasks = new_tasks

    def get_tasks(self) -> list[TodoItem]:
        """Get the current task list."""
        return self.tasks

    def clear(self) -> None:
        """Clear all tasks."""
        self.tasks.clear()

    def get_in_progress_task(self) -> TodoItem | None:
        """Get the currently in-progress task, if any."""
        for task in self.tasks:
            if task.status == "in_progress":
                return task
        return None

    def get_task_by_id(self, task_id: str) -> TodoItem | None:
        """Get a task by its ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None


# ============================================================================
# Helper Functions
# ============================================================================


def generate_task_id() -> str:
    """
    Generate a unique task ID using UUID4.

    Returns:
        A unique task ID string
    """
    return str(uuid.uuid4())


def validate_unique_ids(tasks: list[TodoItem]) -> None:
    """
    Validate that all task IDs are unique.

    Args:
        tasks: List of tasks to validate

    Raises:
        DuplicateTaskIDError: If duplicate IDs are found
    """
    ids = [task.id for task in tasks]
    unique_ids = set(ids)

    if len(ids) != len(unique_ids):
        # Find duplicates
        seen = set()
        duplicates = set()
        for task_id in ids:
            if task_id in seen:
                duplicates.add(task_id)
            seen.add(task_id)

        raise DuplicateTaskIDError(
            f"Duplicate task IDs found: {', '.join(duplicates)}. Each task must have a unique ID."
        )


def validate_single_in_progress(tasks: list[TodoItem]) -> None:
    """
    Validate that only one task is in_progress.

    Args:
        tasks: List of tasks to validate

    Raises:
        MultipleInProgressError: If multiple tasks are in_progress
    """
    in_progress_tasks = [task for task in tasks if task.status == "in_progress"]

    if len(in_progress_tasks) > 1:
        in_progress_ids = [task.id for task in in_progress_tasks]
        raise MultipleInProgressError(
            f"Multiple tasks marked as in_progress: {', '.join(in_progress_ids)}. "
            f"Only ONE task can be in_progress at any time. "
            f"Please mark other tasks as 'pending' or 'completed'."
        )


# ============================================================================
# Pydantic Input Models
# ============================================================================


class TodoItem(BaseModel):
    """
    A single todo item with content, status, and unique ID.

    Attributes:
        content: Task description (required, min length 1)
        status: Task status (pending, in_progress, or completed)
        id: Unique identifier for the task
    """

    content: str = Field(..., min_length=1, description="Task description")
    status: TaskStatus = Field(..., description="Task status: pending, in_progress, or completed")
    id: str = Field(..., min_length=1, description="Unique identifier for the task")

    model_config = {"extra": "forbid"}

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Validate that content is not just whitespace."""
        if not v.strip():
            raise ValueError("Task content cannot be empty or just whitespace")
        return v


class TodoWriteInput(BaseModel):
    """
    Input model for TodoWrite tool.

    Validates business rules:
    - Only one task can be in_progress
    - All task IDs must be unique

    Attributes:
        todos: List of todo items
    """

    todos: list[TodoItem] = Field(..., min_length=0, description="List of todo items")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_business_rules(self) -> TodoWriteInput:
        """Validate business rules for the task list."""
        # Validate unique IDs
        validate_unique_ids(self.todos)

        # Validate single in_progress
        validate_single_in_progress(self.todos)

        return self


class TaskInput(BaseModel):
    """
    Input model for Task (subagent launcher) tool.

    Attributes:
        description: Short 3-5 word task description
        prompt: Detailed task instructions for the agent
        subagent_type: Type of subagent to launch
    """

    description: str = Field(..., min_length=1, max_length=100, description="Short 3-5 word task description")
    prompt: str = Field(..., min_length=10, description="Detailed task instructions for the agent")
    subagent_type: SubagentType = Field(
        ..., description="Agent type: general-purpose, statusline-setup, or output-style-setup"
    )

    model_config = {"extra": "forbid"}

    @field_validator("description")
    @classmethod
    def validate_description_concise(cls, v: str) -> str:
        """Validate that description is concise (3-5 words recommended)."""
        word_count = len(v.split())
        if word_count > 10:
            raise ValueError(
                f"Description should be concise (3-5 words recommended). "
                f"Got {word_count} words. Use 'prompt' field for detailed instructions."
            )
        return v


# ============================================================================
# Core Tool Functions
# ============================================================================


def todo_write(input_params: TodoWriteInput, state: TaskListState) -> str:
    """
    Create and manage a structured task list for tracking progress.

    This function updates the task list state with the provided todos,
    validates business rules, and returns a summary of the operation.

    Business Rules:
        - Only ONE task can be in_progress at any time
        - Task IDs must be unique
        - Each task must have: content (min length 1), status, and id

    Args:
        input_params: Validated todo input with list of tasks
        state: Task list state tracker

    Returns:
        Success message with task list summary

    Raises:
        TodoValidationError: If validation fails
        MultipleInProgressError: If multiple tasks are in_progress
        DuplicateTaskIDError: If duplicate task IDs are found

    Example:
        >>> state = TaskListState()
        >>> todos = [
        ...     TodoItem(content="Research API", status="completed", id="task-1"),
        ...     TodoItem(content="Implement", status="in_progress", id="task-2"),
        ...     TodoItem(content="Test", status="pending", id="task-3")
        ... ]
        >>> result = todo_write(TodoWriteInput(todos=todos), state)
        >>> print(result)
        Successfully updated task list with 3 task(s).
        - 1 completed
        - 1 in progress: Implement
        - 1 pending
    """
    try:
        # Update state with new tasks
        state.update_tasks(input_params.todos)

        # Generate summary
        total = len(input_params.todos)
        completed = sum(1 for t in input_params.todos if t.status == "completed")
        in_progress = sum(1 for t in input_params.todos if t.status == "in_progress")
        pending = sum(1 for t in input_params.todos if t.status == "pending")

        # Get in-progress task name if exists
        in_progress_task = state.get_in_progress_task()
        in_progress_name = in_progress_task.content if in_progress_task else None

        # Build result message
        message_parts = [f"Successfully updated task list with {total} task(s)."]

        if completed > 0:
            message_parts.append(f"- {completed} completed")

        if in_progress > 0 and in_progress_name:
            message_parts.append(f"- {in_progress} in progress: {in_progress_name}")
        elif in_progress > 0:
            message_parts.append(f"- {in_progress} in progress")

        if pending > 0:
            message_parts.append(f"- {pending} pending")

        return "\n".join(message_parts)

    except (MultipleInProgressError, DuplicateTaskIDError):
        # Re-raise validation errors
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise TodoValidationError(f"Failed to update task list: {e}") from e


async def task_launcher(input_params: TaskInput) -> str:
    """
    Launch specialized subagents to handle complex, multi-step tasks autonomously.

    This function creates and executes a subagent with the specified type and prompt.
    Subagents are stateless and return a single final message.

    Subagent Types:
        - general-purpose: Research, code search, multi-step tasks (Tools: all)
        - statusline-setup: Configure Claude Code status line (Tools: Read, Edit)
        - output-style-setup: Create Claude Code output style (Tools: Read, Write, Edit, Glob, LS, Grep)

    Args:
        input_params: Validated task input with description, prompt, and subagent type

    Returns:
        Summary of subagent execution results

    Raises:
        InvalidSubagentTypeError: If invalid subagent type is specified
        SubagentError: If subagent execution fails

    Example:
        >>> result = await task_launcher(TaskInput(
        ...     description="Research API patterns",
        ...     prompt="Research best practices for REST API design",
        ...     subagent_type="general-purpose"
        ... ))
        >>> print(result)
        Subagent 'Research API patterns' (general-purpose) completed successfully.
    """
    try:
        # Validate subagent type (already validated by Pydantic, but double-check)
        valid_types: list[SubagentType] = ["general-purpose", "statusline-setup", "output-style-setup"]
        if input_params.subagent_type not in valid_types:
            raise InvalidSubagentTypeError(
                f"Invalid subagent type: {input_params.subagent_type}. Must be one of: {', '.join(valid_types)}"
            )

        # Import PydanticAI components (lazy import to avoid dependency issues)
        try:
            from pydantic_ai import Agent
        except ImportError as e:
            raise SubagentError(
                "PydanticAI is required for subagent execution. Install with: pip install pydantic-ai"
            ) from e

        # Define system prompts for each subagent type
        system_prompts = {
            "general-purpose": (
                "You are a general-purpose research and development assistant. "
                "You have access to all available tools for code search, file operations, "
                "and analysis. Provide comprehensive, well-researched responses with "
                "specific examples and actionable recommendations."
            ),
            "statusline-setup": (
                "You are a configuration specialist focused on setting up status lines "
                "and UI elements. You have access to file reading and editing tools. "
                "Provide clear, step-by-step configuration instructions with code examples."
            ),
            "output-style-setup": (
                "You are a documentation and output formatting specialist. You have access "
                "to file operations, search tools, and can create comprehensive documentation "
                "structures. Focus on creating well-organized, readable output with proper "
                "formatting and structure."
            ),
        }

        # Create subagent with appropriate system prompt
        system_prompt = system_prompts[input_params.subagent_type]
        subagent = Agent(
            "openai:gpt-4",  # Default model, can be configured
            system_prompt=system_prompt,
        )

        # Execute the subagent with the provided prompt
        result = await subagent.run(input_params.prompt)

        # Format the result
        return (
            f"Subagent '{input_params.description}' ({input_params.subagent_type}) "
            f"completed successfully.\n\n"
            f"Result:\n{result.output}"
        )

    except InvalidSubagentTypeError:
        # Re-raise validation errors
        raise
    except ImportError as e:
        raise SubagentError(f"Failed to import required dependencies: {e}") from e
    except Exception as e:
        # Wrap unexpected errors
        raise SubagentError(f"Failed to launch subagent: {e}") from e


# ============================================================================
# Helper Functions for Task Management
# ============================================================================


def get_task_summary(state: TaskListState) -> str:
    """
    Generate a formatted summary of all tasks in the task list.

    Args:
        state: Task list state tracker

    Returns:
        Formatted string with task summary including counts and details

    Example:
        >>> state = TaskListState()
        >>> state.update_tasks([
        ...     TodoItem(content="Task 1", status="completed", id="id-1"),
        ...     TodoItem(content="Task 2", status="in_progress", id="id-2"),
        ...     TodoItem(content="Task 3", status="pending", id="id-3")
        ... ])
        >>> print(get_task_summary(state))
        Task Summary (3 total):
        - 1 completed
        - 1 in progress: Task 2
        - 1 pending

        Tasks:
        [✓] Task 1 (id-1)
        [→] Task 2 (id-2)
        [ ] Task 3 (id-3)
    """
    tasks = state.get_tasks()

    if not tasks:
        return "No tasks in the list."

    # Count tasks by status
    total = len(tasks)
    completed = sum(1 for t in tasks if t.status == "completed")
    in_progress = sum(1 for t in tasks if t.status == "in_progress")
    pending = sum(1 for t in tasks if t.status == "pending")

    # Get in-progress task name
    in_progress_task = state.get_in_progress_task()
    in_progress_name = in_progress_task.content if in_progress_task else None

    # Build summary header
    summary_parts = [f"Task Summary ({total} total):"]

    if completed > 0:
        summary_parts.append(f"- {completed} completed")

    if in_progress > 0 and in_progress_name:
        summary_parts.append(f"- {in_progress} in progress: {in_progress_name}")
    elif in_progress > 0:
        summary_parts.append(f"- {in_progress} in progress")

    if pending > 0:
        summary_parts.append(f"- {pending} pending")

    # Add task list
    summary_parts.append("\nTasks:")

    for task in tasks:
        if task.status == "completed":
            icon = "[✓]"
        elif task.status == "in_progress":
            icon = "[→]"
        else:  # pending
            icon = "[ ]"

        summary_parts.append(f"{icon} {task.content} ({task.id})")

    return "\n".join(summary_parts)


def mark_task_complete(task_id: str, state: TaskListState) -> str:
    """
    Mark a specific task as completed by its ID.

    Args:
        task_id: ID of the task to mark as completed
        state: Task list state tracker

    Returns:
        Success message indicating the task was marked as completed

    Raises:
        TodoValidationError: If task ID is not found

    Example:
        >>> state = TaskListState()
        >>> state.update_tasks([
        ...     TodoItem(content="Task 1", status="in_progress", id="task-1")
        ... ])
        >>> result = mark_task_complete("task-1", state)
        >>> print(result)
        Task 'Task 1' (task-1) marked as completed.
    """
    task = state.get_task_by_id(task_id)
    if not task:
        raise TodoValidationError(f"Task with ID '{task_id}' not found.")

    if task.status == "completed":
        return f"Task '{task.content}' ({task_id}) is already completed."

    old_status = task.status

    # Create updated task list with the task marked as completed
    updated_tasks = []
    for t in state.get_tasks():
        if t.id == task_id:
            updated_tasks.append(TodoItem(content=t.content, status="completed", id=t.id))
        else:
            updated_tasks.append(t)

    state.update_tasks(updated_tasks)

    status_map = {"pending": "pending", "in_progress": "in progress"}

    old_status_text = status_map.get(old_status, old_status)
    return f"Task '{task.content}' ({task_id}) marked as completed (was {old_status_text})."


def mark_task_in_progress(task_id: str, state: TaskListState) -> str:
    """
    Mark a specific task as in progress by its ID.

    Validates business rules to ensure only one task is in progress at a time.

    Args:
        task_id: ID of the task to mark as in progress
        state: Task list state tracker

    Returns:
        Success message indicating the task was marked as in progress

    Raises:
        TodoValidationError: If task ID is not found or business rules are violated

    Example:
        >>> state = TaskListState()
        >>> state.update_tasks([
        ...     TodoItem(content="Task 1", status="pending", id="task-1")
        ... ])
        >>> result = mark_task_in_progress("task-1", state)
        >>> print(result)
        Task 'Task 1' (task-1) marked as in progress.
    """
    task = state.get_task_by_id(task_id)
    if not task:
        raise TodoValidationError(f"Task with ID '{task_id}' not found.")

    # Check if another task is already in progress
    current_in_progress = state.get_in_progress_task()
    if current_in_progress and current_in_progress.id != task_id:
        raise MultipleInProgressError(
            f"Cannot mark task '{task.content}' as in progress. "
            f"Task '{current_in_progress.content}' ({current_in_progress.id}) is already in progress. "
            f"Only one task can be in progress at a time."
        )

    if task.status == "in_progress":
        return f"Task '{task.content}' ({task_id}) is already in progress."

    old_status = task.status

    # Create updated task list with the task marked as in progress
    updated_tasks = []
    for t in state.get_tasks():
        if t.id == task_id:
            updated_tasks.append(TodoItem(content=t.content, status="in_progress", id=t.id))
        else:
            updated_tasks.append(t)

    state.update_tasks(updated_tasks)

    status_map = {"pending": "pending", "completed": "completed"}

    old_status_text = status_map.get(old_status, old_status)
    return f"Task '{task.content}' ({task_id}) marked as in progress (was {old_status_text})."


def add_task(content: str, state: TaskListState, status: TaskStatus = "pending") -> str:
    """
    Add a new task to the task list.

    Args:
        content: Task description (will be validated)
        state: Task list state tracker
        status: Initial status for the task (default: "pending")

    Returns:
        Success message indicating the task was added with its ID

    Raises:
        TodoValidationError: If content is empty or whitespace

    Example:
        >>> state = TaskListState()
        >>> result = add_task("New task description", state)
        >>> print(result)
        Task 'New task description' added with ID: <generated-uuid>
    """
    # Validate content
    if not content or not content.strip():
        raise TodoValidationError("Task content cannot be empty or just whitespace.")

    # Generate unique ID
    task_id = generate_task_id()

    # Create new task
    new_task = TodoItem(content=content.strip(), status=status, id=task_id)

    # Add to existing tasks
    current_tasks = state.get_tasks()
    updated_tasks = current_tasks + [new_task]
    state.update_tasks(updated_tasks)

    return f"Task '{content}' added with ID: {task_id}"


# ============================================================================
# PydanticAI Tool Registration
# ============================================================================

try:
    from pydantic_ai import RunContext, Tool

    # PydanticAI tool wrapper functions
    def todo_write_func(ctx: RunContext[Any], todos: list[dict[str, Any]]) -> str:
        """PydanticAI tool wrapper for todo_write function."""
        # Convert dict todos to TodoItem objects
        todo_items = [TodoItem(**t) for t in todos]
        input_params = TodoWriteInput(todos=todo_items)

        # Use the state from context if available, otherwise create new
        state = ctx.deps if hasattr(ctx, "deps") and ctx.deps is not None else TaskListState()

        return todo_write(input_params, state)

    async def task_func(ctx: RunContext[Any], description: str, prompt: str, subagent_type: SubagentType) -> str:
        """PydanticAI tool wrapper for task_launcher function."""
        input_params = TaskInput(description=description, prompt=prompt, subagent_type=subagent_type)

        return await task_launcher(input_params)

    # PydanticAI Tool instances
    todo_write_tool = Tool(todo_write_func, takes_ctx=True)
    task_tool = Tool(task_func, takes_ctx=True)

except ImportError:
    # PydanticAI not available - define stub tools
    todo_write_tool = None  # type: ignore[assignment]
    task_tool = None  # type: ignore[assignment]


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # Core types and enums
    "TaskStatus",
    "SubagentType",
    # Models
    "TodoItem",
    "TodoWriteInput",
    "TaskInput",
    # State management
    "TaskListState",
    # Core functions
    "todo_write",
    "task_launcher",
    # Helper functions
    "generate_task_id",
    "validate_unique_ids",
    "validate_single_in_progress",
    "get_task_summary",
    "mark_task_complete",
    "mark_task_in_progress",
    "add_task",
    # Exceptions
    "TaskPlanningError",
    "TodoValidationError",
    "MultipleInProgressError",
    "DuplicateTaskIDError",
    "SubagentError",
    "InvalidSubagentTypeError",
    # PydanticAI tools (if available)
    "todo_write_tool",
    "task_tool",
]
