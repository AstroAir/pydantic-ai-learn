"""
Sub-Agent System for Hierarchical Agent Architecture

Implements a hierarchical agent system where parent agents can delegate tasks
to specialized sub-agents. Each sub-agent maintains its own state and can
report results back to the parent agent.

Features:
- Hierarchical agent delegation
- Sub-agent registration and discovery
- Task delegation with result reporting
- State management per sub-agent
- Error handling and recovery

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class SubAgentStatus(str, Enum):
    """Status of a sub-agent."""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class TaskStatus(str, Enum):
    """Status of a delegated task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubAgentInfo:
    """Information about a sub-agent."""

    agent_id: str
    """Unique identifier for the sub-agent"""

    name: str
    """Human-readable name"""

    description: str
    """Description of sub-agent's responsibilities"""

    capabilities: list[str] = field(default_factory=list)
    """List of capabilities/specializations"""

    status: SubAgentStatus = SubAgentStatus.IDLE
    """Current status"""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Creation timestamp"""

    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Last heartbeat timestamp"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""


@dataclass
class DelegatedTask:
    """A task delegated to a sub-agent."""

    task_id: str
    """Unique task identifier"""

    agent_id: str
    """ID of the sub-agent handling this task"""

    prompt: str
    """Task prompt/instruction"""

    status: TaskStatus = TaskStatus.PENDING
    """Current task status"""

    result: Any = None
    """Task result (when completed)"""

    error: str | None = None
    """Error message (if failed)"""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Creation timestamp"""

    started_at: datetime | None = None
    """Start timestamp"""

    completed_at: datetime | None = None
    """Completion timestamp"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def duration_seconds(self) -> float | None:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class SubAgentResult:
    """Result returned by a sub-agent."""

    task_id: str
    """ID of the completed task"""

    agent_id: str
    """ID of the sub-agent"""

    output: Any
    """The result output"""

    success: bool = True
    """Whether the task succeeded"""

    error: str | None = None
    """Error message if failed"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Result timestamp"""


class SubAgent:
    """
    Base class for a sub-agent in a hierarchical agent system.

    A sub-agent is a specialized agent that can be delegated tasks
    by a parent agent. It maintains its own state and reports results
    back to the parent.
    """

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: list[str] | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize a sub-agent.

        Args:
            name: Human-readable name
            description: Description of responsibilities
            capabilities: List of capabilities
            agent_id: Optional custom agent ID (auto-generated if not provided)
            metadata: Optional metadata dictionary
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.metadata = metadata or {}
        self.status = SubAgentStatus.IDLE
        self.created_at = datetime.now(UTC)
        self.last_heartbeat = datetime.now(UTC)
        self._tasks: dict[str, DelegatedTask] = {}
        self._results: dict[str, SubAgentResult] = {}

    def get_info(self) -> SubAgentInfo:
        """Get information about this sub-agent."""
        return SubAgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
            status=self.status,
            created_at=self.created_at,
            last_heartbeat=self.last_heartbeat,
            metadata=self.metadata,
        )

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.now(UTC)

    def set_status(self, status: SubAgentStatus) -> None:
        """Set the agent status."""
        self.status = status
        self.heartbeat()

    async def execute_task(self, task: DelegatedTask) -> SubAgentResult:
        """
        Execute a delegated task.

        This method should be overridden by subclasses to implement
        actual task execution logic.

        Args:
            task: The task to execute

        Returns:
            SubAgentResult with the execution result
        """
        raise NotImplementedError("Subclasses must implement execute_task")

    def get_task(self, task_id: str) -> DelegatedTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> SubAgentResult | None:
        """Get a result by task ID."""
        return self._results.get(task_id)

    def list_tasks(self, status: TaskStatus | None = None) -> list[DelegatedTask]:
        """List tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks
