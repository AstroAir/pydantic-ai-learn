"""
Task Delegation System for Hierarchical Agents

Manages task delegation from parent agents to sub-agents, including task
routing, execution tracking, and result aggregation.

Features:
- Intelligent task routing based on capabilities
- Async task execution with result tracking
- Load balancing across sub-agents
- Error handling and retry logic
- Result aggregation from multiple sub-agents

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from .agent_registry import AgentRegistry
from .sub_agent import (
    DelegatedTask,
    SubAgent,
    SubAgentResult,
    SubAgentStatus,
    TaskStatus,
)


class TaskDelegator:
    """
    Manages task delegation to sub-agents.

    Routes tasks to appropriate sub-agents based on capabilities,
    tracks execution, and aggregates results.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ):
        """
        Initialize the task delegator.

        Args:
            registry: Agent registry for discovering sub-agents
            max_retries: Maximum retry attempts for failed tasks
            timeout_seconds: Task execution timeout
        """
        self.registry = registry
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self._tasks: dict[str, DelegatedTask] = {}
        self._results: dict[str, SubAgentResult] = {}
        self._task_locks: dict[str, asyncio.Lock] = {}

    def create_task(
        self,
        prompt: str,
        agent_id: str | None = None,
        required_capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DelegatedTask:
        """
        Create a new delegated task.

        Args:
            prompt: Task instruction/prompt
            agent_id: Specific agent ID (if known)
            required_capabilities: Required capabilities for task
            metadata: Additional task metadata

        Returns:
            Created DelegatedTask

        Raises:
            ValueError: If no suitable agent found
        """
        # Find suitable agent
        if agent_id:
            agent = self.registry.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
        elif required_capabilities:
            agents = self.registry.find_by_capabilities(required_capabilities)
            if not agents:
                raise ValueError(f"No agent found with capabilities: {required_capabilities}")
            # Select least busy agent
            agent = self._select_agent(agents)
        else:
            # Get any available agent
            agents = self.registry.list_agents(status=SubAgentStatus.IDLE)
            if not agents:
                agents = self.registry.list_agents()
            if not agents:
                raise ValueError("No agents available")
            agent = self._select_agent(agents)

        task = DelegatedTask(
            task_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            prompt=prompt,
            status=TaskStatus.PENDING,
            metadata=metadata or {},
        )

        self._tasks[task.task_id] = task
        return task

    def _select_agent(self, agents: list[SubAgent]) -> SubAgent:
        """Select the best agent from a list (simple load balancing)."""
        # Prefer idle agents
        idle_agents = [a for a in agents if a.status == SubAgentStatus.IDLE]
        if idle_agents:
            return idle_agents[0]

        # Otherwise return first available
        return agents[0]

    async def delegate_task(
        self,
        task: DelegatedTask,
        retry_count: int = 0,
    ) -> SubAgentResult:
        """
        Delegate a task to a sub-agent for execution.

        Args:
            task: The task to delegate
            retry_count: Current retry attempt

        Returns:
            SubAgentResult with execution result

        Raises:
            TimeoutError: If task execution times out
            RuntimeError: If task fails after max retries
        """
        agent = self.registry.get_agent(task.agent_id)
        if not agent:
            raise ValueError(f"Agent {task.agent_id} not found")

        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(UTC)
        agent.set_status(SubAgentStatus.BUSY)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.execute_task(task),
                timeout=self.timeout_seconds,
            )

            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)
            task.result = result.output

            # Store result
            self._results[task.task_id] = result

            # Update agent status
            agent.set_status(SubAgentStatus.IDLE)

            return result

        except TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Task timed out after {self.timeout_seconds} seconds"
            agent.set_status(SubAgentStatus.ERROR)
            raise TimeoutError(task.error) from None

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            agent.set_status(SubAgentStatus.ERROR)

            # Retry if attempts remaining
            if retry_count < self.max_retries:
                await asyncio.sleep(2**retry_count)  # Exponential backoff
                return await self.delegate_task(task, retry_count + 1)

            raise RuntimeError(f"Task failed after {retry_count} retries: {e}") from e

    async def delegate_and_wait(
        self,
        prompt: str,
        agent_id: str | None = None,
        required_capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        """
        Create and delegate a task, waiting for completion.

        Args:
            prompt: Task instruction
            agent_id: Specific agent ID
            required_capabilities: Required capabilities
            metadata: Task metadata

        Returns:
            SubAgentResult with execution result
        """
        task = self.create_task(prompt, agent_id, required_capabilities, metadata)
        return await self.delegate_task(task)

    async def delegate_multiple(
        self,
        prompts: list[str],
        required_capabilities: list[str] | None = None,
    ) -> list[SubAgentResult]:
        """
        Delegate multiple tasks in parallel.

        Args:
            prompts: List of task prompts
            required_capabilities: Required capabilities for all tasks

        Returns:
            List of SubAgentResults
        """
        tasks = [self.create_task(prompt, required_capabilities=required_capabilities) for prompt in prompts]

        results = await asyncio.gather(
            *[self.delegate_task(task) for task in tasks],
            return_exceptions=True,
        )

        # Convert exceptions to failed results
        final_results: list[SubAgentResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    SubAgentResult(
                        task_id=tasks[i].task_id,
                        agent_id=tasks[i].agent_id,
                        output=None,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)  # type: ignore[arg-type]

        return final_results

    def get_task(self, task_id: str) -> DelegatedTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> SubAgentResult | None:
        """Get a result by task ID."""
        return self._results.get(task_id)

    def list_tasks(self, status: TaskStatus | None = None) -> list[DelegatedTask]:
        """List all tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks

    def get_stats(self) -> dict[str, Any]:
        """Get delegation statistics."""
        return {
            "total_tasks": len(self._tasks),
            "total_results": len(self._results),
            "tasks_by_status": {
                status.value: len([t for t in self._tasks.values() if t.status == status]) for status in TaskStatus
            },
        }

    def _score_agent_for_task(
        self,
        agent: SubAgent,
        required_capabilities: list[str] | None,
        task_metadata: dict[str, Any] | None,
    ) -> float:
        """
        Score an agent for a task based on multiple factors.

        Args:
            agent: Agent to score
            required_capabilities: Required capabilities
            task_metadata: Task metadata

        Returns:
            Score (higher is better, 0 means unsuitable)
        """
        score = 0.0

        # Status score
        if agent.status == SubAgentStatus.IDLE:
            score += 100.0
        elif agent.status == SubAgentStatus.BUSY:
            score += 50.0
        elif agent.status in (SubAgentStatus.ERROR, SubAgentStatus.OFFLINE):
            return 0.0  # Unsuitable

        # Capability match score
        if required_capabilities:
            matching_caps = sum(1 for cap in required_capabilities if cap in agent.capabilities)
            if matching_caps == len(required_capabilities):
                score += 50.0 * (matching_caps / len(agent.capabilities))
            else:
                return 0.0  # Missing required capabilities

        # Specialization bonus (fewer capabilities = more specialized)
        if agent.capabilities:
            specialization_score = 10.0 / len(agent.capabilities)
            score += specialization_score

        # Recent heartbeat bonus
        age_seconds = (datetime.now(UTC) - agent.last_heartbeat).total_seconds()
        if age_seconds < 10:
            score += 10.0
        elif age_seconds < 60:
            score += 5.0

        return score

    def select_best_agent(
        self,
        required_capabilities: list[str] | None,
        task_metadata: dict[str, Any] | None = None,
    ) -> SubAgent | None:
        """
        Select the best agent for a task using intelligent scoring.

        Args:
            required_capabilities: Required capabilities
            task_metadata: Optional task metadata for scoring

        Returns:
            Best matching agent or None
        """
        if required_capabilities:
            candidates = self.registry.find_by_capabilities(required_capabilities)
        else:
            candidates = self.registry.list_agents()

        if not candidates:
            return None

        # Score all candidates
        scored_agents = [
            (agent, self._score_agent_for_task(agent, required_capabilities, task_metadata)) for agent in candidates
        ]

        # Filter out unsuitable agents (score = 0)
        suitable_agents = [(a, s) for a, s in scored_agents if s > 0]

        if not suitable_agents:
            return None

        # Return agent with highest score
        return max(suitable_agents, key=lambda x: x[1])[0]

    async def delegate_with_fallback(
        self,
        prompt: str,
        primary_agent_id: str,
        fallback_agent_id: str | None = None,
        required_capabilities: list[str] | None = None,
    ) -> SubAgentResult:
        """
        Delegate a task with fallback agent support.

        If the primary agent fails, automatically retry with fallback agent.

        Args:
            prompt: Task prompt
            primary_agent_id: Primary agent to try first
            fallback_agent_id: Fallback agent if primary fails
            required_capabilities: Required capabilities

        Returns:
            SubAgentResult

        Example:
            ```python
            result = await delegator.delegate_with_fallback(
                "Analyze code",
                primary_agent_id="fast-analyzer",
                fallback_agent_id="thorough-analyzer"
            )
            ```
        """
        # Try primary agent
        try:
            task = self.create_task(prompt, agent_id=primary_agent_id)
            result = await self.delegate_task(task)
            if result.success:
                return result
        except Exception:
            # Primary agent failed, try fallback
            pass

        # Try fallback agent
        if fallback_agent_id:
            try:
                task = self.create_task(prompt, agent_id=fallback_agent_id)
                return await self.delegate_task(task)
            except Exception:
                # Fallback also failed
                pass

        # Try any suitable agent
        if required_capabilities:
            task = self.create_task(prompt, required_capabilities=required_capabilities)
            return await self.delegate_task(task)

        # All options exhausted
        return SubAgentResult(
            task_id="failed",
            agent_id="none",
            output=None,
            success=False,
            error="No suitable agent available",
        )

    async def delegate_with_timeout(
        self,
        prompt: str,
        timeout_seconds: float,
        required_capabilities: list[str] | None = None,
        agent_id: str | None = None,
    ) -> SubAgentResult:
        """
        Delegate a task with a custom timeout.

        Args:
            prompt: Task prompt
            timeout_seconds: Timeout in seconds
            required_capabilities: Required capabilities
            agent_id: Optional specific agent

        Returns:
            SubAgentResult

        Raises:
            asyncio.TimeoutError: If task times out
        """
        task = self.create_task(prompt, agent_id, required_capabilities)

        try:
            return await asyncio.wait_for(self.delegate_task(task), timeout=timeout_seconds)
        except TimeoutError:
            # Update task status
            task.status = TaskStatus.FAILED
            task.error = f"Task timed out after {timeout_seconds} seconds"

            # Mark agent as busy/error
            agent = self.registry.get_agent(task.agent_id)
            if agent:
                agent.set_status(SubAgentStatus.ERROR)

            return SubAgentResult(
                task_id=task.task_id,
                agent_id=task.agent_id,
                output=None,
                success=False,
                error=task.error,
            )
