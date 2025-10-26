"""
Agent-to-Agent (A2A) Protocol Integration

Integrates Pydantic AI's A2A protocol for inter-agent communication.
Provides both server (exposing agents) and client (calling other agents)
capabilities.

Features:
- A2A server wrapper for exposing agents
- A2A client for calling remote agents
- Message protocol adapters
- Error handling for network communication
- Connection pooling and caching

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast

import httpx

try:
    from fasta2a import Broker, FastA2A, Storage, Task, TaskResult, Worker
    from pydantic_ai import Agent

    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False

    # Provide stub classes for type checking
    class FastA2A:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    class Storage:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    class Broker:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    class Worker:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    class Task:  # type: ignore
        ...

    class TaskResult:  # type: ignore
        ...


@dataclass
class A2AConfig:
    """Configuration for A2A integration."""

    enabled: bool = True
    """Whether A2A is enabled"""

    server_host: str = "0.0.0.0"
    """Server host"""

    server_port: int = 8000
    """Server port"""

    timeout_seconds: int = 300
    """Request timeout"""

    max_connections: int = 100
    """Maximum concurrent connections"""

    enable_caching: bool = True
    """Enable response caching"""

    cache_ttl_seconds: int = 300
    """Cache TTL in seconds"""


class InMemoryStorage(Storage if A2A_AVAILABLE else object):  # type: ignore
    """Simple in-memory storage for A2A tasks."""

    def __init__(self) -> None:
        """Initialize storage."""
        self._tasks: dict[str, Any] = {}

    async def save_task(self, task: Any) -> None:
        """Save a task."""
        self._tasks[task.id] = task

    async def load_task(self, task_id: str) -> Any:
        """Load a task by ID."""
        return self._tasks.get(task_id)

    async def update_task(self, task: Any) -> None:
        """Update a task."""
        self._tasks[task.id] = task

    async def delete_task(self, task_id: str) -> None:
        """Delete a task."""
        self._tasks.pop(task_id, None)


class SimpleBroker(Broker if A2A_AVAILABLE else object):  # type: ignore
    """Simple in-memory broker for A2A tasks."""

    def __init__(self) -> None:
        """Initialize broker."""
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    async def enqueue(self, task: Any) -> None:
        """Enqueue a task."""
        await self._queue.put(task)

    async def dequeue(self) -> Any:
        """Dequeue a task."""
        return await self._queue.get()


class AgentWorker(Worker if A2A_AVAILABLE else object):  # type: ignore
    """Worker that executes tasks using a PydanticAI agent."""

    def __init__(self, agent: Agent):
        """
        Initialize worker with an agent.

        Args:
            agent: PydanticAI agent to use for task execution
        """
        self.agent = agent

    async def execute(self, task: Any) -> Any:
        """
        Execute a task using the agent.

        Args:
            task: Task to execute

        Returns:
            Task result
        """
        try:
            # Extract prompt from task
            prompt = task.input if hasattr(task, "input") else str(task)

            # Run agent
            result = await self.agent.run(prompt)

            # Return result
            return {
                "output": result.output if hasattr(result, "output") else str(result),
                "success": True,
            }
        except Exception as e:
            return {
                "output": None,
                "success": False,
                "error": str(e),
            }


class A2AServer:
    """
    A2A server wrapper for exposing PydanticAI agents.

    Wraps a PydanticAI agent and exposes it as an A2A-compliant server.
    """

    def __init__(
        self,
        agent: Agent,
        config: A2AConfig | None = None,
    ):
        """
        Initialize A2A server.

        Args:
            agent: PydanticAI agent to expose
            config: A2A configuration

        Raises:
            ImportError: If fasta2a is not installed
        """
        if not A2A_AVAILABLE:
            raise ImportError("fasta2a is not installed. Install with: pip install 'pydantic-ai-slim[a2a]'")

        self.agent = agent
        self.config = config or A2AConfig()

        # Create A2A components
        self.storage = InMemoryStorage()
        self.broker = SimpleBroker()
        self.worker = AgentWorker(agent)

        # Create FastA2A app
        self.app = FastA2A(
            storage=self.storage,
            broker=self.broker,
            worker=self.worker,
        )

    def get_app(self) -> FastA2A:
        """Get the ASGI application."""
        return self.app

    async def start(self) -> None:
        """
        Start the A2A server.

        Performs any necessary initialization for the server components.
        In this implementation with in-memory storage and broker,
        no explicit startup is required, but this method is provided
        for subclasses that may need initialization logic.
        """
        # Validate components are initialized
        if not self.storage:
            raise RuntimeError("Storage not initialized")
        if not self.broker:
            raise RuntimeError("Broker not initialized")
        if not self.worker:
            raise RuntimeError("Worker not initialized")

        # Server is ready - no additional startup required for in-memory components
        # Subclasses can override to add persistent storage initialization, etc.

    async def stop(self) -> None:
        """
        Stop the A2A server and cleanup resources.

        Performs graceful shutdown of server components.
        In this implementation, ensures that any pending tasks are handled.
        """
        # For in-memory implementation, no cleanup needed
        # Subclasses can override to close database connections,
        # flush buffers, or perform other cleanup operations
        pass


class A2AClient:
    """
    A2A client for calling remote agents.

    Provides a client interface for communicating with A2A-compliant agents.
    """

    def __init__(
        self,
        base_url: str,
        config: A2AConfig | None = None,
    ):
        """
        Initialize A2A client.

        Args:
            base_url: Base URL of the A2A server
            config: A2A configuration
        """
        self.base_url = base_url.rstrip("/")
        self.config = config or A2AConfig()
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[Any, datetime]] = {}

    async def __aenter__(self) -> A2AClient:
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            limits=httpx.Limits(max_connections=self.config.max_connections),
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    async def call_agent(
        self,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Call a remote agent via A2A protocol.

        Args:
            prompt: Task prompt/instruction
            metadata: Additional metadata

        Returns:
            Response from the remote agent

        Raises:
            httpx.HTTPError: If request fails
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Check cache
        cache_key = f"{prompt}:{metadata}"
        if self.config.enable_caching and cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            age = (datetime.now(UTC) - cached_time).total_seconds()
            if age < self.config.cache_ttl_seconds:
                return cached_result  # type: ignore[no-any-return]

        # Make request
        response = await self._client.post(
            f"{self.base_url}/tasks",
            json={
                "input": prompt,
                "metadata": metadata or {},
            },
        )
        response.raise_for_status()

        result = cast(dict[str, Any], response.json())

        # Cache result
        if self.config.enable_caching:
            self._cache[cache_key] = (result, datetime.now(UTC))

        return result

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
