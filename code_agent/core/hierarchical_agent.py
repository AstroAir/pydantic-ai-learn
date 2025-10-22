"""
Hierarchical Agent System

Implements a hierarchical agent architecture where parent agents can delegate
tasks to specialized sub-agents. Integrates with the existing CodeAgent while
adding hierarchical capabilities.

Features:
- Parent agent with sub-agent delegation
- Automatic task routing based on capabilities
- Result aggregation from multiple sub-agents
- Error handling and recovery
- Integration with A2A protocol

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .a2a_integration import A2AClient, A2AConfig, A2AServer
from .agent_registry import AgentRegistry
from .sub_agent import DelegatedTask, SubAgent, SubAgentResult
from .task_delegator import TaskDelegator


@dataclass
class HierarchicalAgentConfig:
    """Configuration for hierarchical agent."""

    enable_sub_agents: bool = True
    """Enable sub-agent functionality"""

    enable_a2a: bool = False
    """Enable A2A protocol support"""

    a2a_config: A2AConfig | None = None
    """A2A configuration"""

    max_delegation_depth: int = 3
    """Maximum delegation depth (prevent infinite recursion)"""

    auto_register_sub_agents: bool = True
    """Automatically register sub-agents on creation"""


class CodeSubAgent(SubAgent):
    """
    A sub-agent implementation that wraps a CodeAgent.

    Allows CodeAgent instances to participate in hierarchical delegation.
    """

    def __init__(
        self,
        name: str,
        description: str,
        agent: Any,  # CodeAgent instance
        capabilities: list[str] | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize code sub-agent.

        Args:
            name: Agent name
            description: Agent description
            agent: Underlying CodeAgent instance
            capabilities: Agent capabilities
            agent_id: Optional agent ID
            metadata: Optional metadata
        """
        super().__init__(name, description, capabilities, agent_id, metadata)
        self.agent = agent

    async def execute_task(self, task: DelegatedTask) -> SubAgentResult:
        """
        Execute a task using the underlying CodeAgent.

        Args:
            task: Task to execute

        Returns:
            SubAgentResult with execution result
        """
        try:
            # Execute using the CodeAgent
            result = await self.agent.run(task.prompt)

            return SubAgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                output=result.output if hasattr(result, "output") else str(result),
                success=True,
                metadata={
                    "agent_name": self.name,
                    "capabilities": self.capabilities,
                },
            )
        except Exception as e:
            return SubAgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                output=None,
                success=False,
                error=str(e),
                metadata={
                    "agent_name": self.name,
                    "error_type": type(e).__name__,
                },
            )


class HierarchicalAgent:
    """
    Hierarchical agent that can delegate tasks to sub-agents.

    Extends the CodeAgent with hierarchical delegation capabilities,
    allowing complex tasks to be broken down and distributed to
    specialized sub-agents.
    """

    def __init__(
        self,
        parent_agent: Any,  # CodeAgent instance
        config: HierarchicalAgentConfig | None = None,
    ):
        """
        Initialize hierarchical agent.

        Args:
            parent_agent: Parent CodeAgent instance
            config: Hierarchical configuration
        """
        self.parent_agent = parent_agent
        self.config = config or HierarchicalAgentConfig()

        # Initialize sub-agent infrastructure
        self.registry = AgentRegistry()
        self.delegator = TaskDelegator(self.registry)

        # A2A integration
        self.a2a_server: A2AServer | None = None
        self.a2a_clients: dict[str, A2AClient] = {}

        if self.config.enable_a2a:
            self._setup_a2a()

    def _setup_a2a(self) -> None:
        """Setup A2A server for this agent."""
        try:
            from .a2a_integration import A2A_AVAILABLE

            if A2A_AVAILABLE:
                self.a2a_server = A2AServer(
                    self.parent_agent.agent,  # PydanticAI agent
                    self.config.a2a_config,
                )
        except Exception as e:
            print(f"Warning: A2A setup failed: {e}")

    def register_sub_agent(
        self,
        name: str,
        description: str,
        agent: Any,
        capabilities: list[str] | None = None,
    ) -> CodeSubAgent:
        """
        Register a new sub-agent.

        Args:
            name: Sub-agent name
            description: Sub-agent description
            agent: CodeAgent instance
            capabilities: Agent capabilities

        Returns:
            Created CodeSubAgent
        """
        sub_agent = CodeSubAgent(
            name=name,
            description=description,
            agent=agent,
            capabilities=capabilities or [],
        )

        self.registry.register(sub_agent)
        return sub_agent

    def deregister_sub_agent(self, agent_id: str) -> bool:
        """
        Deregister a sub-agent.

        Args:
            agent_id: ID of agent to deregister

        Returns:
            True if deregistered, False if not found
        """
        return self.registry.deregister(agent_id)

    async def delegate(
        self,
        prompt: str,
        required_capabilities: list[str] | None = None,
        agent_id: str | None = None,
    ) -> SubAgentResult:
        """
        Delegate a task to a sub-agent.

        Args:
            prompt: Task prompt
            required_capabilities: Required capabilities
            agent_id: Specific agent ID (optional)

        Returns:
            SubAgentResult from the sub-agent
        """
        return await self.delegator.delegate_and_wait(
            prompt=prompt,
            agent_id=agent_id,
            required_capabilities=required_capabilities,
        )

    async def delegate_multiple(
        self,
        prompts: list[str],
        required_capabilities: list[str] | None = None,
    ) -> list[SubAgentResult]:
        """
        Delegate multiple tasks in parallel.

        Args:
            prompts: List of task prompts
            required_capabilities: Required capabilities

        Returns:
            List of SubAgentResults
        """
        return await self.delegator.delegate_multiple(
            prompts=prompts,
            required_capabilities=required_capabilities,
        )

    async def run_with_delegation(
        self,
        prompt: str,
        use_sub_agents: bool = True,
    ) -> Any:
        """
        Run a task with optional sub-agent delegation.

        This method analyzes the prompt and decides whether to:
        1. Handle it directly with the parent agent
        2. Delegate to a specialized sub-agent
        3. Break it down and delegate to multiple sub-agents

        Args:
            prompt: Task prompt
            use_sub_agents: Whether to use sub-agents

        Returns:
            Task result
        """
        if not use_sub_agents or not self.config.enable_sub_agents:
            # Run directly with parent agent
            return await self.parent_agent.run(prompt)

        # Check if we have suitable sub-agents
        available_agents = self.registry.list_agents()

        if not available_agents:
            # No sub-agents, use parent
            return await self.parent_agent.run(prompt)

        # For now, use simple heuristic: delegate to first available
        # In production, this would use LLM to analyze and route
        result = await self.delegate(prompt)

        if result.success:
            return result.output
        # Fallback to parent agent on failure
        return await self.parent_agent.run(prompt)

    def get_sub_agents(self) -> list[SubAgent]:
        """Get all registered sub-agents."""
        return self.registry.list_agents()

    def get_stats(self) -> dict[str, Any]:
        """Get hierarchical agent statistics."""
        return {
            "registry": self.registry.get_stats(),
            "delegator": self.delegator.get_stats(),
            "a2a_enabled": self.config.enable_a2a,
            "a2a_server_active": self.a2a_server is not None,
        }

    def get_a2a_app(self) -> Any:
        """Get the A2A ASGI application (if enabled)."""
        if self.a2a_server is None:
            return None
        return self.a2a_server.get_app()

    def to_a2a(
        self,
        *,
        storage: Any | None = None,
        broker: Any | None = None,
        name: str | None = None,
        url: str | None = None,
        version: str | None = None,
        description: str | None = None,
        provider: str | None = None,
        skills: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Convert this hierarchical agent to an A2A ASGI application.

        This exposes the hierarchical agent (and its delegation capabilities)
        as an A2A protocol server.

        Args:
            storage: Optional custom storage implementation
            broker: Optional custom broker implementation
            name: Agent name for the agent card
            url: Agent URL for the agent card
            version: Agent version for the agent card
            description: Agent description for the agent card
            provider: Agent provider for the agent card
            skills: Agent skills (aggregated from sub-agents if not provided)
            **kwargs: Additional arguments passed to FastA2A

        Returns:
            ASGI application (FastA2A instance)

        Raises:
            ImportError: If fasta2a is not installed

        Example:
            ```python
            parent = CodeAgent(model="openai:gpt-4")
            h_agent = HierarchicalAgent(parent)
            h_agent.register_sub_agent(
                "Analysis", "Code analyzer", analysis_agent, ["analysis"]
            )
            app = h_agent.to_a2a(
                name="Hierarchical Code Agent",
                description="Multi-agent system with specialized sub-agents"
            )
            ```
        """
        # Aggregate skills from sub-agents if not provided
        if skills is None:
            skills = []
            for agent in self.registry.list_agents():
                skills.extend(agent.capabilities)
            skills = list(set(skills))  # Remove duplicates

        # Use parent agent's to_a2a if available
        if hasattr(self.parent_agent, "to_a2a"):
            return self.parent_agent.to_a2a(
                storage=storage,
                broker=broker,
                name=name or "HierarchicalAgent",
                url=url,
                version=version or "1.0.0",
                description=description or "Hierarchical multi-agent system",
                provider=provider,
                skills=skills,
                **kwargs,
            )

        # Fallback to using parent's underlying agent
        if hasattr(self.parent_agent, "agent"):
            try:
                return self.parent_agent.agent.to_a2a(
                    storage=storage,
                    broker=broker,
                    name=name or "HierarchicalAgent",
                    url=url,
                    version=version or "1.0.0",
                    description=description or "Hierarchical multi-agent system",
                    provider=provider,
                    skills=skills,
                    **kwargs,
                )
            except AttributeError:
                pass

        # Final fallback to A2AServer
        from .a2a_integration import A2AConfig, A2AServer

        a2a_config = A2AConfig()
        if hasattr(self.parent_agent, "agent"):
            a2a_server = A2AServer(self.parent_agent.agent, a2a_config)
        else:
            a2a_server = A2AServer(self.parent_agent, a2a_config)
        return a2a_server.get_app()

    async def broadcast(
        self,
        prompt: str,
        agent_ids: list[str] | None = None,
    ) -> list[SubAgentResult]:
        """
        Broadcast a task to multiple sub-agents in parallel.

        Args:
            prompt: Task prompt to broadcast
            agent_ids: Specific agent IDs to broadcast to (all if None)

        Returns:
            List of SubAgentResults from all agents

        Example:
            ```python
            # Broadcast to all sub-agents
            results = await h_agent.broadcast("Analyze this code pattern")

            # Broadcast to specific agents
            results = await h_agent.broadcast(
                "Check security",
                agent_ids=["security-agent", "audit-agent"]
            )
            ```
        """
        if agent_ids:
            agents = [agent for aid in agent_ids if (agent := self.registry.get_agent(aid)) is not None]
        else:
            agents = self.registry.list_agents()

        if not agents:
            return []

        # Create tasks for all agents
        tasks = []
        for agent in agents:
            task = asyncio.create_task(
                self.delegator.delegate_and_wait(
                    prompt=prompt,
                    agent_id=agent.agent_id,
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return results
        return [r for r in results if isinstance(r, SubAgentResult)]

    async def call_agent_via_a2a(
        self,
        agent_url: str,
        prompt: str,
        context_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Call a remote agent via A2A protocol.

        This enables hierarchical agents to communicate with external
        A2A-compliant agents over HTTP.

        Args:
            agent_url: URL of the remote A2A agent
            prompt: Task prompt
            context_id: Optional context ID for conversation continuity
            metadata: Optional metadata

        Returns:
            Response from the remote agent

        Raises:
            httpx.HTTPError: If request fails

        Example:
            ```python
            # Call a remote specialized agent
            result = await h_agent.call_agent_via_a2a(
                "http://localhost:8001",
                "Refactor this code",
                context_id="conv-123"
            )
            ```
        """
        # Get or create A2A client for this URL
        if agent_url not in self.a2a_clients:
            from .a2a_integration import A2AClient, A2AConfig

            config = self.config.a2a_config or A2AConfig()
            self.a2a_clients[agent_url] = A2AClient(agent_url, config)

        client = self.a2a_clients[agent_url]

        # Build request metadata
        request_metadata = metadata or {}
        if context_id:
            request_metadata["context_id"] = context_id

        # Make the call (client manages its own lifecycle)
        async with client:
            return await client.call_agent(prompt, request_metadata)

    async def chain_delegation(
        self,
        prompts: list[tuple[str, list[str] | None]],
    ) -> list[SubAgentResult]:
        """
        Delegate tasks in a chain where each task depends on the previous.

        This is useful for multi-stage workflows where the output of one
        agent feeds into the next.

        Args:
            prompts: List of (prompt, required_capabilities) tuples

        Returns:
            List of SubAgentResults in order

        Example:
            ```python
            results = await h_agent.chain_delegation([
                ("Analyze code structure", ["analysis"]),
                ("Suggest refactoring based on analysis", ["refactoring"]),
                ("Generate tests for refactored code", ["testing"]),
            ])
            ```
        """
        results = []
        context = ""

        for prompt, capabilities in prompts:
            # Append previous result as context
            full_prompt = f"{prompt}\n\nPrevious result: {context}" if context else prompt

            # Delegate to appropriate agent
            result = await self.delegator.delegate_and_wait(
                prompt=full_prompt,
                required_capabilities=capabilities,
            )
            results.append(result)

            # Update context for next task
            if result.success:
                context = str(result.output)
            else:
                # Stop chain on failure
                break

        return results
