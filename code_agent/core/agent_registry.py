"""
Agent Registry and Discovery System

Manages registration, discovery, and lifecycle of sub-agents in a hierarchical
agent system. Provides service discovery capabilities for agent-to-agent
communication.

Features:
- Agent registration and deregistration
- Agent discovery by ID, name, or capabilities
- Health monitoring and heartbeat tracking
- Agent metadata management
- Thread-safe operations

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any

from .sub_agent import SubAgent, SubAgentInfo, SubAgentStatus


class AgentRegistry:
    """
    Registry for managing sub-agents in a hierarchical system.

    Provides service discovery, health monitoring, and lifecycle management
    for sub-agents.
    """

    def __init__(self, heartbeat_timeout: int = 30):
        """
        Initialize the agent registry.

        Args:
            heartbeat_timeout: Seconds before marking agent as offline
        """
        self._agents: dict[str, SubAgent] = {}
        self._agents_by_name: dict[str, str] = {}  # name -> agent_id
        self._agents_by_capability: dict[str, list[str]] = {}  # capability -> [agent_ids]
        self._lock = RLock()
        self.heartbeat_timeout = heartbeat_timeout

    def register(self, agent: SubAgent) -> None:
        """
        Register a sub-agent.

        Args:
            agent: The sub-agent to register

        Raises:
            ValueError: If agent ID already registered
        """
        with self._lock:
            if agent.agent_id in self._agents:
                raise ValueError(f"Agent {agent.agent_id} already registered")

            self._agents[agent.agent_id] = agent
            self._agents_by_name[agent.name] = agent.agent_id

            for capability in agent.capabilities:
                if capability not in self._agents_by_capability:
                    self._agents_by_capability[capability] = []
                self._agents_by_capability[capability].append(agent.agent_id)

    def deregister(self, agent_id: str) -> bool:
        """
        Deregister a sub-agent.

        Args:
            agent_id: ID of the agent to deregister

        Returns:
            True if agent was deregistered, False if not found
        """
        with self._lock:
            if agent_id not in self._agents:
                return False

            agent = self._agents.pop(agent_id)

            # Remove from name index
            if agent.name in self._agents_by_name:
                del self._agents_by_name[agent.name]

            # Remove from capability index
            for capability in agent.capabilities:
                if capability in self._agents_by_capability:
                    self._agents_by_capability[capability] = [
                        aid for aid in self._agents_by_capability[capability] if aid != agent_id
                    ]
                    if not self._agents_by_capability[capability]:
                        del self._agents_by_capability[capability]

            return True

    def get_agent(self, agent_id: str) -> SubAgent | None:
        """Get an agent by ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def get_agent_by_name(self, name: str) -> SubAgent | None:
        """Get an agent by name."""
        with self._lock:
            agent_id = self._agents_by_name.get(name)
            if agent_id:
                return self._agents.get(agent_id)
            return None

    def find_by_capability(self, capability: str) -> list[SubAgent]:
        """Find all agents with a specific capability."""
        with self._lock:
            agent_ids = self._agents_by_capability.get(capability, [])
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_by_capabilities(self, capabilities: list[str]) -> list[SubAgent]:
        """Find agents that have all specified capabilities."""
        with self._lock:
            if not capabilities:
                return list(self._agents.values())

            # Start with agents having first capability
            matching_ids = set(self._agents_by_capability.get(capabilities[0], []))

            # Intersect with agents having other capabilities
            for capability in capabilities[1:]:
                matching_ids &= set(self._agents_by_capability.get(capability, []))

            return [self._agents[aid] for aid in matching_ids if aid in self._agents]

    def list_agents(self, status: SubAgentStatus | None = None) -> list[SubAgent]:
        """List all agents, optionally filtered by status."""
        with self._lock:
            agents = list(self._agents.values())
            if status:
                agents = [a for a in agents if a.status == status]
            return agents

    def list_agent_infos(self, status: SubAgentStatus | None = None) -> list[SubAgentInfo]:
        """List all agent information."""
        return [agent.get_info() for agent in self.list_agents(status)]

    def check_health(self) -> dict[str, Any]:
        """
        Check health of all registered agents.

        Returns:
            Dictionary with health status information
        """

        def _to_aware_utc(dt: datetime) -> datetime:
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

        with self._lock:
            now = datetime.now(UTC)
            timeout_delta = timedelta(seconds=self.heartbeat_timeout)

            healthy: list[str] = []
            offline: list[str] = []

            for agent in self._agents.values():
                last_hb = _to_aware_utc(agent.last_heartbeat)
                if now - last_hb > timeout_delta:
                    agent.set_status(SubAgentStatus.OFFLINE)
                    offline.append(agent.agent_id)
                else:
                    healthy.append(agent.agent_id)

            return {
                "total": len(self._agents),
                "healthy": len(healthy),
                "offline": len(offline),
                "healthy_agents": healthy,
                "offline_agents": offline,
            }

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            return {
                "total_agents": len(self._agents),
                "total_capabilities": len(self._agents_by_capability),
                "agents_by_status": {
                    status.value: len([a for a in self._agents.values() if a.status == status])
                    for status in SubAgentStatus
                },
                "capabilities": {cap: len(agent_ids) for cap, agent_ids in self._agents_by_capability.items()},
            }

    def clear(self) -> None:
        """Clear all registered agents."""
        with self._lock:
            self._agents.clear()
            self._agents_by_name.clear()
            self._agents_by_capability.clear()

    def discover_agents(
        self,
        *,
        capabilities: list[str] | None = None,
        status: SubAgentStatus | None = None,
        min_heartbeat_age_seconds: int | None = None,
    ) -> list[SubAgent]:
        """
        Discover agents matching specific criteria.

        This provides advanced service discovery with multiple filters.

        Args:
            capabilities: Required capabilities (all must match)
            status: Required status
            min_heartbeat_age_seconds: Maximum age of last heartbeat

        Returns:
            List of matching agents

        Example:
            ```python
            # Find idle analysis agents with recent heartbeat
            agents = registry.discover_agents(
                capabilities=["analysis"],
                status=SubAgentStatus.IDLE,
                min_heartbeat_age_seconds=30
            )
            ```
        """
        with self._lock:
            agents = list(self._agents.values())

            # Filter by capabilities
            if capabilities:
                agents = [a for a in agents if all(cap in a.capabilities for cap in capabilities)]

            # Filter by status
            if status:
                agents = [a for a in agents if a.status == status]

            # Filter by heartbeat age
            if min_heartbeat_age_seconds is not None:
                now = datetime.now(UTC)
                max_age = timedelta(seconds=min_heartbeat_age_seconds)

                def _to_aware_utc(dt: datetime) -> datetime:
                    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

                agents = [a for a in agents if now - _to_aware_utc(a.last_heartbeat) <= max_age]

            return agents

    def get_agent_load(self, agent_id: str) -> int:
        """
        Get the current load (number of tasks) for an agent.

        This is a placeholder for more sophisticated load tracking.
        Subclasses can override to integrate with actual task tracking.

        Args:
            agent_id: Agent ID

        Returns:
            Current load (0 if agent not found)
        """
        # Basic implementation - could be enhanced with actual task tracking
        agent = self.get_agent(agent_id)
        if agent and agent.status == SubAgentStatus.BUSY:
            return 1
        return 0

    def select_best_agent(
        self,
        capabilities: list[str] | None = None,
        strategy: str = "load_balanced",
    ) -> SubAgent | None:
        """
        Select the best agent for a task using various strategies.

        Args:
            capabilities: Required capabilities
            strategy: Selection strategy:
                - "load_balanced": Choose agent with lowest load
                - "round_robin": Simple round-robin selection
                - "random": Random selection
                - "first": First available agent

        Returns:
            Selected agent or None if no suitable agent

        Example:
            ```python
            # Get least busy analysis agent
            agent = registry.select_best_agent(
                capabilities=["analysis"],
                strategy="load_balanced"
            )
            ```
        """
        # Get candidate agents
        agents = self.find_by_capabilities(capabilities) if capabilities else self.list_agents()

        if not agents:
            return None

        # Filter to active agents
        active_agents = [a for a in agents if a.status != SubAgentStatus.OFFLINE]
        if not active_agents:
            return None

        # Prefer idle agents
        idle_agents = [a for a in active_agents if a.status == SubAgentStatus.IDLE]
        if idle_agents:
            active_agents = idle_agents

        # Apply selection strategy
        if strategy == "load_balanced":
            # Select agent with lowest load
            return min(active_agents, key=lambda a: self.get_agent_load(a.agent_id))
        if strategy == "round_robin":
            # Simple round-robin (could be enhanced with state tracking)
            return active_agents[0]
        if strategy == "random":
            import random

            return random.choice(active_agents)
        # "first" or default
        return active_agents[0]

    def list_capabilities(self) -> list[str]:
        """
        List all available capabilities across all agents.

        Returns:
            Sorted list of unique capabilities
        """
        with self._lock:
            return sorted(self._agents_by_capability.keys())

    def get_agents_with_all_capabilities(self) -> list[SubAgent]:
        """
        Get agents that have the most capabilities.

        Returns:
            List of agents sorted by number of capabilities (descending)
        """
        with self._lock:
            agents = list(self._agents.values())
            return sorted(agents, key=lambda a: len(a.capabilities), reverse=True)
