"""
Hierarchical Agent with A2A Communication Example

Demonstrates the sub-agent functionality and A2A protocol integration.

Features demonstrated:
- Creating specialized sub-agents with specific capabilities
- Hierarchical task delegation with intelligent routing
- Agent-to-agent communication via A2A protocol
- Service discovery and load balancing
- Task chaining and broadcasting
- Fallback and timeout handling

Author: The Augster
Python Version: 3.12+
"""

import asyncio

from code_agent import (
    CodeAgent,
)


async def basic_hierarchical_example() -> None:
    """Basic example of hierarchical agents with sub-agent delegation."""
    print("=" * 80)
    print("Basic Hierarchical Agent Example")
    print("=" * 80)

    # Create parent agent
    parent_agent = CodeAgent(
        model="openai:gpt-4",
        enable_hierarchical=True,
    )

    # Access the hierarchical agent
    h_agent = parent_agent.hierarchical_agent
    if h_agent is None:
        print("Error: Hierarchical agent not enabled")
        return

    # Create specialized sub-agents
    analysis_agent = CodeAgent(model="openai:gpt-4", enable_streaming=False)
    refactoring_agent = CodeAgent(model="openai:gpt-4", enable_streaming=False)
    testing_agent = CodeAgent(model="openai:gpt-4", enable_streaming=False)

    # Register sub-agents with specific capabilities
    h_agent.register_sub_agent(
        name="Analysis Specialist",
        description="Specialized in code analysis and metrics",
        agent=analysis_agent,
        capabilities=["analysis", "metrics", "complexity"],
    )

    h_agent.register_sub_agent(
        name="Refactoring Specialist",
        description="Expert in code refactoring and optimization",
        agent=refactoring_agent,
        capabilities=["refactoring", "optimization", "patterns"],
    )

    h_agent.register_sub_agent(
        name="Testing Specialist",
        description="Creates comprehensive test suites",
        agent=testing_agent,
        capabilities=["testing", "test-generation", "coverage"],
    )

    print("\n✓ Registered 3 specialized sub-agents")

    # Get sub-agent information
    sub_agents = h_agent.get_sub_agents()
    print(f"\nSub-agents: {len(sub_agents)}")
    for agent in sub_agents:
        info = agent.get_info()
        print(f"  - {info.name}: {', '.join(info.capabilities)}")

    # Delegate a task to a specific capability
    print("\n--- Delegating Analysis Task ---")
    result = await h_agent.delegate(
        prompt="Analyze the code structure in main.py",
        required_capabilities=["analysis"],
    )

    if result.success:
        print(f"✓ Task completed by agent: {result.agent_id}")
        print(f"  Result: {result.output[:100]}...")
    else:
        print(f"✗ Task failed: {result.error}")

    # Get statistics
    stats = h_agent.get_stats()
    print("\n--- Statistics ---")
    print(f"Registry: {stats['registry']}")
    print(f"Delegator: {stats['delegator']}")


async def service_discovery_example() -> None:
    """Example of service discovery and intelligent routing."""
    print("\n" + "=" * 80)
    print("Service Discovery and Intelligent Routing Example")
    print("=" * 80)

    # Create hierarchical agent
    parent_agent = CodeAgent(model="openai:gpt-4", enable_hierarchical=True)
    h_agent = parent_agent.hierarchical_agent
    if h_agent is None:
        print("Error: Hierarchical agent not enabled")
        return

    # Register agents with overlapping capabilities
    for i in range(3):
        agent = CodeAgent(model="openai:gpt-4")
        capabilities = []

        if i == 0:
            capabilities = ["analysis", "python", "fast"]
        elif i == 1:
            capabilities = ["analysis", "python", "thorough"]
        else:
            capabilities = ["analysis", "javascript", "fast"]

        h_agent.register_sub_agent(
            name=f"Agent-{i}",
            description=f"Specialized agent {i}",
            agent=agent,
            capabilities=capabilities,
        )

    # Discover agents by capability
    registry = h_agent.registry

    print("\n--- Service Discovery ---")
    python_agents = registry.find_by_capability("python")
    print(f"Python agents: {len(python_agents)}")

    fast_agents = registry.find_by_capability("fast")
    print(f"Fast agents: {len(fast_agents)}")

    # Find agents with multiple capabilities
    python_analysis_agents = registry.find_by_capabilities(["python", "analysis"])
    print(f"Python + Analysis agents: {len(python_analysis_agents)}")

    # Use advanced discovery
    idle_python_agents = registry.discover_agents(
        capabilities=["python"],
        status=registry.list_agents()[0].status,
        min_heartbeat_age_seconds=60,
    )
    print(f"Idle Python agents (recent heartbeat): {len(idle_python_agents)}")

    # Select best agent using load balancing
    best_agent = registry.select_best_agent(capabilities=["analysis"], strategy="load_balanced")
    if best_agent:
        print(f"\n✓ Selected best agent: {best_agent.name}")
        print(f"  Capabilities: {', '.join(best_agent.capabilities)}")


async def broadcast_example() -> None:
    """Example of broadcasting tasks to multiple agents."""
    print("\n" + "=" * 80)
    print("Task Broadcasting Example")
    print("=" * 80)

    parent_agent = CodeAgent(model="openai:gpt-4", enable_hierarchical=True)
    h_agent = parent_agent.hierarchical_agent
    if h_agent is None:
        print("Error: Hierarchical agent not enabled")
        return

    # Register multiple analysis agents
    for i in range(3):
        agent = CodeAgent(model="openai:gpt-4")
        h_agent.register_sub_agent(
            name=f"Analyzer-{i}",
            description=f"Analysis agent {i}",
            agent=agent,
            capabilities=["analysis"],
        )

    print(f"\n✓ Registered {len(h_agent.get_sub_agents())} agents")

    # Broadcast a task to all agents
    print("\n--- Broadcasting Task ---")
    results = await h_agent.broadcast("Identify potential security issues in the authentication module")

    print(f"\nReceived {len(results)} results:")
    for i, result in enumerate(results):
        if result.success:
            print(f"  ✓ Agent {i}: {result.output[:80]}...")
        else:
            print(f"  ✗ Agent {i}: Failed - {result.error}")


async def task_chaining_example() -> None:
    """Example of chaining tasks across multiple agents."""
    print("\n" + "=" * 80)
    print("Task Chaining Example")
    print("=" * 80)

    parent_agent = CodeAgent(model="openai:gpt-4", enable_hierarchical=True)
    h_agent = parent_agent.hierarchical_agent
    if h_agent is None:
        print("Error: Hierarchical agent not enabled")
        return

    # Register specialized agents
    agents_config = [
        ("Analyzer", ["analysis"]),
        ("Refactorer", ["refactoring"]),
        ("Tester", ["testing"]),
    ]

    for name, capabilities in agents_config:
        agent = CodeAgent(model="openai:gpt-4")
        h_agent.register_sub_agent(
            name=name,
            description=f"{name} specialist",
            agent=agent,
            capabilities=capabilities,
        )

    print("\n✓ Registered analysis pipeline agents")

    # Chain tasks - each depends on previous
    print("\n--- Executing Task Chain ---")
    tasks: list[tuple[str, list[str] | None]] = [
        ("Analyze the module structure", ["analysis"]),
        ("Suggest refactoring improvements", ["refactoring"]),
        ("Generate tests for the refactored code", ["testing"]),
    ]

    results = await h_agent.chain_delegation(tasks)

    print(f"\nCompleted {len(results)} chained tasks:")
    for i, result in enumerate(results):
        task_name = tasks[i][0]
        if result.success:
            print(f"  {i + 1}. ✓ {task_name}")
            print(f"     Result: {str(result.output)[:60]}...")
        else:
            print(f"  {i + 1}. ✗ {task_name}")
            print(f"     Error: {result.error}")


async def a2a_server_example() -> None:
    """Example of exposing agent as A2A server."""
    print("\n" + "=" * 80)
    print("A2A Server Example")
    print("=" * 80)

    try:
        # Create a hierarchical agent
        parent_agent = CodeAgent(model="openai:gpt-4", enable_hierarchical=True)
        h_agent = parent_agent.hierarchical_agent
        if h_agent is None:
            print("Error: Hierarchical agent not enabled")
            return

        # Register some sub-agents
        analysis_agent = CodeAgent(model="openai:gpt-4")
        h_agent.register_sub_agent(
            name="Code Analyzer",
            description="Analyzes code quality and structure",
            agent=analysis_agent,
            capabilities=["analysis", "metrics"],
        )

        # Convert to A2A server
        app = h_agent.to_a2a(
            name="Hierarchical Code Agent",
            description="Multi-agent system with specialized sub-agents",
            version="1.0.0",
            skills=["python", "analysis", "refactoring"],
        )

        print("\n✓ Created A2A ASGI application")
        print(f"  App type: {type(app)}")
        print("\nTo run this server:")
        print("  uvicorn your_module:app --host 0.0.0.0 --port 8000")
        print("\nThe server exposes:")
        print("  - Hierarchical agent with task delegation")
        print("  - All registered sub-agent capabilities")
        print("  - A2A protocol compliance for inter-agent communication")

    except ImportError as e:
        print(f"\n✗ A2A support not available: {e}")
        print("\nTo enable A2A:")
        print("  pip install 'pydantic-ai-slim[a2a]'")


async def a2a_client_example() -> None:
    """Example of calling remote agents via A2A."""
    print("\n" + "=" * 80)
    print("A2A Client Communication Example")
    print("=" * 80)

    try:
        parent_agent = CodeAgent(model="openai:gpt-4", enable_hierarchical=True)
        h_agent = parent_agent.hierarchical_agent
        if h_agent is None:
            print("Error: Hierarchical agent not enabled")
            return

        # Call a remote A2A agent
        print("\nAttempting to call remote agent...")
        print("(This requires an A2A server running at http://localhost:8000)")

        try:
            result = await h_agent.call_agent_via_a2a(
                agent_url="http://localhost:8000",
                prompt="Analyze the authentication module",
                context_id="demo-session-001",
                metadata={"priority": "high"},
            )

            print("\n✓ Received response from remote agent:")
            print(f"  {result}")

        except Exception as e:
            print(f"\n✗ Could not connect to remote agent: {e}")
            print("\nTo test this:")
            print("1. Start an A2A server on port 8000")
            print("2. Run this example again")

    except ImportError as e:
        print(f"\n✗ A2A support not available: {e}")


async def advanced_routing_example() -> None:
    """Example of advanced task routing and fallback."""
    print("\n" + "=" * 80)
    print("Advanced Routing and Fallback Example")
    print("=" * 80)

    parent_agent = CodeAgent(model="openai:gpt-4", enable_hierarchical=True)
    h_agent = parent_agent.hierarchical_agent
    if h_agent is None:
        print("Error: Hierarchical agent not enabled")
        return
    delegator = h_agent.delegator

    # Register primary and backup agents
    fast_agent = CodeAgent(model="openai:gpt-4")
    thorough_agent = CodeAgent(model="openai:gpt-4")

    fast_sub = h_agent.register_sub_agent(
        name="Fast Analyzer",
        description="Quick analysis",
        agent=fast_agent,
        capabilities=["analysis", "fast"],
    )

    thorough_sub = h_agent.register_sub_agent(
        name="Thorough Analyzer",
        description="Comprehensive analysis",
        agent=thorough_agent,
        capabilities=["analysis", "thorough"],
    )

    print("\n✓ Registered fast and thorough analyzers")

    # Try with fallback
    print("\n--- Delegation with Fallback ---")
    result = await delegator.delegate_with_fallback(
        prompt="Analyze code complexity",
        primary_agent_id=fast_sub.agent_id,
        fallback_agent_id=thorough_sub.agent_id,
        required_capabilities=["analysis"],
    )

    if result.success:
        print("✓ Task completed successfully")
        print(f"  Agent: {result.agent_id}")

    # Try with timeout
    print("\n--- Delegation with Timeout ---")
    result = await delegator.delegate_with_timeout(
        prompt="Quick analysis",
        timeout_seconds=30.0,
        required_capabilities=["analysis"],
    )

    if result.success:
        print("✓ Task completed within timeout")
    else:
        print(f"✗ Task failed or timed out: {result.error}")


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 80)
    print("ENHANCED HIERARCHICAL AGENT EXAMPLES")
    print("=" * 80)

    try:
        await basic_hierarchical_example()
        await service_discovery_example()
        await broadcast_example()
        await task_chaining_example()
        await advanced_routing_example()
        await a2a_server_example()
        await a2a_client_example()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("  ✓ Sub-agent registration with capabilities")
        print("  ✓ Intelligent service discovery")
        print("  ✓ Load-balanced task delegation")
        print("  ✓ Task broadcasting to multiple agents")
        print("  ✓ Sequential task chaining")
        print("  ✓ A2A protocol server exposure")
        print("  ✓ A2A protocol client communication")
        print("  ✓ Fallback and timeout handling")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
