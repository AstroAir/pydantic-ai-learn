"""
Hierarchical Agent System Example

Demonstrates how to use the hierarchical agent system with sub-agents
for specialized task delegation.

Author: The Augster
Python Version: 3.12+
"""

import asyncio

from code_agent import (
    CodeAgent,
    HierarchicalAgent,
    HierarchicalAgentConfig,
)


async def basic_hierarchical_example() -> None:
    """Basic example of hierarchical agent with sub-agents."""
    print("=" * 60)
    print("Basic Hierarchical Agent Example")
    print("=" * 60)

    # Create parent agent
    parent_agent = CodeAgent(
        model="openai:gpt-4",
        enable_streaming=False,
    )

    # Create hierarchical agent
    h_agent = HierarchicalAgent(parent_agent)

    # Create specialized sub-agents
    analysis_agent = CodeAgent(
        model="openai:gpt-4",
        enable_streaming=False,
    )

    refactoring_agent = CodeAgent(
        model="openai:gpt-4",
        enable_streaming=False,
    )

    # Register sub-agents with capabilities
    h_agent.register_sub_agent(
        name="Analysis Specialist",
        description="Specialized in code analysis and quality metrics",
        agent=analysis_agent,
        capabilities=["analysis", "metrics", "quality"],
    )

    h_agent.register_sub_agent(
        name="Refactoring Specialist",
        description="Specialized in code refactoring and optimization",
        agent=refactoring_agent,
        capabilities=["refactoring", "optimization"],
    )

    print(f"\nRegistered {len(h_agent.get_sub_agents())} sub-agents")

    # Delegate a task to analysis specialist
    print("\n--- Delegating analysis task ---")
    result = await h_agent.delegate(
        prompt="Analyze the code quality of my_module.py",
        required_capabilities=["analysis"],
    )

    print(f"Task completed: {result.success}")
    print(f"Result: {result.output}")

    # Get statistics
    stats = h_agent.get_stats()
    print("\nHierarchical Agent Stats:")
    print(f"  Total sub-agents: {stats['registry']['total_agents']}")
    print(f"  Total capabilities: {stats['registry']['total_capabilities']}")


async def parallel_delegation_example() -> None:
    """Example of parallel task delegation to multiple sub-agents."""
    print("\n" + "=" * 60)
    print("Parallel Delegation Example")
    print("=" * 60)

    # Create parent agent
    parent_agent = CodeAgent(model="openai:gpt-4")
    h_agent = HierarchicalAgent(parent_agent)

    # Register multiple sub-agents
    for i in range(3):
        sub_agent = CodeAgent(model="openai:gpt-4")
        h_agent.register_sub_agent(
            name=f"Worker Agent {i + 1}",
            description=f"Worker agent {i + 1}",
            agent=sub_agent,
            capabilities=["analysis"],
        )

    # Delegate multiple tasks in parallel
    print("\n--- Delegating multiple tasks in parallel ---")
    tasks = [
        "Analyze file1.py",
        "Analyze file2.py",
        "Analyze file3.py",
    ]

    results = await h_agent.delegate_multiple(
        prompts=tasks,
        required_capabilities=["analysis"],
    )

    print(f"\nCompleted {len(results)} tasks:")
    for i, result in enumerate(results, 1):
        print(f"  Task {i}: {'Success' if result.success else 'Failed'}")


async def smart_delegation_example() -> None:
    """Example of smart delegation based on task analysis."""
    print("\n" + "=" * 60)
    print("Smart Delegation Example")
    print("=" * 60)

    # Create parent agent
    parent_agent = CodeAgent(model="openai:gpt-4")

    # Create hierarchical agent with custom config
    config = HierarchicalAgentConfig(
        enable_sub_agents=True,
        max_delegation_depth=3,
    )
    h_agent = HierarchicalAgent(parent_agent, config)

    # Register specialized sub-agents
    analysis_agent = CodeAgent(model="openai:gpt-4")
    h_agent.register_sub_agent(
        name="Code Analyzer",
        description="Expert in code analysis",
        agent=analysis_agent,
        capabilities=["analysis", "metrics"],
    )

    refactoring_agent = CodeAgent(model="openai:gpt-4")
    h_agent.register_sub_agent(
        name="Code Refactorer",
        description="Expert in refactoring",
        agent=refactoring_agent,
        capabilities=["refactoring", "optimization"],
    )

    testing_agent = CodeAgent(model="openai:gpt-4")
    h_agent.register_sub_agent(
        name="Test Generator",
        description="Expert in test generation",
        agent=testing_agent,
        capabilities=["testing", "quality"],
    )

    # Use smart delegation
    print("\n--- Using smart delegation ---")
    result = await h_agent.run_with_delegation(
        "Analyze and refactor my_module.py with comprehensive tests",
        use_sub_agents=True,
    )

    print(f"Result: {result}")


async def agent_discovery_example() -> None:
    """Example of agent discovery by capabilities."""
    print("\n" + "=" * 60)
    print("Agent Discovery Example")
    print("=" * 60)

    # Create parent agent
    parent_agent = CodeAgent(model="openai:gpt-4")
    h_agent = HierarchicalAgent(parent_agent)

    # Register agents with different capabilities
    agents_config = [
        ("Python Expert", ["python", "analysis", "refactoring"]),
        ("JavaScript Expert", ["javascript", "analysis"]),
        ("Testing Expert", ["testing", "python", "javascript"]),
        ("Performance Expert", ["optimization", "profiling"]),
    ]

    for name, capabilities in agents_config:
        agent = CodeAgent(model="openai:gpt-4")
        h_agent.register_sub_agent(
            name=name,
            description=f"{name} agent",
            agent=agent,
            capabilities=capabilities,
        )

    # Discover agents by capability
    print("\n--- Discovering agents by capability ---")

    python_agents = h_agent.registry.find_by_capability("python")
    print(f"\nAgents with 'python' capability: {len(python_agents)}")
    for sub_agent in python_agents:
        print(f"  - {sub_agent.name}")

    testing_agents = h_agent.registry.find_by_capability("testing")
    print(f"\nAgents with 'testing' capability: {len(testing_agents)}")
    for sub_agent in testing_agents:
        print(f"  - {sub_agent.name}")

    # Find agents with multiple capabilities
    multi_cap_agents = h_agent.registry.find_by_capabilities(["python", "analysis"])
    print(f"\nAgents with both 'python' and 'analysis': {len(multi_cap_agents)}")
    for sub_agent in multi_cap_agents:
        print(f"  - {sub_agent.name}: {sub_agent.capabilities}")


async def health_monitoring_example() -> None:
    """Example of health monitoring for sub-agents."""
    print("\n" + "=" * 60)
    print("Health Monitoring Example")
    print("=" * 60)

    # Create parent agent
    parent_agent = CodeAgent(model="openai:gpt-4")
    h_agent = HierarchicalAgent(parent_agent)

    # Register sub-agents
    for i in range(5):
        agent = CodeAgent(model="openai:gpt-4")
        h_agent.register_sub_agent(
            name=f"Agent {i + 1}",
            description=f"Agent {i + 1}",
            agent=agent,
        )

    # Check health
    print("\n--- Checking agent health ---")
    health = h_agent.registry.check_health()

    print(f"Total agents: {health['total']}")
    print(f"Healthy agents: {health['healthy']}")
    print(f"Offline agents: {health['offline']}")

    # Get detailed stats
    stats = h_agent.registry.get_stats()
    print("\nDetailed statistics:")
    print(f"  Total agents: {stats['total_agents']}")
    print("  Agents by status:")
    for status, count in stats["agents_by_status"].items():
        if count > 0:
            print(f"    {status}: {count}")


async def main() -> None:
    """Run all examples."""
    try:
        await basic_hierarchical_example()
        await parallel_delegation_example()
        await smart_delegation_example()
        await agent_discovery_example()
        await health_monitoring_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
