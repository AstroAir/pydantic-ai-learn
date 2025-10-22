"""
Agent-to-Agent (A2A) Protocol Integration Example

Demonstrates how to use the A2A protocol for inter-agent communication.

Note: Requires fasta2a to be installed:
    pip install 'pydantic-ai-slim[a2a]'

Author: The Augster
Python Version: 3.12+
"""

import asyncio

from code_agent import (
    A2AClient,
    A2AConfig,
    A2AServer,
    CodeAgent,
    HierarchicalAgent,
    HierarchicalAgentConfig,
)


async def basic_a2a_server_example() -> None:
    """Basic example of exposing an agent as an A2A server."""
    print("=" * 60)
    print("Basic A2A Server Example")
    print("=" * 60)

    try:
        # Create a CodeAgent
        agent = CodeAgent(model="openai:gpt-4")

        # Create A2A server
        a2a_config = A2AConfig(
            server_host="0.0.0.0",
            server_port=8000,
        )

        a2a_server = A2AServer(
            agent=agent.agent,  # type: ignore[arg-type]  # PydanticAI agent
            config=a2a_config,
        )

        print("\nA2A server created successfully")
        print(f"Server will listen on {a2a_config.server_host}:{a2a_config.server_port}")
        print("\nTo run the server, use:")
        print("  uvicorn your_module:app --host 0.0.0.0 --port 8000")
        print("\nWhere 'app' is the ASGI application from a2a_server.get_app()")

        # Get the ASGI app
        app = a2a_server.get_app()
        print(f"\nASGI app: {app}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install fasta2a:")
        print("  pip install 'pydantic-ai-slim[a2a]'")


async def a2a_client_example() -> None:
    """Example of using A2A client to call remote agents."""
    print("\n" + "=" * 60)
    print("A2A Client Example")
    print("=" * 60)

    try:
        # Create A2A client
        a2a_config = A2AConfig(
            timeout_seconds=60,
            enable_caching=True,
        )

        # Connect to remote agent
        async with A2AClient("http://localhost:8000", a2a_config) as client:
            print("\nConnected to A2A server at http://localhost:8000")

            # Call remote agent
            print("\n--- Calling remote agent ---")
            result = await client.call_agent(
                prompt="Analyze the code in my_module.py",
                metadata={"priority": "high"},
            )

            print(f"Result: {result}")

            # Make another call (should use cache if enabled)
            print("\n--- Making cached call ---")
            result2 = await client.call_agent(
                prompt="Analyze the code in my_module.py",
                metadata={"priority": "high"},
            )

            print(f"Result: {result2}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install fasta2a:")
        print("  pip install 'pydantic-ai-slim[a2a]'")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure an A2A server is running at http://localhost:8000")


async def hierarchical_with_a2a_example() -> None:
    """Example of hierarchical agent with A2A support."""
    print("\n" + "=" * 60)
    print("Hierarchical Agent with A2A Example")
    print("=" * 60)

    try:
        # Create parent agent
        parent_agent = CodeAgent(model="openai:gpt-4")

        # Create hierarchical agent with A2A enabled
        config = HierarchicalAgentConfig(
            enable_sub_agents=True,
            enable_a2a=True,
            a2a_config=A2AConfig(
                server_host="0.0.0.0",
                server_port=8001,
            ),
        )

        h_agent = HierarchicalAgent(parent_agent, config)

        print("\nHierarchical agent created with A2A support")

        # Register sub-agents
        analysis_agent = CodeAgent(model="openai:gpt-4")
        h_agent.register_sub_agent(
            name="Analysis Agent",
            description="Code analysis specialist",
            agent=analysis_agent,
            capabilities=["analysis"],
        )

        # Get A2A app
        app = h_agent.get_a2a_app()
        if app:
            print(f"\nA2A app available: {app}")
            print("This agent can now be accessed via A2A protocol")
        else:
            print("\nA2A app not available (may need fasta2a installed)")

        # Get stats
        stats = h_agent.get_stats()
        print(f"\nA2A enabled: {stats['a2a_enabled']}")
        print(f"A2A server active: {stats['a2a_server_active']}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install fasta2a:")
        print("  pip install 'pydantic-ai-slim[a2a]'")


async def multi_agent_communication_example() -> None:
    """Example of multiple agents communicating via A2A."""
    print("\n" + "=" * 60)
    print("Multi-Agent Communication Example")
    print("=" * 60)

    print("\nThis example demonstrates how multiple agents can communicate")
    print("using the A2A protocol.")
    print("\nSetup:")
    print("1. Agent A (Analysis) - Port 8000")
    print("2. Agent B (Refactoring) - Port 8001")
    print("3. Agent C (Testing) - Port 8002")
    print("\nEach agent can call other agents via A2A protocol")

    try:
        # Create three specialized agents
        agents_config = [
            ("Analysis Agent", 8000, ["analysis"]),
            ("Refactoring Agent", 8001, ["refactoring"]),
            ("Testing Agent", 8002, ["testing"]),
        ]

        servers = []

        for name, port, _capabilities in agents_config:
            agent = CodeAgent(model="openai:gpt-4")

            a2a_server = A2AServer(
                agent=agent.agent,  # type: ignore[arg-type]
                config=A2AConfig(
                    server_host="0.0.0.0",
                    server_port=port,
                ),
            )

            servers.append((name, port, a2a_server))
            print(f"\n{name} ready on port {port}")

        print("\n--- Agent Communication Flow ---")
        print("1. Client calls Analysis Agent (port 8000)")
        print("2. Analysis Agent calls Refactoring Agent (port 8001)")
        print("3. Refactoring Agent calls Testing Agent (port 8002)")
        print("4. Results flow back through the chain")

        print("\nTo run this example:")
        print("1. Start each agent server in separate terminals")
        print("2. Use A2A clients to orchestrate communication")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install fasta2a:")
        print("  pip install 'pydantic-ai-slim[a2a]'")


async def a2a_with_caching_example() -> None:
    """Example of A2A with response caching."""
    print("\n" + "=" * 60)
    print("A2A with Caching Example")
    print("=" * 60)

    try:
        # Create client with caching enabled
        config = A2AConfig(
            enable_caching=True,
            cache_ttl_seconds=300,  # 5 minutes
        )

        async with A2AClient("http://localhost:8000", config) as client:
            print("\nA2A client with caching enabled")
            print(f"Cache TTL: {config.cache_ttl_seconds} seconds")

            # First call - will hit the server
            print("\n--- First call (cache miss) ---")
            import time

            start = time.time()
            _result1 = await client.call_agent("Analyze code")
            elapsed1 = time.time() - start
            print(f"Time: {elapsed1:.3f}s")

            # Second call - should use cache
            print("\n--- Second call (cache hit) ---")
            start = time.time()
            _result2 = await client.call_agent("Analyze code")
            elapsed2 = time.time() - start
            print(f"Time: {elapsed2:.3f}s")

            print(f"\nSpeedup: {elapsed1 / elapsed2:.2f}x")

            # Clear cache
            client.clear_cache()
            print("\nCache cleared")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install fasta2a:")
        print("  pip install 'pydantic-ai-slim[a2a]'")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure an A2A server is running")


async def main() -> None:
    """Run all examples."""
    try:
        await basic_a2a_server_example()
        await a2a_client_example()
        await hierarchical_with_a2a_example()
        await multi_agent_communication_example()
        await a2a_with_caching_example()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        print("\nNote: Some examples require running A2A servers.")
        print("See the example code for setup instructions.")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
