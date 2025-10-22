from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_servers

# Load all servers from configuration file
servers = load_mcp_servers("mcp_config.json")

import logfire  # noqa: E402
from pydantic_ai.models.anthropic import AnthropicModel  # noqa: E402
from pydantic_ai.providers.anthropic import AnthropicProvider  # noqa: E402

logfire.configure(token="pylf_v1_us_NJrHYjFmpZCLMfF5YrVmTDZmvw1Bd0q30BqBQnlg9sKc")
logfire.instrument_pydantic_ai()

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

# Create agent with all loaded servers
agent = Agent(model, toolsets=servers)


async def main() -> None:
    async with agent:
        result = await agent.run("What is 7 plus 5?")
    print(result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
