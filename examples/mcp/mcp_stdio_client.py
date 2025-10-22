import logfire
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

logfire.configure(token="pylf_v1_us_NJrHYjFmpZCLMfF5YrVmTDZmvw1Bd0q30BqBQnlg9sKc")
logfire.instrument_pydantic_ai()

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

server = MCPServerStdio("uv", args=["run", "mcp-run-python", "stdio"], timeout=10)
agent = Agent(model, toolsets=[server])


async def main() -> None:
    async with agent:
        result = await agent.run("How many days between 2000-01-01 and 2025-03-18?")
    print(result.output)
    # > There are 9,208 days between January 1, 2000, and March 18, 2025.


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
