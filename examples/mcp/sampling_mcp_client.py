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

server = MCPServerStdio("python", args=["generate_svg.py"])
agent = Agent(model, toolsets=[server])


async def main() -> None:
    async with agent:
        agent.set_mcp_sampling_model()
        result = await agent.run("Create an image of a robot in a punk style.")
    print(result.output)
    # > Image file written to robot_punk.svg.


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
