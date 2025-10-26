import asyncio
from typing import Any

import logfire
from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider


async def handle_elicitation(
    context: RequestContext[ClientSession, Any, Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Handle elicitation requests from MCP server."""
    print(f"\n{params.message}")

    if not params.requestedSchema:
        response = input("Response: ")
        return ElicitResult(action="accept", content={"response": response})

    # Collect data for each field
    properties = params.requestedSchema["properties"]
    data = {}

    for field, info in properties.items():
        description = info.get("description", field)

        value = input(f"{description}: ")

        # Convert to proper type based on JSON schema
        if info.get("type") == "integer":
            data[field] = int(value)
        else:
            data[field] = value  # type: ignore[assignment]

    # Confirm
    confirm = input("\nConfirm booking? (y/n/c): ").lower()

    if confirm == "y":
        print("Booking details:", data)
        return ElicitResult(action="accept", content=data)  # type: ignore[arg-type]
    if confirm == "n":
        return ElicitResult(action="decline")
    return ElicitResult(action="cancel")


# Set up MCP server connection
restaurant_server = MCPServerStdio("python", args=["restaurant_server.py"], elicitation_callback=handle_elicitation)


logfire.configure(token="pylf_v1_us_NJrHYjFmpZCLMfF5YrVmTDZmvw1Bd0q30BqBQnlg9sKc")
logfire.instrument_pydantic_ai()

# model = AnthropicModel('glm-4.6', provider=AnthropicProvider(
#    base_url="https://open.bigmodel.cn/api/anthropic/",
#  api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"),
# )

from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402
from pydantic_ai.providers.openrouter import OpenRouterProvider  # noqa: E402

model = OpenAIChatModel(
    "qwen/qwen3-coder:free",
    provider=OpenRouterProvider(api_key="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"),
)


# Create agent
agent = Agent(
    model,
    toolsets=[restaurant_server],
    system_prompt="You are a helpful assistant. If user asks for restaurant booking, use the `book_table` tool.",
)


async def main() -> None:
    """Run the agent to book a restaurant table."""
    async with agent:
        result = await agent.run("Book me a table")
        print(f"\nResult: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
