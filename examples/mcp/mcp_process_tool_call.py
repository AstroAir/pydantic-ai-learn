from typing import Any

import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPServerStdio, ToolResult
from pydantic_ai.models.test import TestModel

logfire.configure(token="pylf_v1_us_NJrHYjFmpZCLMfF5YrVmTDZmvw1Bd0q30BqBQnlg9sKc")
logfire.instrument_pydantic_ai()


async def process_tool_call(
    ctx: RunContext[int],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """A tool call processor that passes along the deps."""
    return await call_tool(name, tool_args, {"deps": ctx.deps})


server = MCPServerStdio("python", args=["mcp_server.py"], process_tool_call=process_tool_call)
agent = Agent(model=TestModel(call_tools=["echo_deps"]), deps_type=int, toolsets=[server])


async def main() -> None:
    async with agent:
        result = await agent.run("Echo with deps set to 42", deps=42)
    print(result.output)
    # > {"echo_deps":{"echo":"This is an echo message","deps":42}}


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
