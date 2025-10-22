from pydantic_ai import Agent, RunContext, ToolDefinition

agent = Agent("test")


async def only_if_42(ctx: RunContext[int], tool_def: ToolDefinition) -> ToolDefinition | None:
    if ctx.deps == 42:
        return tool_def
    return None


@agent.tool(prepare=only_if_42)  # type: ignore[arg-type]
def hitchhiker(ctx: RunContext[int], answer: str) -> str:
    return f"{ctx.deps} {answer}"


result = agent.run_sync("testing...", deps=41)  # type: ignore[call-overload]
print(result.output)
# > success (no tool calls)
result = agent.run_sync("testing...", deps=42)  # type: ignore[call-overload]
print(result.output)
# > {"hitchhiker":"42 a"}
