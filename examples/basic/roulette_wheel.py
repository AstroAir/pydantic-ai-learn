from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

roulette_agent = Agent(
    model,
    deps_type=int,
    output_type=bool,
    system_prompt=(
        "Use the `roulette_wheel` function to see if the customer has won based on the number they provide."
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    """check if the square is a winner"""
    return "winner" if square == ctx.deps else "loser"


# Run the agent
success_number = 18
result = roulette_agent.run_sync("Put my money on square eighteen", deps=success_number)
print(result.output)
# > True

result = roulette_agent.run_sync("I bet five is the winner", deps=success_number)
print(result.output)
# > False
