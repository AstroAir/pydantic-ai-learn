from datetime import date

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

agent = Agent(
    model,
    deps_type=str,
    instructions="Use the customer's name while replying to them.",
)


@agent.instructions
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.instructions
def add_the_date() -> str:
    return f"The date is {date.today()}."


result = agent.run_sync("What is the date?", deps="Frank")
print(result.output)
# > Hello Frank, the date today is 2032-01-02.

result = agent.run_sync("Is the user Max?", deps="Frank")
print(result.output)
