import logfire
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

logfire.configure(token="pylf_v1_us_NJrHYjFmpZCLMfF5YrVmTDZmvw1Bd0q30BqBQnlg9sKc")
logfire.instrument_pydantic_ai()

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

lite_model = model


joke_selection_agent = Agent(
    model,
    system_prompt=(
        "Use the `joke_factory` to generate some jokes, then choose the best. You must return just a single joke."
    ),
)
joke_generation_agent = Agent(lite_model, output_type=list[str])


@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
    r = await joke_generation_agent.run(
        f"Please generate {count} jokes in brief.",
        usage=ctx.usage,
    )
    return r.output


result = joke_selection_agent.run_sync(
    "Tell me a joke.",
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=1000),
)
print(result.output)
# > Did you hear about the toothpaste scandal? They called it Colgate.
print(result.usage())
# > RunUsage(input_tokens=204, output_tokens=24, requests=3, tool_calls=1)
