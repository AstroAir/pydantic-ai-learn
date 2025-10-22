from pydantic_ai import Agent, ModelRetry, UsageLimitExceeded, UsageLimits
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from typing_extensions import TypedDict


class NeverOutputType(TypedDict):
    """
    Never ever coerce data to this type.
    """

    never_use_this: str


model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)


agent = Agent(
    model,
    retries=3,
    output_type=NeverOutputType,
    system_prompt="Any time you get a response, call the `infinite_retry_tool` to produce another response.",
)


@agent.tool_plain(retries=5)
def infinite_retry_tool() -> int:
    raise ModelRetry("Please try again.")


try:
    result_sync = agent.run_sync("Begin infinite retry loop!", usage_limits=UsageLimits(request_limit=3))
except UsageLimitExceeded as e:
    print(e)
    # > The next request would exceed the request_limit of 3
