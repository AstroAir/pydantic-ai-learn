from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

agent = Agent(model)

result_sync = agent.run_sync(
    "What is the capital of Italy? Answer with just the city.",
    usage_limits=UsageLimits(response_tokens_limit=10),
)
print(result_sync.output)
# > Rome
print(result_sync.usage())
# > RunUsage(input_tokens=62, output_tokens=1, requests=1)

try:
    result_sync = agent.run_sync(
        "What is the capital of Italy? Answer with a paragraph.",
        usage_limits=UsageLimits(response_tokens_limit=10),
    )
except UsageLimitExceeded as e:
    print(e)
    # > Exceeded the output_tokens_limit of 10 (output_tokens=32)
