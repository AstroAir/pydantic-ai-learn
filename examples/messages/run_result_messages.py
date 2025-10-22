from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/",
        api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF",
    ),
)

agent = Agent(model, system_prompt="Be a helpful assistant.")

result = agent.run_sync("Tell me a joke.")
print(result.output)
# > Did you hear about the toothpaste scandal? They called it Colgate.

# all messages from the run
print(result.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                timestamp=datetime.datetime(...),
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=RequestUsage(input_tokens=60, output_tokens=12),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
]
"""
