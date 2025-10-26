from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

agent = Agent(model)

result_sync = agent.run_sync("What is the capital of Italy?")
print(result_sync.output)
# > The capital of Italy is Rome.


async def main() -> None:
    result = await agent.run("What is the capital of France?")
    print(result.output)
    # > The capital of France is Paris.

    async with agent.run_stream("What is the capital of the UK?") as response:
        async for text in response.stream_text():
            print(text)
            # > The capital of
            # > The capital of the UK is
            # > The capital of the UK is London.

    events: list[AgentStreamEvent | AgentRunResultEvent] = []
    async for event in agent.run_stream_events("What is the capital of Mexico?"):
        events.append(event)
    print(events)
    """
    [
        PartStartEvent(index=0, part=TextPart(content='The capital of ')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='Mexico is Mexico ')),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='City.')),
        AgentRunResultEvent(
            result=AgentRunResult(output='The capital of Mexico is Mexico City.')
        ),
    ]
    """


import asyncio  # noqa: E402

asyncio.run(main())
