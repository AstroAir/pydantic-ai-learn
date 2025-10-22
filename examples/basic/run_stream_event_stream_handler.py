import asyncio
from collections.abc import AsyncIterable
from datetime import date

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

weather_agent = Agent(
    model,
    system_prompt="Providing a weather forecast at the locations the user provides.",
)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext,
    location: str,
    forecast_date: date,
) -> str:
    return f"The forecast in {location} on {forecast_date} is 24°C and sunny."


output_messages: list[str] = []


async def handle_event(event: AgentStreamEvent):
    if isinstance(event, PartStartEvent):
        output_messages.append(f"[Request] Starting part {event.index}: {event.part!r}")
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            output_messages.append(f"[Request] Part {event.index} text delta: {event.delta.content_delta!r}")
        elif isinstance(event.delta, ThinkingPartDelta):
            output_messages.append(f"[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}")
        elif isinstance(event.delta, ToolCallPartDelta):
            output_messages.append(f"[Request] Part {event.index} args delta: {event.delta.args_delta}")
    elif isinstance(event, FunctionToolCallEvent):
        output_messages.append(
            f"[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} "
            f"(tool_call_id={event.part.tool_call_id!r})"
        )
    elif isinstance(event, FunctionToolResultEvent):
        output_messages.append(f"[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}")
    elif isinstance(event, FinalResultEvent):
        output_messages.append(f"[Result] The model starting producing a final result (tool_name={event.tool_name})")


async def event_stream_handler(
    ctx: RunContext,
    event_stream: AsyncIterable[AgentStreamEvent],
):
    async for event in event_stream:
        await handle_event(event)


async def main():
    user_prompt = "What will the weather be like in Paris on 2025/10/12?"

    async with weather_agent.run_stream(user_prompt, event_stream_handler=event_stream_handler) as run:
        async for output in run.stream_text():
            output_messages.append(f"[Output] {output}")


if __name__ == "__main__":
    asyncio.run(main())

    print(output_messages)
    """
    [
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        (
            '[Tools] The LLM calls tool=\'weather_forecast\' with '
            'args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')'
        ),
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24°C and sunny.",
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model starting producing a final result (tool_name=None)',
        '[Output] It will be ',
        '[Output] It will be warm and sunny ',
        '[Output] It will be warm and sunny in Paris on ',
        '[Output] It will be warm and sunny in Paris on Tuesday.',
    ]
    """
