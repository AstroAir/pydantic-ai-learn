import asyncio
from dataclasses import dataclass
from datetime import date

from pydantic_ai import Agent, RunContext, ThinkingPartDelta
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from utils.terminal_stream import TerminalStreamComponent

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/",
        api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF",
    ),
)


@dataclass
class WeatherService:
    async def get_forecast(self, location: str, forecast_date: date) -> str:
        return f"The forecast in {location} on {forecast_date} is 24°C and sunny."

    async def get_historic_weather(self, location: str, forecast_date: date) -> str:
        return f"The weather in {location} on {forecast_date} was 18°C and partly cloudy."


weather_agent = Agent[WeatherService, str](
    model,
    deps_type=WeatherService,
    output_type=str,
    system_prompt="Providing a weather forecast at the locations the user provides.",
)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext[WeatherService],
    location: str,
    forecast_date: date,
) -> str:
    if forecast_date >= date.today():
        return await ctx.deps.get_forecast(location, forecast_date)
    return await ctx.deps.get_historic_weather(location, forecast_date)


async def run_with_events() -> None:
    print("=== Streaming with events ===")
    captured_events: list[str] = []
    streamed_chunks: list[str] = []

    def record_event(event: object, message: str | None) -> None:
        if message:
            captured_events.append(message)

    from typing import Any

    async def after_stream(_run: Any) -> None:
        print(f"[post-run] streamed text chunks: {len(streamed_chunks)}")

    component = TerminalStreamComponent(
        show_events=True,
        event_sink=record_event,
        event_filter=lambda event: not isinstance(getattr(event, "delta", None), ThinkingPartDelta),
        text_delta=True,
    )
    result = await component.run_agent(
        weather_agent,
        "What will the weather be like in Paris on 2025/10/12?",
        deps=WeatherService(),
        text_handler=lambda chunk: streamed_chunks.append(chunk),
        on_complete=after_stream,
    )
    final_text = await result.validate_response_output(result.response)
    print("Final result:", final_text)
    if captured_events:
        print(f"Captured {len(captured_events)} events (showing first 3):")
        for line in captured_events[:3]:
            print("  ", line)


async def run_without_events() -> None:
    print("=== Streaming responses without terminal events ===")

    from typing import Any

    def response_printer(message: Any, is_last: bool) -> None:
        text = "".join(
            getattr(part, "content", "") for part in getattr(message, "parts", []) if hasattr(part, "content")
        )
        label = "final" if is_last else "partial"
        print(f"[response {label}] {text}")

    def usage_summary(usage: object) -> None:
        print(f"[usage summary] {usage}")

    component = TerminalStreamComponent(show_events=False, stream_kind="responses")
    result = await component.run_agent(
        weather_agent,
        "Give me a short description of the weather in Tokyo tomorrow.",
        deps=WeatherService(),
        response_handler=response_printer,
        usage_handler=usage_summary,
    )
    final_text = await result.validate_response_output(result.response)
    print("Final result:", final_text)


async def main() -> None:
    await run_with_events()
    await run_without_events()


if __name__ == "__main__":
    asyncio.run(main())
