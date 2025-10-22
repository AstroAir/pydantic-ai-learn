from __future__ import annotations

import inspect
import json
import sys
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any, Literal, TextIO

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
from pydantic_ai.result import StreamedRunResult


class TerminalStreamer:
    """Small helper that prints streamed chunks to a terminal."""

    def __init__(self, *, output: TextIO | None = None) -> None:
        self._output = output or sys.stdout

    def write(self, text: str) -> None:
        self._output.write(text)
        self._output.flush()

    def write_line(self, text: str = "") -> None:
        self.write(f"{text}\n")

    def write_json(self, data: Any, *, indent: int = 2) -> None:
        def _default_serializer(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return repr(obj)

        self.write_line(json.dumps(data, ensure_ascii=True, indent=indent, default=_default_serializer))

    async def stream_text(
        self,
        text_stream: AsyncIterable[str],
        *,
        prefix: str = "",
        ensure_newline: bool = True,
    ) -> None:
        if prefix:
            self.write(prefix)
        async for chunk in text_stream:
            self.write(chunk)
        if ensure_newline:
            self.write_line()


class TerminalEventFormatter:
    """Converts agent stream events into short terminal friendly messages."""

    def format(self, event: AgentStreamEvent) -> str | None:
        if isinstance(event, PartStartEvent):
            return f"[request] part {event.index} start: {event.part!r}"
        if isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                return f"[request] part {event.index} text += {event.delta.content_delta!r}"
            if isinstance(event.delta, ThinkingPartDelta):
                return f"[request] part {event.index} thinking += {event.delta.content_delta!r}"
            if isinstance(event.delta, ToolCallPartDelta):
                args = json.dumps(event.delta.args_delta, ensure_ascii=True)
                return f"[request] part {event.index} args += {args}"
            return None
        if isinstance(event, FunctionToolCallEvent):
            return (
                f"[tool] call {event.part.tool_call_id}: "
                f"{event.part.tool_name}({json.dumps(event.part.args, ensure_ascii=True)})"
            )
        if isinstance(event, FunctionToolResultEvent):
            return f"[tool] result {event.tool_call_id}: {event.result.content}"
        if isinstance(event, FinalResultEvent):
            return f"[result] final output started (tool={event.tool_name})"
        return None


class TerminalStreamComponent:
    """Glue code that streams agent tokens and optional events to the terminal.

    Example
    -------
    ```python
    from pydantic_ai import ThinkingPartDelta

    component = TerminalStreamComponent(
        show_events=False,
        show_usage=True,
        stream_kind='structured',
        event_filter=lambda e: not isinstance(getattr(e, "delta", None), ThinkingPartDelta),
    )
    run = await component.run_agent(
        agent,
        "List two fruit objects as JSON",
        output_handler=lambda chunk: print("chunk", chunk),
        on_complete=lambda r: print("finished"),
    )
    final_text = await run.validate_response_output(run.response)
    print("final:", final_text)
    ```
    """

    def __init__(
        self,
        *,
        streamer: TerminalStreamer | None = None,
        event_formatter: TerminalEventFormatter | None = None,
        show_events: bool = True,
        show_usage: bool = True,
        stream_kind: Literal["text", "structured", "responses"] = "text",
        text_delta: bool = True,
        text_ensure_newline: bool = True,
        structured_indent: int = 2,
        event_sink: Callable[[AgentStreamEvent, str | None], Awaitable[None] | None] | None = None,
        event_filter: Callable[[AgentStreamEvent], bool] | None = None,
    ) -> None:
        self._streamer = streamer or TerminalStreamer()
        self._event_formatter = event_formatter or TerminalEventFormatter()
        self._show_events = show_events
        self._show_usage = show_usage
        self._stream_kind = stream_kind
        self._text_delta = text_delta
        self._text_ensure_newline = text_ensure_newline
        self._structured_indent = structured_indent
        self._event_sink = event_sink
        self._event_filter = event_filter

    async def _dispatch_event(self, event: AgentStreamEvent, message: str | None) -> None:
        if self._event_sink is None:
            return
        maybe = self._event_sink(event, message)
        if inspect.isawaitable(maybe):
            await maybe

    def _build_event_stream_handler(
        self,
    ) -> Callable[[RunContext[Any], AsyncIterable[AgentStreamEvent]], Any]:
        formatter = self._event_formatter
        streamer = self._streamer

        async def handler(
            ctx: RunContext[Any],
            event_stream: AsyncIterable[AgentStreamEvent],
        ) -> None:
            async for event in event_stream:
                should_emit = self._event_filter(event) if self._event_filter is not None else True
                message = None
                if should_emit:
                    message = formatter.format(event)
                    if message:
                        streamer.write_line(message)
                await self._dispatch_event(event, message)

        return handler

    @staticmethod
    async def _maybe_await(result: Any) -> None:
        if inspect.isawaitable(result):
            await result

    def _format_response_message(self, message: Any, *, is_last: bool) -> str:
        prefix = "[response-final]" if is_last else "[response]"
        return f"{prefix} {message!r}"

    async def run_agent(
        self,
        agent: Agent[Any, Any],
        prompt: str,
        *,
        deps: Any | None = None,
        prefix: str = "Assistant: ",
        show_events: bool | None = None,
        stream_kind: Literal["text", "structured", "responses"] | None = None,
        text_delta: bool | None = None,
        text_handler: Callable[[str], Awaitable[None] | None] | None = None,
        text_ensure_newline: bool | None = None,
        output_handler: Callable[[Any], Awaitable[None] | None] | None = None,
        response_handler: Callable[[Any, bool], Awaitable[None] | None] | None = None,
        show_usage: bool | None = None,
        usage_handler: Callable[[Any], Awaitable[None] | None] | None = None,
        usage_label: str = "Usage:",
        structured_indent: int | None = None,
        on_complete: Callable[[StreamedRunResult[Any, Any]], Awaitable[None] | None] | None = None,
    ) -> StreamedRunResult[Any, Any]:
        event_stream_handler = None
        if show_events if show_events is not None else self._show_events:
            event_stream_handler = self._build_event_stream_handler()

        async with agent.run_stream(
            prompt,
            deps=deps,
            event_stream_handler=event_stream_handler,
        ) as run:
            kind = stream_kind or self._stream_kind
            delta = self._text_delta if text_delta is None else text_delta
            indent = self._structured_indent if structured_indent is None else structured_indent
            ensure_newline = self._text_ensure_newline if text_ensure_newline is None else text_ensure_newline

            if kind == "text":
                text_stream = run.stream_text(delta=delta)
                if prefix:
                    self._streamer.write(prefix)
                async for chunk in text_stream:
                    self._streamer.write(chunk)
                    if text_handler is not None:
                        await self._maybe_await(text_handler(chunk))
                if ensure_newline:
                    self._streamer.write_line()
            elif kind == "structured":
                async for chunk in run.stream_output():
                    if output_handler is not None:
                        await self._maybe_await(output_handler(chunk))
                    else:
                        self._streamer.write_json(chunk, indent=indent)
            elif kind == "responses":
                async for message, is_last in run.stream_responses():
                    if response_handler is not None:
                        await self._maybe_await(response_handler(message, is_last))
                    else:
                        self._streamer.write_line(self._format_response_message(message, is_last=is_last))
            else:
                raise ValueError(f"Unsupported stream_kind: {kind}")

            if show_usage if show_usage is not None else self._show_usage:
                usage = run.usage()
                if usage_handler is not None:
                    await self._maybe_await(usage_handler(usage))
                else:
                    self._streamer.write_line(f"{usage_label} {usage}")

        if on_complete is not None:
            await self._maybe_await(on_complete(run))

        return run
