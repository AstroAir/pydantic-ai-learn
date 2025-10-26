import importlib.util
from datetime import UTC, datetime

import pytest

# import the formatter module by path to ensure we get the latest file
spec = importlib.util.spec_from_file_location("utils.formatter", r"d:\\Project\\pydantic-ai-learn\\utils\\formatter.py")
if spec is None or spec.loader is None:
    raise ImportError("Failed to load formatter module")
fmt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmt_mod)
ConversationFormatter = fmt_mod.ConversationFormatter
_make_wrap = getattr(fmt_mod, "_make_wrap", None)


def make_wrapped_messages():
    messages = [
        {
            "__class__": "ModelRequest",
            "parts": [
                {"__class__": "SystemPromptPart", "content": "Be a helpful assistant.", "timestamp": datetime.now(UTC)},
                {"__class__": "UserPromptPart", "content": "Tell me a joke.", "timestamp": datetime.now(UTC)},
            ],
        },
        {
            "__class__": "ModelResponse",
            "parts": [
                {
                    "__class__": "TextPart",
                    "content": "Did you hear about the toothpaste scandal? They called it Colgate.",
                }
            ],
            "usage": {"input_tokens": 60, "output_tokens": 12},
            "model_name": "gpt-4o",
            "timestamp": datetime.now(UTC),
        },
    ]

    if _make_wrap is None:
        pytest.skip("_make_wrap helper missing from formatter module")

    wrapped = []
    for m in messages:
        if m.get("__class__") in ("ModelRequest", "ModelResponse"):
            parts = m.get("parts", [])
            new_parts = [_make_wrap(p) for p in parts]
            m2 = dict(m)
            m2["parts"] = new_parts
            wrapped.append(_make_wrap(m2))
        else:
            wrapped.append(_make_wrap(m))
    return wrapped


def test_with_dict_messages():
    wrapped = make_wrapped_messages()
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation(wrapped)
    assert "MODEL REQUEST" in out
    assert "MODEL RESPONSE" in out
    assert "Be a helpful assistant." in out
    assert "Tell me a joke." in out
    assert "Did you hear about the toothpaste scandal?" in out
    assert "Total input tokens" in out


# dataclass test
from dataclasses import dataclass  # noqa: E402


@dataclass
class SimplePart:
    content: str
    timestamp: datetime


@dataclass
class SimpleRequest:
    parts: list[SimplePart]


@dataclass
class SimpleResponse:
    parts: list[SimplePart]
    usage: object
    model_name: str
    timestamp: datetime


def test_with_dataclass_messages():
    req = SimpleRequest(parts=[SimplePart(content="Hello", timestamp=datetime.now(UTC))])
    resp = SimpleResponse(
        parts=[SimplePart(content="World", timestamp=datetime.now(UTC))],
        usage=type("U", (), {"input_tokens": 1, "output_tokens": 2})(),
        model_name="m",
        timestamp=datetime.now(UTC),
    )
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation([req, resp])
    assert "Hello" in out
    assert "World" in out


# pydantic test (optional)
try:
    from pydantic import BaseModel

    class PPart(BaseModel):
        content: str
        timestamp: datetime

    class PRequest(BaseModel):
        parts: list[PPart]

    class PResponse(BaseModel):
        parts: list[PPart]
        usage: object
        model_name: str
        timestamp: datetime

    def test_with_pydantic_models():
        req = PRequest(parts=[PPart(content="Hi", timestamp=datetime.now(UTC))])
        resp = PResponse(
            parts=[PPart(content="There", timestamp=datetime.now(UTC))],
            usage=type("U", (), {"input_tokens": 3, "output_tokens": 4})(),
            model_name="m",
            timestamp=datetime.now(UTC),
        )
        fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
        out = fmt.format_conversation([req, resp])
        assert "Hi" in out
        assert "There" in out
except Exception:
    # pydantic not installed -> skip
    pass


# Additional helpers for building wrapped requests/responses
def _wrap_request(parts):
    if _make_wrap is None:
        pytest.skip("_make_wrap helper missing from formatter module")
    wrapped_parts = [_make_wrap(p) if isinstance(p, dict) else p for p in parts]
    return _make_wrap({"__class__": "ModelRequest", "parts": wrapped_parts})


def _wrap_response(parts, usage=None, model_name=None, provider_name=None, timestamp=None):
    if _make_wrap is None:
        pytest.skip("_make_wrap helper missing from formatter module")
    wrapped_parts = [_make_wrap(p) if isinstance(p, dict) else p for p in parts]
    m = {"__class__": "ModelResponse", "parts": wrapped_parts}
    if usage is not None:
        m["usage"] = usage
    if model_name is not None:
        m["model_name"] = model_name
    if provider_name is not None:
        m["provider_name"] = provider_name
    if timestamp is not None:
        m["timestamp"] = timestamp
    return _make_wrap(m)


def test_charset_ascii_and_utf8_branch_chars():
    req = _wrap_request([{"__class__": "UserPromptPart", "content": "Hello"}])
    fmt_ascii = ConversationFormatter(width=60, indent=2, use_color=False, use_rich=False, charset="ascii")
    out_ascii = fmt_ascii.format_conversation([req])
    assert "|- Parts: 1" in out_ascii

    fmt_utf8 = ConversationFormatter(width=60, indent=2, use_color=False, use_rich=False, charset="utf8")
    out_utf8 = fmt_utf8.format_conversation([req])
    assert "├─ Parts: 1" in out_utf8


def test_color_output_when_enabled():
    req = _wrap_request([{"__class__": "UserPromptPart", "content": "Hi"}])
    resp = _wrap_response(
        [{"__class__": "TextPart", "content": "There"}],
        usage={"input_tokens": 1, "output_tokens": 1},
        model_name="m",
    )
    fmt = ConversationFormatter(width=60, indent=2, use_color=True, use_rich=False, charset="ascii")
    out = fmt.format_conversation([req, resp])
    assert "\x1b[" in out  # ANSI escape present


def test_usage_summary_disabled():
    resp = _wrap_response(
        [{"__class__": "TextPart", "content": "X"}],
        usage={"input_tokens": 5, "output_tokens": 7},
        model_name="m",
    )
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation([resp], include_usage_summary=False)
    assert "USAGE SUMMARY" not in out
    assert "Total input tokens" not in out


def test_args_and_metadata_serialization():
    part = {
        "__class__": "ToolCallPart",
        "content": "Running tool",
        "args": {"a": 1, "b": 2},
        "metadata": {"note": "ok", "tags": {"beta", "alpha"}},
    }
    req = _wrap_request([part])
    fmt = ConversationFormatter(width=100, indent=2, use_color=False, use_rich=False, charset="ascii")
    out = fmt.format_conversation([req])
    assert "Args:" in out
    assert '"a": 1' in out and '"b": 2' in out
    assert "Metadata:" in out
    # Set serialized as sorted list of strings
    assert '"alpha"' in out and '"beta"' in out


def test_response_content_wrapping_use_rich_false():
    long_text = "This is a very long line that should wrap across multiple lines when the width is small. " * 3
    resp = _wrap_response([{"__class__": "TextPart", "content": long_text}])
    fmt = ConversationFormatter(width=40, indent=2, use_color=False, use_rich=False, charset="ascii")
    out = fmt.format_conversation([resp])
    # The full unwrapped string should not appear as a single segment
    assert long_text not in out
    # But parts of it do
    assert "This is a very long line" in out


def test_request_content_truncation_with_long_text():
    long_text = "x" * 200
    req = _wrap_request([{"__class__": "UserPromptPart", "content": long_text}])
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False, charset="ascii")
    out = fmt.format_conversation([req])
    assert "..." in out
    assert long_text not in out  # truncated version only


def test_generic_dataclass_formatting() -> None:
    from dataclasses import dataclass

    @dataclass
    class MiscEvent:
        a: int
        b: str

    evt = MiscEvent(a=42, b="ok")
    fmt = ConversationFormatter(width=60, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation([evt])
    assert "[1] MISCEVENT" in out
    assert "a:" in out and "42" in out
    assert "b:" in out and "ok" in out


def test_provider_and_model_fields_in_response():
    resp = _wrap_response(
        [{"__class__": "TextPart", "content": "hi"}],
        usage={"input_tokens": 1, "output_tokens": 1},
        model_name="gpt-x",
        provider_name="SomeProvider",
    )
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation([resp])
    assert "Model: gpt-x" in out
    assert "Provider: SomeProvider" in out


def test_timestamp_formatting_on_parts_and_response():
    ts = datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    req = _wrap_request([{"__class__": "UserPromptPart", "content": "c", "timestamp": ts}])
    resp = _wrap_response(
        [{"__class__": "TextPart", "content": "d", "timestamp": ts}],
        model_name="m",
        usage={"input_tokens": 1, "output_tokens": 1},
        timestamp=ts,
    )
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation([req, resp])
    assert "2020-01-02 03:04:05 UTC" in out


def test_mapping_like_part_supports_get_field():
    class DictLikePart:
        def __init__(self, data):
            self._d = data

        def get(self, key, default=None):
            return self._d.get(key, default)

    part = DictLikePart({"content": "mapped content", "timestamp": None})
    req = _wrap_request([part])
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation([req])
    assert "mapped content" in out


def test_usage_lines_in_response():
    resp = _wrap_response(
        [{"__class__": "TextPart", "content": "payload"}],
        usage={"input_tokens": 12, "output_tokens": 34},
        model_name="m",
    )
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation([resp])
    assert "Input tokens: 12" in out
    assert "Output tokens: 34" in out


# Realistic tests using pydantic_ai message classes when available
try:
    from pydantic_ai.messages import (
        ModelRequest as PAIModelRequest,
    )
    from pydantic_ai.messages import (
        ModelResponse as PAIModelResponse,
    )
    from pydantic_ai.messages import (
        SystemPromptPart as PAISystemPromptPart,
    )
    from pydantic_ai.messages import (
        TextPart as PAITextPart,
    )
    from pydantic_ai.messages import (
        UserPromptPart as PAIUserPromptPart,
    )
    from pydantic_ai.usage import RequestUsage as PAIRequestUsage

    def test_with_pydantic_ai_messages():
        now = datetime.now(UTC)
        req = PAIModelRequest(
            parts=[
                PAISystemPromptPart(content="Be a helpful assistant.", timestamp=now),
                PAIUserPromptPart(content="Tell me a joke.", timestamp=now),
            ]
        )
        resp = PAIModelResponse(
            parts=[PAITextPart(content="Did you hear about the toothpaste scandal? They called it Colgate.")],
            usage=PAIRequestUsage(input_tokens=60, output_tokens=12),
            model_name="gpt-4o",
            timestamp=now,
        )

        fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
        out = fmt.format_conversation([req, resp])
        assert "MODEL REQUEST" in out
        assert "MODEL RESPONSE" in out
        assert "Be a helpful assistant." in out
        assert "Tell me a joke." in out
        assert "toothpaste scandal" in out
        assert "Total input tokens" in out

    def test_pydantic_ai_usage_summary_disabled():
        now = datetime.now(UTC)
        resp = PAIModelResponse(
            parts=[PAITextPart(content="X")],
            usage=PAIRequestUsage(input_tokens=5, output_tokens=7),
            model_name="m",
            timestamp=now,
        )
        fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
        out = fmt.format_conversation([resp], include_usage_summary=False)
        assert "USAGE SUMMARY" not in out
        assert "Total input tokens" not in out

except Exception:
    # pydantic_ai not installed or API changed; skip these realistic tests
    pass

# --- merged from test_formatter_extended.py ---


def _wrap_messages(messages):
    if _make_wrap is None:
        pytest.skip("_make_wrap helper missing from formatter module")
    wrapped = []
    for m in messages:
        if m.get("__class__") in ("ModelRequest", "ModelResponse"):
            parts = m.get("parts", [])
            new_parts = [_make_wrap(p) for p in parts]
            m2 = dict(m)
            m2["parts"] = new_parts
            wrapped.append(_make_wrap(m2))
        else:
            wrapped.append(_make_wrap(m))
    return wrapped


def test_metadata_and_args_and_tool_fields():
    messages = [
        {
            "__class__": "ModelRequest",
            "parts": [
                {
                    "__class__": "ToolPart",
                    "content": "result",
                    "tool_name": "search",
                    "tool_call_id": "call-123",
                    "mime_type": "application/json",
                    "args": {"q": "python", "page": 1},
                    "metadata": {"nested": {"a": 1, "b": [1, 2]}},
                    "timestamp": datetime.now(UTC),
                }
            ],
        }
    ]

    wrapped = _wrap_messages(messages)
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation(wrapped)
    assert "Tool" in out or "tool" in out
    assert "Args" in out
    assert "Metadata" in out
    assert "search" in out


def test_usage_aggregation_over_multiple_responses():
    messages = []
    for i in range(3):
        messages.append(
            {
                "__class__": "ModelResponse",
                "parts": [{"__class__": "TextPart", "content": f"msg{i}"}],
                "usage": {"input_tokens": 10 + i, "output_tokens": 2 + i},
                "model_name": "m",
                "timestamp": datetime.now(UTC),
            }
        )

    wrapped = _wrap_messages(messages)
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation(wrapped)
    # totals: input = 10+11+12=33, output=2+3+4=9
    assert "Total input tokens" in out and "33" in out
    assert "Total output tokens" in out and "9" in out


def test_parts_as_generator_handling():
    # parts provided as generator
    parts = ({"__class__": "TextPart", "content": f"gen{i}"} for i in range(2))
    messages = [{"__class__": "ModelRequest", "parts": parts}]
    wrapped = _wrap_messages(messages)
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation(wrapped)
    assert "gen0" in out and "gen1" in out


def test_set_and_nested_metadata_serialization():
    messages = [
        {
            "__class__": "ModelRequest",
            "parts": [
                {
                    "__class__": "DataPart",
                    "content": "data",
                    "metadata": {"tags": {"b", "a"}, "values": [1, {"x": "y"}]},
                }
            ],
        }
    ]
    wrapped = _wrap_messages(messages)
    fmt = ConversationFormatter(width=80, indent=2, use_color=False, use_rich=False)
    out = fmt.format_conversation(wrapped)
    # sets should be serialized deterministically
    assert "tags" in out
    assert "values" in out


def test_optional_rich_markdown_rendering():
    # only run if rich is available in module
    if not getattr(fmt_mod, "RICH_AVAILABLE", False):
        pytest.skip("rich not available")

    md = "# Title\n\n- item1\n- item2"
    messages = [
        {
            "__class__": "ModelResponse",
            "parts": [{"__class__": "TextPart", "content": md}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "model_name": "m",
            "timestamp": datetime.now(UTC),
        }
    ]
    wrapped = _wrap_messages(messages)
    fmt = ConversationFormatter(width=60, indent=2, use_color=False, use_rich=True)
    out = fmt.format_conversation(wrapped)
    # rendered markdown should include 'Title' or list markers
    assert "Title" in out or "item1" in out
