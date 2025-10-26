import importlib.util
from datetime import UTC, datetime

import pytest

# load formatter module by path
spec = importlib.util.spec_from_file_location("utils.formatter", r"d:\\Project\\pydantic-ai-learn\\utils\\formatter.py")
assert spec is not None, "Failed to load spec"
fmt_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None, "Loader is None"
spec.loader.exec_module(fmt_mod)
ConversationFormatter = fmt_mod.ConversationFormatter
_make_wrap = getattr(fmt_mod, "_make_wrap", None)


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
