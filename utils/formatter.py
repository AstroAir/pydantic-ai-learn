import json
import platform
import sys
import textwrap
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

try:
    from pydantic import BaseModel as _PydanticBaseModel

    _PYDANTIC_AVAILABLE = True
except Exception:
    _PYDANTIC_AVAILABLE = False

    class _PydanticBaseModel:  # type: ignore
        pass


import importlib.util

RICH_AVAILABLE = importlib.util.find_spec("rich") is not None


class ConversationFormatter:
    """Cross-platform formatter for conversation history output."""

    # Tree characters for multiple platform encodings
    TREE_CHARS = {
        "utf8": {"branch": "├─", "last": "└─", "vertical": "│", "horizontal": "─"},
        "ascii": {"branch": "|-", "last": "`-", "vertical": "|", "horizontal": "-"},
        "box": {"branch": "├─", "last": "└─", "vertical": "│", "horizontal": "─"},
    }

    def __init__(
        self,
        width: int = 100,
        indent: int = 2,
        use_color: bool | None = None,
        charset: str | None = None,
        use_rich: bool | None = None,
    ):
        self.width = width
        self.indent = indent

        # Auto-detect color support when not explicitly provided
        if use_color is None:
            self.use_color = self._supports_color()
        else:
            self.use_color = use_color

        # Auto-detect tree character set when unspecified
        if charset is None:
            self.charset = self._detect_charset()
        else:
            self.charset = charset

        self.tree = self.TREE_CHARS[self.charset]

        if use_rich is None:
            self.use_rich = RICH_AVAILABLE
        elif use_rich:
            if not RICH_AVAILABLE:
                raise ImportError("Rich is not installed; install 'rich' or set use_rich=False")
            self.use_rich = True
        else:
            self.use_rich = False

        # ANSI color codes used during formatting
        self.colors = (
            {
                "header": "\033[1;36m",
                "request": "\033[1;32m",
                "response": "\033[1;34m",
                "field": "\033[33m",
                "value": "\033[37m",
                "timestamp": "\033[35m",
                "reset": "\033[0m",
            }
            if self.use_color
            else dict.fromkeys(["header", "request", "response", "field", "value", "timestamp", "reset"], "")
        )

    def _supports_color(self) -> bool:
        """Check whether the current terminal supports ANSI colors."""
        # Windows 10+ consoles support ANSI colors
        if platform.system() == "Windows":
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except Exception:
                return False

        # Unix/Linux/macOS terminals usually support ANSI colors by default
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    # Helpers for robust attribute access across pydantic BaseModel, dataclasses,
    # plain objects and mapping-like objects (dicts).
    def _is_pydantic_model(self, obj: Any) -> bool:
        return _PYDANTIC_AVAILABLE and isinstance(obj, _PydanticBaseModel)

    def _get_field(self, obj: Any, name: str, default: Any = None) -> Any:
        """Get `name` from obj supporting attribute access, mapping access and pydantic BaseModel."""
        if obj is None:
            return default

        # mapping-like (dict)
        try:
            if isinstance(obj, dict):
                return obj.get(name, default)
        except Exception:
            pass

        # pydantic BaseModel
        if self._is_pydantic_model(obj):
            # getattr works for pydantic models; fallback to dict() if available
            try:
                return getattr(obj, name, default)
            except Exception:
                try:
                    if hasattr(obj, "dict") and callable(obj.dict):
                        d = cast(Any, obj).dict()
                        if isinstance(d, dict):
                            return d.get(name, default)
                except Exception:
                    pass
                return default

        # generic mapping-like objects
        if hasattr(obj, "get") and callable(obj.get):
            try:
                return obj.get(name, default)
            except Exception:
                pass

        # fallback to attribute
        return getattr(obj, name, default) if hasattr(obj, name) else default

    def _detect_charset(self) -> str:
        """Detect the best-fit character set for tree characters."""
        # Inspect stdout encoding
        encoding = sys.stdout.encoding or "ascii"

        if encoding.lower() in ["utf-8", "utf8"]:
            # Verify UTF-8 characters can be encoded
            try:
                test_str = "├─└│"
                test_str.encode(encoding)
                return "utf8"
            except (UnicodeEncodeError, LookupError):
                return "ascii"

        # Windows often prefers codepage-specific box drawing characters
        if platform.system() == "Windows" and (encoding.lower().startswith("cp") or encoding.lower().startswith("gb")):
            try:
                "├─└│".encode(encoding)
                return "box"
            except (UnicodeEncodeError, LookupError):
                return "ascii"

        return "ascii"

    def format_conversation(self, messages: list[Any], include_usage_summary: bool = True) -> str:
        """Format an entire conversation history tree."""
        output = []
        header = "CONVERSATION HISTORY"
        separator = self.tree["horizontal"] * self.width

        output.append(self._colorize(self._center_text(header, self.width, self.tree["horizontal"]), "header"))
        output.append(separator)

        total_input_tokens = 0
        total_output_tokens = 0

        for idx, msg in enumerate(messages, 1):
            output.append(f"\n{self._format_message(msg, idx)}")
            if idx < len(messages):
                output.append(self.tree["horizontal"] * self.width)

            if include_usage_summary:
                usage = self._get_field(msg, "usage", None)
                if usage is not None:
                    total_input_tokens += int(self._get_field(usage, "input_tokens", 0) or 0)
                    total_output_tokens += int(self._get_field(usage, "output_tokens", 0) or 0)

        if include_usage_summary and (total_input_tokens or total_output_tokens):
            output.append(self.tree["horizontal"] * self.width)
            output.append("")
            output.extend(self._format_usage_summary(total_input_tokens, total_output_tokens))

        output.append("\n" + separator)
        return "\n".join(output)

    def _format_message(self, msg: Any, index: int) -> str:
        """Format an individual message with contextual styling."""
        msg_type = type(msg).__name__

        if msg_type == "ModelRequest":
            return self._format_request(msg, index)
        if msg_type == "ModelResponse":
            return self._format_response(msg, index)

        # Fallback inference: if an object has 'parts' treat as request/response
        parts = self._get_field(msg, "parts", None)
        if parts is not None:
            # If it has usage or model_name it's likely a response
            if self._get_field(msg, "model_name", None) is not None or self._get_field(msg, "usage", None) is not None:
                return self._format_response(msg, index)
            return self._format_request(msg, index)

        # last resort: generic formatting
        return self._format_generic(msg, index)

    def _format_request(self, request: Any, index: int) -> str:
        """Format a model request payload."""
        output = []

        title = f"[{index}] MODEL REQUEST"
        output.append(self._colorize(title, "request"))

        parts = self._get_field(request, "parts", []) or []
        try:
            total_parts = len(parts)
        except Exception:
            # parts may be an iterator; convert to list for formatting
            parts = list(parts)
            total_parts = len(parts)

        if total_parts:
            output.append(f"{' ' * self.indent}{self.tree['branch']} Parts: {total_parts}")
            for part_idx, part in enumerate(parts, 1):
                is_last_part = part_idx == total_parts
                self._format_part(output, part, part_idx, is_last_part, self.indent)

        return "\n".join(output)

    def _format_part(self, output: list[str], part: Any, part_idx: int, is_last: bool, base_indent: int) -> None:
        """Format an individual request part block."""
        part_type = type(part).__name__
        prefix = " " * base_indent
        connector = self.tree["last"] if is_last else self.tree["branch"]

        output.append(f"{prefix}{self.tree['vertical']}")
        output.append(f"{prefix}{connector} Part {part_idx}: {self._colorize(part_type, 'field')}")

        continuation = " " if is_last else self.tree["vertical"]
        detail_prefix = f"{prefix}{continuation}{' ' * self.indent}"
        part_entries = self._collect_part_entries(part, context="request")
        self._emit_detail_entries(output, detail_prefix, part_entries)

    def _format_response(self, response: Any, index: int) -> str:
        """Format a model response payload."""
        output = []

        title = f"[{index}] MODEL RESPONSE"
        output.append(self._colorize(title, "response"))

        items = []

        # Collect metadata that should be shown with the response
        model_name = self._get_field(response, "model_name", None)
        if model_name is not None:
            items.append(("Model", model_name, False))

        provider_name = self._get_field(response, "provider_name", None)
        if provider_name is not None:
            items.append(("Provider", provider_name, False))

        parts = self._get_field(response, "parts", None)
        if parts is not None:
            items.append(("Content", parts, True))

        usage = self._get_field(response, "usage", None)
        if usage is not None:
            items.append(("Usage", usage, True))

        timestamp = self._get_field(response, "timestamp", None)
        if timestamp is not None:
            items.append(("Time", timestamp, False))

        # Render each collected field in order
        for idx, (label, value, is_complex) in enumerate(items):
            is_last = idx == len(items) - 1
            connector = self.tree["last"] if is_last else self.tree["branch"]
            prefix = " " * self.indent

            if is_complex:
                if label == "Content":
                    output.append(f"{prefix}{connector} {label}:")
                    self._format_content(output, value, is_last)
                elif label == "Usage":
                    output.append(f"{prefix}{connector} {label}:")
                    self._format_usage(output, value, is_last)
            else:
                if label == "Time":
                    value = self._format_timestamp(value)
                    output.append(f"{prefix}{connector} {label}: {self._colorize(value, 'timestamp')}")
                else:
                    output.append(f"{prefix}{connector} {label}: {self._colorize(str(value), 'value')}")

        return "\n".join(output)

    def _format_content(self, output: list[str], parts: list[Any], is_last_item: bool) -> None:
        """Format response content sections, applying optional Markdown."""
        prefix = " " * self.indent
        continuation = " " if is_last_item else self.tree["vertical"]
        try:
            total_parts = len(parts)
        except Exception:
            parts = list(parts)
            total_parts = len(parts)

        for part_idx, part in enumerate(parts, 1):
            part_prefix = f"{prefix}{continuation}{' ' * self.indent}"
            connector = self.tree["last"] if part_idx == total_parts else self.tree["branch"]
            part_type = type(part).__name__
            output.append(f"{part_prefix}{connector} Part {part_idx}: {self._colorize(part_type, 'field')}")

            continuation_symbol = " " if part_idx == total_parts else self.tree["vertical"]
            detail_prefix = f"{part_prefix}{continuation_symbol}{' ' * self.indent}"
            part_entries = self._collect_part_entries(part, context="response")
            self._emit_detail_entries(output, detail_prefix, part_entries)

    def _format_usage(self, output: list[str], usage: Any, is_last_item: bool) -> None:
        """Format usage metrics for input and output tokens."""
        prefix = " " * self.indent
        continuation = " " if is_last_item else self.tree["vertical"]

        input_tokens = self._get_field(usage, "input_tokens", 0)
        output_tokens = self._get_field(usage, "output_tokens", 0)
        output.append(
            f"{prefix}{continuation}{' ' * self.indent}{self.tree['branch']} "
            f"Input tokens: {self._colorize(str(input_tokens), 'value')}"
        )
        output.append(
            f"{prefix}{continuation}{' ' * self.indent}{self.tree['last']} "
            f"Output tokens: {self._colorize(str(output_tokens), 'value')}"
        )

    def _format_usage_summary(self, total_input: int, total_output: int) -> list[str]:
        """Build a usage summary footer aggregating token counts."""
        header = self._colorize("USAGE SUMMARY", "header") if self.use_color else "USAGE SUMMARY"
        prefix = " " * self.indent
        return [
            header,
            f"{prefix}{self.tree['branch']} Total input tokens: {self._colorize(str(total_input), 'value')}",
            f"{prefix}{self.tree['last']} Total output tokens: {self._colorize(str(total_output), 'value')}",
        ]

    def _format_generic(self, obj: Any, index: int) -> str:
        """Fallback formatter for generic dataclass payloads."""
        output = []
        obj_type = type(obj).__name__

        title = f"[{index}] {obj_type.upper()}"
        output.append(self._colorize(title, "header"))

        if is_dataclass(obj):
            field_list = list(fields(obj))
            for idx, field in enumerate(field_list):
                is_last = idx == len(field_list) - 1
                connector = self.tree["last"] if is_last else self.tree["branch"]
                value = getattr(obj, field.name)
                formatted_value = self._format_value(value)
                output.append(
                    f"{' ' * self.indent}{connector} {self._colorize(field.name, 'field')}: {formatted_value}"
                )

        return "\n".join(output)

    def _format_value(self, value: Any) -> str:
        """Format a value based on its type."""
        if isinstance(value, datetime):
            return self._colorize(self._format_timestamp(value), "timestamp")
        if isinstance(value, str):
            return self._colorize(self._truncate_text(value, 80), "value")
        return self._colorize(str(value), "value")

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate long text while preserving whitespace."""
        text = text.replace("\n", " ")
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _wrap_text(self, text: str, width: int) -> list[str]:
        """Wrap long text segments to the configured width."""
        return textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=True)

    def _collect_part_entries(self, part: Any, context: str) -> list[dict[str, Any]]:
        entries = []
        content = self._get_field(part, "content", None)
        if content is not None:
            if context == "response":
                lines = self._prepare_content_lines(content)
                entries.append(
                    {
                        "label": "Content",
                        "lines": lines or [""],
                        "color": None if self.use_rich else "value",
                        "precolored": self.use_rich,
                    }
                )
            else:
                text = self._truncate_text(str(content), 150)
                entries.append(
                    {
                        "label": "Content",
                        "lines": [text],
                        "color": "value",
                        "precolored": False,
                    }
                )

        for attr, label in (
            ("tool_name", "Tool"),
            ("tool_call_id", "Call ID"),
            ("role", "Role"),
            ("mime_type", "MIME Type"),
        ):
            value = self._get_field(part, attr, None)
            if value is not None:
                entries.append(
                    {
                        "label": label,
                        "lines": [str(value)],
                        "color": "value",
                        "precolored": False,
                    }
                )

        args_val = self._get_field(part, "args", None)
        if args_val is not None:
            args_lines = self._to_pretty_lines(args_val)
            entries.append(
                {
                    "label": "Args",
                    "lines": args_lines,
                    "color": "value",
                    "precolored": False,
                }
            )
        metadata_val = self._get_field(part, "metadata", None)
        if metadata_val:
            metadata_lines = self._to_pretty_lines(metadata_val)
            entries.append(
                {
                    "label": "Metadata",
                    "lines": metadata_lines,
                    "color": "value",
                    "precolored": False,
                }
            )
        ts_val = self._get_field(part, "timestamp", None)
        if ts_val is not None:
            ts = self._format_timestamp(ts_val)
            entries.append(
                {
                    "label": "Time",
                    "lines": [ts],
                    "color": "timestamp",
                    "precolored": False,
                }
            )

        return entries

    def _emit_detail_entries(self, output: list[str], base_prefix: str, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return

        for idx, entry in enumerate(entries):
            lines = entry.get("lines") or []
            if not lines:
                continue

            is_last = idx == len(entries) - 1
            connector = self.tree["last"] if is_last else self.tree["branch"]
            label_text = self._colorize(entry["label"], "field")
            first_line = self._apply_color(lines[0], entry.get("color"), entry.get("precolored", False))
            output.append(f"{base_prefix}{connector} {label_text}: {first_line}")

            for continuation_line in lines[1:]:
                cont_symbol = " " if is_last else self.tree["vertical"]
                continuation_prefix = f"{base_prefix}{cont_symbol}{' ' * self.indent}"
                colored_line = self._apply_color(continuation_line, entry.get("color"), entry.get("precolored", False))
                output.append(f"{continuation_prefix}{colored_line}")

    def _apply_color(self, text: str, color_key: str | None, precolored: bool) -> str:
        if precolored or not color_key:
            return text
        return self._colorize(text, color_key)

    def _to_pretty_lines(self, value: Any) -> list[str]:
        if isinstance(value, str):
            lines = value.splitlines() or [""]
            pretty_lines: list[str] = []
            max_width = max(20, self.width - self.indent * 4)
            for line in lines:
                wrapped = self._wrap_text(line, max_width) or [""]
                pretty_lines.extend(wrapped)
            return pretty_lines

        if isinstance(value, set):
            value = sorted(value)

        if isinstance(value, (dict, list, tuple)):
            try:
                text = json.dumps(value, ensure_ascii=not self.use_rich, indent=2, default=self._json_default)
            except TypeError:
                text = str(value)
            return text.splitlines()

        return [str(value)]

    def _json_default(self, obj: Any) -> Any:
        if isinstance(obj, set):
            return sorted(obj)
        return str(obj)

    def _prepare_content_lines(self, text: str) -> list[str]:
        content_width = max(20, self.width - self.indent * 3)
        if self.use_rich:
            return self._render_markdown(text, content_width)
        return self._wrap_text(text, content_width)

    def _render_markdown(self, text: str, content_width: int) -> list[str]:
        if not self.use_rich or not RICH_AVAILABLE:
            return self._wrap_text(text, content_width)
        # Import locally to avoid top-level type assignment issues when rich is missing
        try:
            from rich.console import Console
            from rich.markdown import Markdown
        except ImportError as e:  # pragma: no cover - safety net
            raise RuntimeError("Rich support is unavailable despite use_rich=True") from e

        console = Console(
            width=content_width,
            record=True,
            color_system="truecolor" if self.use_color else None,
            no_color=not self.use_color,
        )

        with console.capture() as capture:
            console.print(Markdown(text))

        rendered = capture.get().rstrip()
        if not rendered:
            return []
        return rendered.splitlines()

    def _format_timestamp(self, ts: datetime) -> str:
        """Format a timestamp in UTC."""
        return ts.strftime("%Y-%m-%d %H:%M:%S UTC")

    def _center_text(self, text: str, width: int, fill_char: str) -> str:
        """Center text within a fill character line."""
        padding = (width - len(text) - 2) // 2
        return f"{fill_char * padding} {text} {fill_char * padding}"

    def _colorize(self, text: str, color_key: str) -> str:
        """Apply an ANSI color if enabled."""
        if not self.use_color:
            return text
        return f"{self.colors[color_key]}{text}{self.colors['reset']}"


# Small helpers used by tests/demo to wrap plain dicts into objects
class _WrapBase:
    def __init__(self, mapping: dict[str, Any]):
        self._m: dict[str, Any] = mapping

    def __getattr__(self, item: str) -> Any:
        if item in self._m:
            return self._m[item]
        raise AttributeError(item)

    def __repr__(self) -> str:
        return repr(self._m)


def _make_wrap(mapping: dict[str, Any]) -> _WrapBase:
    """Return an object wrapping `mapping` whose type name equals mapping['__class__'].

    Tests import and use this to create fake ModelRequest/ModelResponse objects.
    """
    class_name = mapping.get("__class__", "Object")
    try:
        dyn = type(class_name, (_WrapBase,), {})
        return cast(_WrapBase, dyn(mapping))
    except Exception:
        return _WrapBase(mapping)


def _ensure_mapping(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of SDK objects into plain dicts."""
    if isinstance(obj, dict):
        mapping: dict[str, Any] | None = {str(k): v for k, v in obj.items()}
    else:
        mapping = None
        for attr in ("model_dump", "dict"):
            method = getattr(obj, attr, None)
            if callable(method):
                try:
                    candidate = method()
                    if isinstance(candidate, dict):
                        mapping = {str(k): v for k, v in candidate.items()}
                        break
                except Exception:
                    continue

        if mapping is None:
            obj_dict = getattr(obj, "__dict__", None)
            mapping = {str(k): v for k, v in obj_dict.items()} if isinstance(obj_dict, dict) else {"value": repr(obj)}

    if mapping is None:
        mapping = {}
    if "__class__" not in mapping:
        mapping["__class__"] = type(obj).__name__

    return mapping


if __name__ == "__main__":
    # Lightweight local demo: build a tiny synthetic conversation without
    # any network calls or external dependencies. For a richer demo, see
    # basic/format_conversation_demo.py.
    from datetime import datetime

    now = datetime.now(UTC)
    messages = [
        {
            "__class__": "ModelRequest",
            "parts": [
                {"__class__": "SystemPromptPart", "content": "Be a helpful assistant.", "timestamp": now},
                {"__class__": "UserPromptPart", "content": "Tell me a joke in detail.", "timestamp": now},
            ],
        },
        {
            "__class__": "ModelResponse",
            "parts": [
                {
                    "__class__": "TextPart",
                    "content": "Why did the function return early? Because it had too many arguments!",
                    "timestamp": now,
                },
            ],
            "usage": {"input_tokens": 42, "output_tokens": 21},
            "model_name": "demo-model",
            "timestamp": now,
        },
    ]

    wrapped = []
    for m in messages:
        mapping = _ensure_mapping(m)
        if mapping.get("__class__") in ("ModelRequest", "ModelResponse"):
            parts = mapping.get("parts", []) or []
            new_parts = []
            for p in parts:
                p_mapping = _ensure_mapping(p)
                new_parts.append(_make_wrap(p_mapping))
            mapping["parts"] = new_parts
        wrapped.append(_make_wrap(mapping))

    formatter = ConversationFormatter(width=100, indent=2, use_color=True, use_rich=False)
    print(formatter.format_conversation(wrapped))

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    model = AnthropicModel(
        "glm-4.6",
        provider=AnthropicProvider(
            base_url="https://open.bigmodel.cn/api/anthropic/",
            api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF",
        ),
    )
    agent = Agent(
        model, instructions="Be concise, reply with one sentence.", system_prompt="You are a helpful assistant."
    )

    result = agent.run_sync('Where does "hello world" come from?')
    print(result.output)

    result1 = agent.run_sync("Tell me a joke.", message_history=result.new_messages())
    print(result1.output)

    print(formatter.format_conversation(result1.all_messages()))
