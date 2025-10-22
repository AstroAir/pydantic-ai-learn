"""
Demo: Format a synthetic conversation using utils.formatter.ConversationFormatter
without any network calls. This keeps the library file clean and lets you
preview output styles safely.

Usage (Windows PowerShell):
  python examples/basic/format_conversation_demo.py
  python examples/basic/format_conversation_demo.py --width 100 --indent 2 --no-color --charset ascii
  python examples/basic/format_conversation_demo.py --rich  # requires 'rich' to be installed
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timezone
from typing import Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the formatter and its small wrapping helper to build fake SDK-like objects
from utils.formatter import ConversationFormatter, _make_wrap


def _wrap_messages(messages: list[dict[str, Any]]):
    """Wrap dict messages so their types are "ModelRequest"/"ModelResponse" for the formatter.

    The helper `_make_wrap` creates dynamic objects whose type name matches
    `__class__`, which is what ConversationFormatter uses for dispatch.
    """
    wrapped = []
    for m in messages:
        if m.get("__class__") in ("ModelRequest", "ModelResponse"):
            parts = m.get("parts", []) or []
            new_parts = [_make_wrap(p) for p in parts]
            m2 = dict(m)
            m2["parts"] = new_parts
            wrapped.append(_make_wrap(m2))
        else:
            wrapped.append(_make_wrap(m))
    return wrapped


def build_demo_messages():
    now = datetime.now(UTC)
    messages: list[dict[str, Any]] = [
        {
            "__class__": "ModelRequest",
            "parts": [
                {
                    "__class__": "SystemPromptPart",
                    "content": "You are a helpful assistant.",
                    "timestamp": now,
                },
                {
                    "__class__": "UserPromptPart",
                    "content": "List three benefits of unit tests.",
                    "timestamp": now,
                },
            ],
        },
        {
            "__class__": "ModelResponse",
            "parts": [
                {
                    "__class__": "TextPart",
                    "content": (
                        "- Catches regressions early and improves reliability.\n"
                        "- Documents expected behavior for future readers.\n"
                        "- Enables safe refactoring and faster iteration."
                    ),
                    "timestamp": now,
                }
            ],
            "usage": {"input_tokens": 48, "output_tokens": 32},
            "model_name": "demo-model",
            "provider_name": "demo-provider",
            "timestamp": now,
        },
    ]
    return _wrap_messages(messages)


def parse_args():
    parser = argparse.ArgumentParser(description="Format a demo conversation")
    parser.add_argument("--width", type=int, default=80, help="Output width")
    parser.add_argument("--indent", type=int, default=2, help="Indent size")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors regardless of terminal support",
    )
    parser.add_argument(
        "--charset",
        choices=["utf8", "ascii", "box"],
        default=None,
        help="Tree drawing charset to use (default: auto-detect)",
    )
    parser.add_argument(
        "--rich",
        action="store_true",
        help="Enable rich markdown rendering if 'rich' is installed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    fmt = ConversationFormatter(
        width=args.width,
        indent=args.indent,
        use_color=not args.no_color,
        charset=args.charset,
        use_rich=args.rich,
    )

    out = fmt.format_conversation(build_demo_messages())
    print(out)


if __name__ == "__main__":
    main()
