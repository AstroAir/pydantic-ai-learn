# mypy: ignore-errors
import importlib.util
from datetime import UTC, datetime

# load utils/formatter.py as a module regardless of package layout
spec = importlib.util.spec_from_file_location("utils.formatter", r"d:\\Project\\pydantic-ai-learn\\utils\\formatter.py")
assert spec is not None and spec.loader is not None
fmt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmt_mod)
ConversationFormatter = fmt_mod.ConversationFormatter  # type: Any

messages = [
    {
        "__class__": "ModelRequest",
        "parts": [
            {
                "__class__": "SystemPromptPart",
                "content": "Be a helpful assistant.",
                "timestamp": datetime.now(UTC),
            },
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


class _WrapBase:
    def __init__(self, mapping):
        self._m = mapping

    def __getattr__(self, item):
        if item in self._m:
            return self._m[item]
        raise AttributeError(item)

    def __repr__(self):
        return repr(self._m)


def _make_wrap(mapping):
    class_name = mapping.get("__class__", "Object")
    try:
        dyn = type(class_name, (_WrapBase,), {})
        return dyn(mapping)
    except Exception:
        return _WrapBase(mapping)


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

fmt = ConversationFormatter(width=100, indent=2)
# debug: inspect collected entries for parts
for idx, m in enumerate(wrapped, start=1):
    cls_name = type(m).__name__
    print(f"Message {idx}: type={cls_name}")
    parts = getattr(m, "parts", None)
    print(f"  has parts? {bool(parts)}")
    if parts:
        for pidx, p in enumerate(parts, start=1):
            print(f"    Part {pidx}: type={type(p).__name__}, repr={repr(p)}")
            entries_req = fmt._collect_part_entries(p, context="request")
            entries_res = fmt._collect_part_entries(p, context="response")
            print(f"      _collect_part_entries(request) => {entries_req}")
            print(f"      _collect_part_entries(response) => {entries_res}")

res = fmt.format_conversation(wrapped)
print("---START REPR---")
print(repr(res))
print("---END REPR---")
print("\n---START OUTPUT---")
print(res)
print("---END OUTPUT---")
