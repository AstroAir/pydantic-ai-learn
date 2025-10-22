from pydantic_ai import (
    Agent,
    ApprovalRequired,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

agent = Agent(model, output_type=[str, DeferredToolRequests])

PROTECTED_FILES = {".env"}


@agent.tool
def update_file(ctx: RunContext, path: str, content: str) -> str:
    if path in PROTECTED_FILES and not ctx.tool_call_approved:
        raise ApprovalRequired
    return f"File {path!r} updated: {content!r}"


@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    return f"File {path!r} deleted"


first_run = agent.run_sync(
    "Delete `__init__.py`, write `Hello, world!` to `README.md`, and clear `.env`. "
    "Then create a summary of what you did."
)
messages = first_run.all_messages()

assert isinstance(first_run.output, DeferredToolRequests)
requests = first_run.output
print(requests)
"""
DeferredToolRequests(
    calls=[],
    approvals=[
        ToolCallPart(
            tool_name='update_file',
            args={'path': '.env', 'content': ''},
            tool_call_id='update_file_dotenv',
        ),
        ToolCallPart(
            tool_name='delete_file',
            args={'path': '__init__.py'},
            tool_call_id='delete_file',
        ),
    ],
)
"""

results = DeferredToolResults()
for call in requests.approvals:
    if call.tool_name == "update_file":
        # Approve all updates
        results.approvals[call.tool_call_id] = True
    elif call.tool_name == "delete_file":
        # Deny all deletes
        results.approvals[call.tool_call_id] = ToolDenied("Deleting files is not allowed")
    else:
        # Default: not approved
        results.approvals[call.tool_call_id] = False

second_run = agent.run_sync(message_history=messages, deferred_tool_results=results)
print(second_run.output)
"""
I successfully updated `README.md` and cleared `.env`, but was not able to delete `__init__.py`.
"""
print(second_run.all_messages())
"""
[
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Delete `__init__.py`, write `Hello, world!` to `README.md`, and clear `.env`',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='delete_file',
                args={'path': '__init__.py'},
                tool_call_id='delete_file',
            ),
            ToolCallPart(
                tool_name='update_file',
                args={'path': 'README.md', 'content': 'Hello, world!'},
                tool_call_id='update_file_readme',
            ),
            ToolCallPart(
                tool_name='update_file',
                args={'path': '.env', 'content': ''},
                tool_call_id='update_file_dotenv',
            ),
        ],
        usage=RequestUsage(input_tokens=63, output_tokens=21),
        model_name='gpt-5',
        timestamp=datetime.datetime(...),
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='update_file',
                content="File 'README.md' updated: 'Hello, world!'",
                tool_call_id='update_file_readme',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='delete_file',
                content='Deleting files is not allowed',
                tool_call_id='delete_file',
                timestamp=datetime.datetime(...),
            ),
            ToolReturnPart(
                tool_name='update_file',
                content="File '.env' updated: ''",
                tool_call_id='update_file_dotenv',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content=(
                    'I successfully updated `README.md` and cleared `.env`, '
                    'but was not able to delete `__init__.py`.'
                )
            )
        ],
        usage=RequestUsage(input_tokens=79, output_tokens=39),
        model_name='gpt-5',
        timestamp=datetime.datetime(...),
    ),
]
"""
