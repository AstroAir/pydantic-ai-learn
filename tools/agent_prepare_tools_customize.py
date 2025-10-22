from dataclasses import replace

from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.models.test import TestModel


async def turn_on_strict_if_openai(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    if ctx.model.system == "openai":
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    return tool_defs


test_model = TestModel()
agent = Agent(test_model, prepare_tools=turn_on_strict_if_openai)


@agent.tool_plain
def echo(message: str) -> str:
    return message


agent.run_sync("testing...")
request = test_model.last_model_request_parameters
assert request is not None
assert request.function_tools[0].strict is None

# Set the system attribute of the test_model to 'openai'
test_model._system = "openai"

agent.run_sync("testing with openai...")
request = test_model.last_model_request_parameters
assert request is not None
assert request.function_tools[0].strict
