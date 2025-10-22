from __future__ import annotations

from enum import Enum

from pydantic_ai import Agent, RunContext, Tool, ToolDefinition
from pydantic_ai.models.test import TestModel


class GreetDeps(str, Enum):
    HUMAN = "human"
    MACHINE = "machine"


def greet(name: str) -> str:
    return f"hello {name}"


async def prepare_greet(ctx: RunContext[GreetDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    d = f"Name of the {ctx.deps} to greet."
    tool_def.parameters_json_schema["properties"]["name"]["description"] = d
    return tool_def


greet_tool = Tool(greet, prepare=prepare_greet)
test_model = TestModel()
agent = Agent(test_model, tools=[greet_tool], deps_type=GreetDeps)

result = agent.run_sync("testing...", deps=GreetDeps.MACHINE)
print(result.output)
deps_request = test_model.last_model_request_parameters
if deps_request is None:
    print("No tool invocation recorded.")
else:
    print(deps_request.function_tools)
"""
[
    ToolDefinition(
        name='greet',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {
                'name': {'type': 'string', 'description': 'Name of the human to greet.'}
            },
            'required': ['name'],
            'type': 'object',
        },
    )
]
"""
