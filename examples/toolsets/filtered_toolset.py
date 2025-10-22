from combined_toolset import combined_toolset
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

filtered_toolset = combined_toolset.filtered(lambda ctx, tool_def: "fahrenheit" not in tool_def.name)

test_model = TestModel()
agent = Agent(test_model, toolsets=[filtered_toolset])
result = agent.run_sync("What tools are available?")
print([t.name for t in test_model.last_model_request_parameters.function_tools])
# > ['weather_temperature_celsius', 'weather_conditions', 'datetime_now']
