from function_toolset import datetime_toolset, weather_toolset
from pydantic_ai import Agent, CombinedToolset
from pydantic_ai.models.test import TestModel

combined_toolset = CombinedToolset([weather_toolset, datetime_toolset])

test_model = TestModel()
agent = Agent(test_model, toolsets=[combined_toolset])
result = agent.run_sync("What tools are available?")
print([t.name for t in test_model.last_model_request_parameters.function_tools])
# > ['temperature_celsius', 'temperature_fahrenheit', 'conditions', 'now']
