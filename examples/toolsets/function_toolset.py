from datetime import datetime

from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.models.test import TestModel


def temperature_celsius(city: str) -> float:
    # Dummy variation that uses the city parameter
    return 18.0 + float(len(city) % 7)


def temperature_fahrenheit(city: str) -> float:
    # Convert the city-influenced Celsius temperature to Fahrenheit
    c = temperature_celsius(city)
    return round(c * 9.0 / 5.0 + 32.0, 1)


weather_toolset = FunctionToolset(tools=[temperature_celsius, temperature_fahrenheit])


@weather_toolset.tool
def conditions(ctx: RunContext, city: str) -> str:
    # Use both ctx and city to avoid "not accessed" warnings and show variability
    if (ctx.run_step + len(city)) % 2 == 0:
        return f"It's sunny in {city}"
    return f"It's raining in {city}"


datetime_toolset = FunctionToolset()
datetime_toolset.add_function(lambda: datetime.now(), name="now")

test_model = TestModel()
agent = Agent(test_model)

result = agent.run_sync("What tools are available?", toolsets=[weather_toolset])
request = test_model.last_model_request_parameters
if request is None:
    print("No tool invocation recorded.")
else:
    print([t.name for t in request.function_tools])
# > ['temperature_celsius', 'temperature_fahrenheit', 'conditions']

result = agent.run_sync("What tools are available?", toolsets=[datetime_toolset])
request = test_model.last_model_request_parameters
if request is None:
    print("No tool invocation recorded.")
else:
    print([t.name for t in request.function_tools])
# > ['now']
