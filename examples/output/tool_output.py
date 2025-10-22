from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)


class Fruit(BaseModel):
    name: str
    color: str


class Vehicle(BaseModel):
    name: str
    wheels: int


agent = Agent(
    model,
    output_type=[
        ToolOutput(Fruit, name="return_fruit"),
        ToolOutput(Vehicle, name="return_vehicle"),
    ],
)
result = agent.run_sync("What is a banana?")
print(type(result.output))
print(result.output)
# > Fruit(name='banana', color='yellow')

result = agent.run_sync("What is a r?")
print(type(result.output))
print(result.output)
