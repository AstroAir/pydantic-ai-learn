from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent(model, output_type=CityLocation)
result = agent.run_sync("Where were the olympics held in 2012?")
print(result.output)
# > city='London' country='United Kingdom'
print(result.usage())
# > RunUsage(input_tokens=57, output_tokens=8, requests=1)
