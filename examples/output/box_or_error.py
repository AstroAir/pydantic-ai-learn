from pydantic import BaseModel
from pydantic_ai import Agent


class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str


agent = Agent(
    "openai:gpt-4o-mini",
    output_type=[Box, str],
    system_prompt=("Extract me the dimensions of a box, if you can't extract all data, ask the user to try again."),
)

result = agent.run_sync("The box is 10x20x30")
print(result.output)
# > Please provide the units for the dimensions (e.g., cm, in, m).

result = agent.run_sync("The box is 10x20x30 cm")
print(result.output)
# > width=10 height=20 depth=30 units='cm'
