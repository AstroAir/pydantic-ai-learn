from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

# Create two servers with different prefixes
weather_server = MCPServerSSE(
    "http://localhost:8000/sse",
    tool_prefix="weather",  # Tools will be prefixed with 'weather_'
)

calculator_server = MCPServerSSE(
    "http://localhost:8000/sse",
    tool_prefix="calc",  # Tools will be prefixed with 'calc_'
)

# Both servers might have a tool named 'get_data', but they'll be exposed as:
# - 'weather_get_data'
# - 'calc_get_data'
agent = Agent("openai:gpt-4o", toolsets=[weather_server, calculator_server])
