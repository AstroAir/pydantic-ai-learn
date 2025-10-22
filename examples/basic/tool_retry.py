from fake_database import DatabaseConn
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext


class ChatResult(BaseModel):
    user_id: int
    message: str


agent = Agent(
    "openai:gpt-4o",
    deps_type=DatabaseConn,
    output_type=ChatResult,
)


@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
    """Get a user's ID from their full name."""
    print(name)
    # > John
    # > John Doe
    user_id = ctx.deps.users.get(name=name)
    if user_id is None:
        raise ModelRetry(f"No user found with name {name!r}, remember to provide their full name")
    return user_id


result = agent.run_sync("Send a message to John Doe asking for coffee next week", deps=DatabaseConn())
print(result.output)
"""
user_id=123 message='Hello John, would you be free for coffee sometime next week? Let me know what works for you!'
"""
