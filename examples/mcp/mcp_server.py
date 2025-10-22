from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

mcp = FastMCP("Pydantic AI MCP Server")
log_level = "unset"


@mcp.tool()
async def echo_deps(ctx: Context[ServerSession, None]) -> dict[str, Any]:
    """Echo the run context.

    Args:
        ctx: Context object containing request and session information.

    Returns:
        Dictionary with an echo message and the deps.
    """
    await ctx.info("This is an info message")

    deps: Any = ctx.request_context.meta.deps
    return {"echo": "This is an echo message", "deps": deps}


if __name__ == "__main__":
    mcp.run()
