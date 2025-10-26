"""
Example usage of filesystem_tools module.

This file demonstrates how to use the Glob, Grep, and LS tools both
standalone and integrated with PydanticAI agents.

Run with: python examples/tools/filesystem_tools_example.py

Author: The Augster
Python Version: 3.12+
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic_ai import Agent

from tools.filesystem_tools import (
    GlobInput,
    GrepInput,
    LSInput,
    glob_files,
    glob_tool,
    grep_search,
    grep_tool,
    ls_directory,
    ls_tool,
)

# ============================================================================
# Standalone Usage Examples
# ============================================================================


def example_glob_basic() -> None:
    """Example: Basic glob pattern matching."""
    print("\n=== Example: Basic Glob ===")

    # Find all Python files in current directory
    result = glob_files(GlobInput(pattern="*.py"))

    print(f"Found {result.total_count} Python files:")
    for file in result.files[:5]:  # Show first 5
        print(f"  - {file}")

    if result.truncated:
        print(f"  ... and {result.total_count - len(result.files)} more")


def example_glob_recursive() -> None:
    """Example: Recursive glob pattern."""
    print("\n=== Example: Recursive Glob ===")

    # Find all TypeScript files recursively
    result = glob_files(GlobInput(pattern="**/*.ts", path="./src", max_results=10))

    print(f"Found {result.total_count} TypeScript files in ./src:")
    for file in result.files:
        print(f"  - {file}")


def example_glob_multiple_extensions() -> None:
    """Example: Glob with multiple file extensions."""
    print("\n=== Example: Multiple Extensions ===")

    # Find JavaScript and TypeScript files
    result = glob_files(GlobInput(pattern="**/*.{js,ts,jsx,tsx}", path="./src"))

    print(f"Found {result.total_count} JS/TS files:")
    for file in result.files[:10]:
        print(f"  - {file}")


def example_grep_find_todos() -> None:
    """Example: Find TODO comments in code."""
    print("\n=== Example: Find TODOs ===")

    try:
        result = grep_search(
            GrepInput(pattern="TODO|FIXME|XXX", path="./src", output_mode="files_with_matches", ignore_case=True)
        )

        print(f"Files with TODO/FIXME/XXX comments ({result.total_matches}):")
        for file in result.files or []:
            print(f"  - {file}")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: This example requires ripgrep to be installed")


def example_grep_with_context() -> None:
    """Example: Search with context lines."""
    print("\n=== Example: Grep with Context ===")

    try:
        result = grep_search(
            GrepInput(
                pattern="class\\s+\\w+", glob="*.py", output_mode="content", line_number=True, context=2, head_limit=5
            )
        )

        print(f"Found {result.total_matches} class definitions:")
        for match in result.matches or []:
            print(f"\n{match.file_path}:{match.line_number}")
            print(f"  {match.line_content}")

    except Exception as e:
        print(f"Error: {e}")


def example_grep_count_matches() -> None:
    """Example: Count matches per file."""
    print("\n=== Example: Count Matches ===")

    try:
        result = grep_search(GrepInput(pattern="import", path="./tools", glob="*.py", output_mode="count"))

        print("Import statements per file:")
        for file, count in (result.counts or {}).items():
            print(f"  {file}: {count}")

        print(f"\nTotal imports: {result.total_matches}")

    except Exception as e:
        print(f"Error: {e}")


def example_ls_basic() -> None:
    """Example: Basic directory listing."""
    print("\n=== Example: Basic LS ===")

    current_dir = Path.cwd().absolute()

    result = ls_directory(LSInput(path=str(current_dir)))

    print(f"Contents of {result.directory_path}:")
    print(f"Total entries: {result.total_count}\n")

    for entry in result.entries[:20]:  # Show first 20
        entry_type = "DIR " if entry.is_directory else "FILE"
        size_str = f"{entry.size:>10}" if entry.size else "         -"
        print(f"  [{entry_type}] {size_str} bytes  {entry.name}")


def example_ls_with_ignore() -> None:
    """Example: Directory listing with ignore patterns."""
    print("\n=== Example: LS with Ignore Patterns ===")

    current_dir = Path.cwd().absolute()

    result = ls_directory(
        LSInput(path=str(current_dir), ignore=["*.pyc", "__pycache__", ".git", "node_modules", ".venv"])
    )

    print("Contents (excluding common artifacts):")
    for entry in result.entries[:15]:
        entry_type = "📁" if entry.is_directory else "📄"
        print(f"  {entry_type} {entry.name}")


# ============================================================================
# PydanticAI Agent Integration Examples
# ============================================================================


def example_agent_with_filesystem_tools() -> None:
    """Example: PydanticAI agent with filesystem tools."""
    print("\n=== Example: PydanticAI Agent Integration ===")

    # Create an agent with filesystem tools using the new @tool decorators
    _agent = Agent(
        "openai:gpt-4",
        system_prompt=(
            "You are a helpful code assistant with access to filesystem tools. "
            "You can search for files, search within files, and list directories. "
            "Use the tools to help users navigate and understand their codebase."
        ),
    )

    # Note: Agent.include_tools() is not available in the current PydanticAI version
    # Tools should be registered directly using @agent.tool decorator or passed to Agent() constructor
    # This is示例代码 demonstrating the concept
    # agent.include_tools(glob_tool)  # type: ignore[attr-defined]
    # agent.include_tools(grep_tool)  # type: ignore[attr-defined]
    # agent.include_tools(ls_tool)  # type: ignore[attr-defined]

    print("Agent created with filesystem tools!")
    print("Available tools: glob_tool, grep_tool, ls_tool")
    print("\nExample queries you could ask:")
    print("  - 'Find all Python files in the tools directory'")
    print("  - 'Search for TODO comments in the src folder'")
    print("  - 'List the contents of the current directory'")
    print("  - 'Find all files with TODO comments and show me the content'")
    print("  - 'What JavaScript files are in my project and how big are they?'")

    # Demonstrate how to use the agent (commented out since it requires API key)
    print("\n" + "=" * 50)
    print("To use the agent, uncomment the code below and add your API key:")
    print("=" * 50)
    print("""
# Example usage:
# async def main():
#     result = await agent.run("Find all Python files in the tools directory")
#     print(result.data)
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
    """)


def example_modern_agent_integration() -> None:
    """Example: Modern PydanticAI agent integration with structured responses."""
    print("\n=== Example: Modern Agent Integration ===")

    # Create a more specialized agent
    _code_explorer_agent = Agent(
        "openai:gpt-4",
        system_prompt=(
            "You are a code explorer assistant. Your job is to help users understand "
            "their codebase by finding files, searching for patterns, and providing "
            "structured information about project structure. Always provide clear, "
            "organized responses with file counts and relevant details."
        ),
    )

    # Note: include_tools is not available in current PydanticAI version
    # In production, tools should be registered using @agent.tool decorator
    # code_explorer_agent.include_tools(glob_tool)  # type: ignore[attr-defined]
    # code_explorer_agent.include_tools(grep_tool)  # type: ignore[attr-defined]
    # code_explorer_agent.include_tools(ls_tool)  # type: ignore[attr-defined]

    print("Code Explorer Agent created!")
    print("This agent can help you:")
    print("  • Analyze project structure")
    print("  • Find specific patterns or code")
    print("  • Navigate directories efficiently")
    print("  • Search across multiple file types")

    # Example tool usage patterns
    print("\n" + "=" * 50)
    print("Example workflow patterns:")
    print("=" * 50)
    print("""
1. Project Overview:
   - Use ls_tool to see the main directory structure
   - Use glob_tool to find all source files by type
   - Use grep_tool to find important patterns (imports, classes, etc.)

2. Code Search:
   - Use grep_tool with 'files_with_matches' to locate relevant files
   - Use grep_tool with 'content' mode to see actual matches
   - Use glob_tool to find related files in the same directory

3. File Discovery:
   - Use glob_tool with '**/*.ext' for recursive file searches
   - Use multiple patterns with '*.{py,js,ts}' for multi-language projects
   - Use ignore patterns with ls_tool to exclude build artifacts
    """)


def example_direct_tool_usage() -> None:
    """Example: Using the tools directly without an agent."""
    print("\n=== Example: Direct Tool Usage ===")

    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage, ModelRequest
    from pydantic_ai.models import KnownModelName, Model

    # Create a mock context for direct tool usage
    class MockContext:
        def __init__(self) -> None:
            # Note: Model is abstract and cannot be instantiated directly
            # In real usage, tools are called within agent context
            self.messages: list[ModelMessage] = []

    ctx = MockContext()  # type: ignore

    print("Testing tools directly...")

    # Test glob_tool
    try:
        result = glob_tool(ctx, "*.py", max_results=5)  # type: ignore
        print("\n1. Glob Tool Result:")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"\n1. Glob Tool Error: {e}")

    # Test ls_tool
    try:
        current_dir = str(Path.cwd())
        result = ls_tool(ctx, current_dir, ignore=["*.pyc", "__pycache__"])  # type: ignore[arg-type]
        print("\n2. LS Tool Result:")
        print(result[:300] + "..." if len(result) > 300 else result)
    except Exception as e:
        print(f"\n2. LS Tool Error: {e}")

    # Test grep_tool (if ripgrep is available)
    try:
        result = grep_tool(ctx, "import", glob="*.py", output_mode="count", head_limit=5)  # type: ignore[arg-type]
        print("\n3. Grep Tool Result:")
        print(result[:300] + "..." if len(result) > 300 else result)
    except Exception as e:
        print(f"\n3. Grep Tool Error: {e}")
        print("   (Note: This is expected if ripgrep is not installed)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Filesystem Tools Examples")
    print("=" * 70)

    # Standalone examples
    example_glob_basic()
    example_glob_recursive()
    example_glob_multiple_extensions()

    example_grep_find_todos()
    example_grep_with_context()
    example_grep_count_matches()

    example_ls_basic()
    example_ls_with_ignore()

    # Agent integration examples
    example_agent_with_filesystem_tools()
    example_modern_agent_integration()
    example_direct_tool_usage()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
