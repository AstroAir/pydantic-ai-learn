"""
Code Agent - Autonomous Code Analysis and Manipulation Agent (Example Wrapper)

A fully-functional code agent built on PydanticAI that provides comprehensive
code analysis, refactoring, and generation capabilities.

This is an example wrapper showing how to create a custom CodeAgent.
For production use, import directly from the code_agent package.

Features:
- Autonomous code analysis and understanding
- Pattern detection and code smell identification
- Refactoring suggestions and code generation
- Integration with task planning and file editing tools
- Comprehensive error handling and validation

Example Usage:
    ```python
    from code_agent import CodeAgent

    # Create agent
    agent = CodeAgent()

    # Analyze code
    result = agent.run_sync("Analyze the code in tools/task_planning_toolkit.py")
    print(result.output)

    # Suggest refactoring
    result = agent.run_sync("Suggest refactoring for complex functions in main.py")
    print(result.output)
    ```

Run with: python examples/code_agent/code_agent_wrapper.py

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic_ai import Agent, RunContext

# Import code agent toolkit
from tools.code_agent_toolkit import (
    AnalyzeCodeInput,
    CodeAgentState,
    DetectPatternsInput,
    GenerateCodeInput,
    SuggestRefactoringInput,
    ValidateSyntaxInput,
    analyze_code,
    calculate_metrics,
    detect_patterns,
    find_dependencies,
    generate_code,
    suggest_refactoring,
    validate_syntax,
)

# Import existing toolkits for integration
try:
    from tools.task_planning_toolkit import (
        TaskListState,
        TodoItem,
        TodoWriteInput,
        todo_write,
    )

    TASK_PLANNING_AVAILABLE = True
except ImportError:
    TASK_PLANNING_AVAILABLE = False

try:
    from tools.filesystem_tools import (
        GlobInput,
        GrepInput,
        LSInput,
        glob_files,
        grep_search,
        ls_directory,
    )

    FILESYSTEM_TOOLS_AVAILABLE = True
except ImportError:
    FILESYSTEM_TOOLS_AVAILABLE = False

try:
    from tools.file_editing_toolkit import (
        EditInput,
        FileEditState,
        WriteInput,
        edit_file,
        write_file,
    )

    FILE_EDITING_AVAILABLE = True
except ImportError:
    FILE_EDITING_AVAILABLE = False


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert code analysis and refactoring agent with deep \
knowledge of Python best practices, design patterns, and software engineering principles.

Your capabilities include:
- Analyzing code structure, complexity, and quality metrics
- Detecting code smells, anti-patterns, and potential issues
- Suggesting refactoring opportunities and improvements
- Generating well-structured, documented Python code
- Understanding and explaining code dependencies
- Validating Python syntax and style

When analyzing code:
1. Always start by understanding the code's purpose and context
2. Look for both structural issues and potential improvements
3. Provide specific, actionable recommendations
4. Consider maintainability, readability, and performance
5. Follow Python best practices (PEP 8, type hints, docstrings)

When suggesting refactoring:
1. Prioritize high-impact improvements
2. Explain the benefits of each suggestion
3. Consider the trade-offs and complexity
4. Provide concrete examples when helpful

When generating code:
1. Follow the existing codebase patterns and conventions
2. Include comprehensive docstrings and type hints
3. Implement proper error handling
4. Write clean, maintainable code

You have access to task planning tools to break down complex analysis tasks.
You can read files, analyze their structure, and suggest improvements.
Always be thorough, precise, and helpful in your analysis."""


# ============================================================================
# CodeAgent Class
# ============================================================================


class CodeAgent:
    """
    Autonomous code analysis and manipulation agent.

    Integrates code analysis tools with PydanticAI for intelligent,
    context-aware code operations.

    Attributes:
        agent: PydanticAI agent instance
        state: Code agent state tracker
    """

    def __init__(self, model: str = "openai:gpt-4"):
        """
        Initialize the code agent.

        Args:
            model: Model to use for the agent (default: openai:gpt-4)
        """
        self.state = CodeAgentState()

        # Create PydanticAI agent
        self.agent = Agent(
            model,
            deps_type=CodeAgentState,
            system_prompt=SYSTEM_PROMPT,
        )

        # Register code analysis tools
        self._register_code_tools()

        # Register integration tools if available
        if TASK_PLANNING_AVAILABLE:
            self._register_task_planning_tools()
        if FILESYSTEM_TOOLS_AVAILABLE:
            self._register_filesystem_tools()
        if FILE_EDITING_AVAILABLE:
            self._register_file_editing_tools()

    def _register_code_tools(self) -> None:
        """Register code analysis and manipulation tools."""

        @self.agent.tool
        def analyze_python_code(
            ctx: RunContext[CodeAgentState],
            file_path: str,
            analysis_type: str = "full",
            include_metrics: bool = True,
            include_patterns: bool = True,
        ) -> str:
            """
            Analyze Python code structure, metrics, and patterns.

            Args:
                file_path: Path to Python file to analyze
                analysis_type: Type of analysis (structure, metrics, patterns, dependencies, full)
                include_metrics: Include code quality metrics
                include_patterns: Detect code smells and patterns
            """
            return analyze_code(
                AnalyzeCodeInput(
                    file_path=file_path,
                    analysis_type=analysis_type,  # type: ignore
                    include_metrics=include_metrics,
                    include_patterns=include_patterns,
                ),
                ctx.deps,
            )

        @self.agent.tool
        def validate_python_syntax(ctx: RunContext[CodeAgentState], file_path: str, strict: bool = False) -> str:
            """
            Validate Python file syntax.

            Args:
                file_path: Path to Python file to validate
                strict: Use strict validation mode
            """
            return validate_syntax(ValidateSyntaxInput(file_path=file_path, strict=strict), ctx.deps)

        @self.agent.tool
        def detect_code_patterns(
            ctx: RunContext[CodeAgentState], file_path: str, severity_threshold: str = "low"
        ) -> str:
            """
            Detect code patterns, smells, and anti-patterns.

            Args:
                file_path: Path to Python file to analyze
                severity_threshold: Minimum severity level (low, medium, high)
            """
            return detect_patterns(
                DetectPatternsInput(
                    file_path=file_path,
                    severity_threshold=severity_threshold,  # type: ignore
                ),
                ctx.deps,
            )

        @self.agent.tool
        def get_code_metrics(ctx: RunContext[CodeAgentState], file_path: str) -> str:
            """
            Calculate comprehensive code quality metrics.

            Args:
                file_path: Path to Python file
            """
            return calculate_metrics(file_path, ctx.deps)

        @self.agent.tool
        def analyze_dependencies(ctx: RunContext[CodeAgentState], file_path: str) -> str:
            """
            Find and analyze code dependencies.

            Args:
                file_path: Path to Python file
            """
            return find_dependencies(file_path, ctx.deps)

        @self.agent.tool
        def get_refactoring_suggestions(
            ctx: RunContext[CodeAgentState], file_path: str, include_examples: bool = True
        ) -> str:
            """
            Suggest refactoring opportunities for code improvement.

            Args:
                file_path: Path to Python file to analyze
                include_examples: Include code examples in suggestions
            """
            return suggest_refactoring(
                SuggestRefactoringInput(file_path=file_path, include_examples=include_examples), ctx.deps
            )

        @self.agent.tool
        def generate_python_code(
            ctx: RunContext[CodeAgentState],
            description: str,
            code_type: str = "function",
            include_docstrings: bool = True,
            include_type_hints: bool = True,
        ) -> str:
            """
            Generate Python code from description.

            Args:
                description: Detailed description of code to generate
                code_type: Type of code (function, class, module, snippet)
                include_docstrings: Include docstrings
                include_type_hints: Include type hints
            """
            return generate_code(
                GenerateCodeInput(
                    description=description,
                    code_type=code_type,  # type: ignore
                    include_docstrings=include_docstrings,
                    include_type_hints=include_type_hints,
                ),
                ctx.deps,
            )

    def _register_task_planning_tools(self) -> None:
        """Register task planning tools."""
        if not TASK_PLANNING_AVAILABLE:
            return

        @self.agent.tool
        def manage_task_list(
            ctx: RunContext[CodeAgentState], task_content: str, task_status: str = "pending", task_id: str = "task_1"
        ) -> str:
            """
            Create and manage a structured task list.

            Args:
                task_content: Task description
                task_status: Task status (pending, in_progress, completed)
                task_id: Unique task identifier
            """
            if ctx.deps.task_state is None:
                ctx.deps.task_state = TaskListState()

            todo_item = TodoItem(
                content=task_content,
                status=task_status,  # type: ignore
                id=task_id,
            )
            return todo_write(TodoWriteInput(todos=[todo_item]), ctx.deps.task_state)

    def _register_filesystem_tools(self) -> None:
        """Register filesystem tools."""
        if not FILESYSTEM_TOOLS_AVAILABLE:
            return

        @self.agent.tool
        def search_files(ctx: RunContext[CodeAgentState], pattern: str, path: str = ".", max_results: int = 50) -> str:
            """
            Search for files matching a glob pattern.

            Args:
                pattern: Glob pattern (e.g., "*.py", "**/*.js")
                path: Directory path to search
                max_results: Maximum results to return
            """
            result = glob_files(GlobInput(pattern=pattern, path=path, max_results=max_results))
            return str(result)

        @self.agent.tool
        def search_in_files(
            ctx: RunContext[CodeAgentState],
            pattern: str,
            path: str = ".",
            ignore_case: bool = False,
            output_mode: str = "files_with_matches",
        ) -> str:
            """
            Search for text pattern in files using ripgrep.

            Args:
                pattern: Search pattern
                path: Directory path to search
                ignore_case: Case-insensitive search
                output_mode: Output mode (files_with_matches, content, count)
            """
            return grep_search(
                GrepInput(pattern=pattern, path=path, ignore_case=ignore_case, output_mode=output_mode)  # type: ignore
            )

        @self.agent.tool
        def list_directory(ctx: RunContext[CodeAgentState], path: str, ignore: list[str] | None = None) -> str:
            """
            List directory contents.

            Args:
                path: Directory path
                ignore: Patterns to ignore
            """
            result = ls_directory(LSInput(path=path, ignore=ignore))
            return str(result)

    def _register_file_editing_tools(self) -> None:
        """Register file editing tools."""
        if not FILE_EDITING_AVAILABLE:
            return

        @self.agent.tool
        def edit_code_file(
            ctx: RunContext[CodeAgentState],
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
        ) -> str:
            """
            Edit a file by replacing text.

            Args:
                file_path: Path to file
                old_string: Text to replace
                new_string: Replacement text
                replace_all: Replace all occurrences
            """
            if ctx.deps.edit_state is None:
                ctx.deps.edit_state = FileEditState()

            return edit_file(
                EditInput(file_path=file_path, old_string=old_string, new_string=new_string, replace_all=replace_all),
                ctx.deps.edit_state,
            )

        @self.agent.tool
        def create_file(ctx: RunContext[CodeAgentState], file_path: str, content: str) -> str:
            """
            Create a new file.

            Args:
                file_path: Path for new file
                content: File content
            """
            if ctx.deps.edit_state is None:
                ctx.deps.edit_state = FileEditState()

            return write_file(WriteInput(file_path=file_path, content=content), ctx.deps.edit_state)

    def run_sync(self, prompt: str) -> Any:
        """
        Run the agent synchronously.

        Args:
            prompt: User prompt

        Returns:
            Agent run result
        """
        return self.agent.run_sync(prompt, deps=self.state)

    async def run(self, prompt: str) -> Any:
        """
        Run the agent asynchronously.

        Args:
            prompt: User prompt

        Returns:
            Agent run result
        """
        return await self.agent.run(prompt, deps=self.state)


# ============================================================================
# Convenience Functions
# ============================================================================


def create_code_agent(model: str = "openai:gpt-4") -> CodeAgent:
    """
    Create a code agent instance.

    Args:
        model: Model to use for the agent

    Returns:
        CodeAgent instance
    """
    return CodeAgent(model=model)


__all__ = ["CodeAgent", "create_code_agent"]
