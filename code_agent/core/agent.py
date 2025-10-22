"""
Code Agent - Enhanced Autonomous Code Analysis Agent with Auto-Debugging

A sophisticated code agent with advanced capabilities:
- **Automatic Error Detection & Recovery**: Circuit breakers, intelligent retry strategies
- **Workflow Automation**: Multi-step debugging workflows with validation and rollback
- **Structured Logging**: JSON-formatted logs with performance metrics
- **Error Diagnosis**: Automatic error categorization and recovery suggestions
- **Streaming Support**: Real-time feedback during analysis
- **Context Management**: Multi-turn conversations with history
- **Usage Tracking**: Token limits and comprehensive usage statistics

Core Features:
- Autonomous code analysis and understanding
- Pattern detection and code smell identification
- Refactoring suggestions and code generation
- Automatic debugging with fix strategies
- Circuit breaker pattern for failing operations
- Exponential backoff retry with jitter
- State checkpointing and recovery
- Graceful degradation on errors

Example Usage:
    ```python
    from code_agent import CodeAgent
    from pydantic_ai import UsageLimits

    # Create agent with logging and error handling
    agent = CodeAgent(
        model="openai:gpt-4",
        enable_streaming=True,
        usage_limits=UsageLimits(request_limit=100)
    )

    # Analyze code with automatic retry and error recovery
    result = agent.run_sync("Analyze the code in tools/task_planning_toolkit.py")
    print(result.output)

    # Stream analysis with real-time progress
    async for event in agent.run_stream("Analyze my_module.py"):
        print(event)

    # Multi-turn conversation with context
    result1 = agent.run_sync("Analyze main.py")
    result2 = agent.run_sync("What are the top 3 issues?", message_history=result1.new_messages())

    # Check error history and metrics
    print(agent.state.get_error_summary())
    print(agent.state.logger.get_metrics_summary())
    ```

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import inspect

# Import existing toolkits for integration
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, cast

AgentLogLevelLiteral = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
AgentLogFormatLiteral = Literal["json", "human"]

try:
    from pydantic_ai import (
        Agent,
        AgentStreamEvent,
        ModelRetry,
        RunContext,
        UsageLimitExceeded,
        UsageLimits,
    )
except Exception:  # pragma: no cover - optional during static checks without dependency
    from typing import Generic, TypeVar

    StateT = TypeVar("StateT")
    F = TypeVar("F", bound=Callable[..., Any])

    class _EmptyAsyncIterator(AsyncIterator[Any]):
        def __aiter__(self) -> _EmptyAsyncIterator:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    class Agent:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def tool(self, *_args: Any, **_kwargs: Any) -> Callable[[F], F]:
            def decorator(func: F) -> F:
                return func

            return decorator

        def run_sync(self, *_args: Any, **_kwargs: Any) -> Any:
            return None

        async def run(self, *_args: Any, **_kwargs: Any) -> Any:
            return None

        def run_stream(self, *_args: Any, **_kwargs: Any) -> AsyncIterable[Any]:
            return _EmptyAsyncIterator()

        async def iter(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            return _EmptyAsyncIterator()

    class AgentStreamEvent:  # type: ignore[no-redef]
        pass

    class ModelRetry(Exception):  # type: ignore[no-redef]  # noqa: N818
        pass

    class RunContext(Generic[StateT]):  # type: ignore[no-redef]  # noqa: UP046
        def __init__(self, deps: StateT) -> None:
            self.deps = deps

    class UsageLimitExceeded(Exception):  # type: ignore[no-redef]  # noqa: N818
        pass

    class UsageLimits:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass


# Import enhanced toolkit with new capabilities
from ..tools.toolkit import (  # noqa: E402
    AnalyzeCodeInput,
    CodeAgentState,
    DetectPatternsInput,
    GenerateCodeInput,
    GraphConfig,
    GraphPersistenceAdapter,
    GraphState,
    LogFormat,
    # New imports
    LogLevel,
    SuggestRefactoringInput,
    ValidateSyntaxInput,
    WorkflowOrchestrator,
    _calculate_metrics,
    _find_dependencies,
    _generate_code,
    _suggest_refactoring,
    analyze_code_with_retry,
    detect_patterns_with_retry,
    validate_syntax_with_retry,
)
from .config import AgentConfig  # noqa: E402

if TYPE_CHECKING:
    from .hierarchical_agent import HierarchicalAgent
    from .model_router import ModelRouter
    from .prompt_enhancer import PromptEnhancer
    from .request_classifier import RequestClassifier
    from .routing_config import RoutingConfig

# Task planning tools are imported lazily inside _register_task_planning_tools


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert code analysis and refactoring agent with deep knowledge of Python best \
practices, design patterns, and software engineering principles.

Your capabilities include:
- Analyzing code structure, complexity, and quality metrics
- Detecting code smells, anti-patterns, and potential issues
- Suggesting refactoring opportunities and improvements
- Generating well-structured, documented Python code
- Understanding and explaining code dependencies
- Validating Python syntax and style
- Providing real-time streaming analysis feedback
- Managing multi-turn conversations with context
- **Automatic error detection and recovery with intelligent retry strategies**
- **Circuit breaker pattern to prevent cascading failures**
- **Workflow automation for multi-step debugging operations**
- **Comprehensive logging and performance metrics tracking**
- **Graph-based workflow orchestration for complex multi-step operations**
- **State machine execution with persistence and recovery**

When analyzing code:
1. Always start by understanding the code's purpose and context
2. Look for both structural issues and potential improvements
3. Provide specific, actionable recommendations
4. Consider maintainability, readability, and performance
5. Follow Python best practices (PEP 8, type hints, docstrings)
6. Use streaming for long-running analysis to provide progress updates
7. Automatically detect and diagnose errors with recovery suggestions

When suggesting refactoring:
1. Prioritize high-impact improvements
2. Explain the benefits of each suggestion
3. Consider the trade-offs and complexity
4. Provide concrete examples when helpful
5. Use retry logic with exponential backoff to ensure robust analysis
6. Leverage circuit breakers to gracefully handle failing operations

When generating code:
1. Follow the existing codebase patterns and conventions
2. Include comprehensive docstrings and type hints
3. Implement proper error handling with recovery mechanisms
4. Write clean, maintainable code with logging

When encountering errors:
1. Automatically categorize errors (transient/permanent, recoverable/fatal)
2. Apply appropriate fix strategies based on error type
3. Validate fixes before applying them
4. Rollback changes if fixes fail
5. Provide detailed diagnostic information and recovery suggestions

When using graph workflows:
1. Use graphs for complex multi-step operations that require state management
2. Leverage graph persistence for long-running or resumable workflows
3. Monitor graph execution health and statistics
4. Use graph checkpoints for recovery and debugging
5. Combine graph workflows with code analysis for sophisticated automation

You have access to task planning tools to break down complex analysis tasks.
You can read files, analyze their structure, and suggest improvements.
You can orchestrate complex workflows using graph-based state machines.
All operations are logged with performance metrics for observability.
Always be thorough, precise, and helpful in your analysis."""


# ============================================================================
# Enhanced CodeAgent Class
# ============================================================================


class CodeAgent:
    """
    Enhanced autonomous code analysis and manipulation agent with auto-debugging.

    Integrates code analysis tools with advanced PydanticAI features:
    - **Automatic Error Recovery**: Circuit breakers, intelligent retry with exponential backoff
    - **Workflow Automation**: Multi-step debugging with validation and rollback
    - **Structured Logging**: JSON-formatted logs with performance metrics
    - **Error Diagnosis**: Automatic categorization and recovery suggestions
    - **Streaming Support**: Real-time feedback during analysis
    - **Context Management**: Multi-turn conversations with history
    - **Usage Tracking**: Token limits and comprehensive statistics

    Attributes:
        agent: PydanticAI agent instance
        state: Enhanced code agent state with logging, error handling, and workflow
        default_usage_limits: Default usage limits
    """

    def __init__(
        self,
        model: str | AgentConfig = "openai:gpt-4",
        usage_limits: UsageLimits | None = None,
        enable_streaming: bool = False,
        log_level: LogLevel = LogLevel.INFO,
        log_format: LogFormat = LogFormat.HUMAN,
        enable_workflow: bool = True,
        enable_context_management: bool = True,
        max_context_tokens: int = 100_000,
        enable_graph: bool = True,
        graph_config: GraphConfig | None = None,
        # Extensions
        prepare_tools: Any | None = None,
        enable_filesystem_tools: bool = True,
        enable_file_editing_tools: bool = False,
        # Hierarchical agent support
        enable_hierarchical: bool = False,
        hierarchical_config: Any | None = None,
    ) -> None:
        """
        Initialize the enhanced code agent with auto-debugging capabilities.

        Args:
            model: Model to use for the agent (default: openai:gpt-4) or AgentConfig object
            usage_limits: Optional usage limits for token control
            enable_streaming: Enable streaming by default
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Log format (JSON or HUMAN)
            enable_workflow: Enable workflow orchestration for debugging
            enable_context_management: Enable intelligent context management
            max_context_tokens: Maximum context tokens (default: 100,000)
            enable_graph: Enable graph workflow integration
            graph_config: Optional graph configuration
        """
        # Handle AgentConfig object as first parameter
        if isinstance(model, AgentConfig):
            config = model
            model = config.model
            usage_limits = usage_limits or config.usage_limits
            enable_streaming = config.enable_streaming
            log_level = LogLevel[config.log_level] if isinstance(config.log_level, str) else config.log_level
            log_format = (
                LogFormat[config.log_format.upper()] if isinstance(config.log_format, str) else config.log_format
            )
            enable_workflow = config.enable_error_recovery  # Map to workflow
            enable_context_management = config.enable_context_management
            max_context_tokens = config.max_context_tokens
            enable_graph = config.enable_graph_integration
            # Store the config object
            self.config = config
        else:
            log_level_literal_value = (log_level.name if isinstance(log_level, LogLevel) else str(log_level)).upper()
            log_level_literal = cast(AgentLogLevelLiteral, log_level_literal_value)

            log_format_value = str(log_format.value) if isinstance(log_format, LogFormat) else str(log_format)
            log_format_literal_value = log_format_value.lower()
            log_format_literal = cast(AgentLogFormatLiteral, log_format_literal_value)

            usage_limits_value = cast(Any, usage_limits)

            # Create config from individual parameters
            self.config = AgentConfig(
                model=model,
                enable_streaming=enable_streaming,
                enable_logging=True,
                log_level=log_level_literal,
                log_format=log_format_literal,
                usage_limits=usage_limits_value,
                enable_error_recovery=enable_workflow,
                enable_context_management=enable_context_management,
                max_context_tokens=max_context_tokens,
                enable_graph_integration=enable_graph,
            )
        # Initialize state with enhanced features
        self.state = CodeAgentState(streaming_enabled=enable_streaming)
        self.default_usage_limits: UsageLimits | None = usage_limits

        # Configure logging
        from ..config.logging import create_logger

        self.state.logger = create_logger(name="code_agent", level=log_level, format_type=log_format)

        # Initialize workflow orchestrator if enabled
        if enable_workflow:
            self.state.workflow_orchestrator = WorkflowOrchestrator(operation_name="code_analysis")

        # Initialize context manager if enabled
        if enable_context_management:
            from ..adapters.context import PruningStrategy, create_context_manager

            self.state.context_manager = create_context_manager(
                max_tokens=max_context_tokens,
                model_name=model.split(":")[-1] if ":" in model else model,
                strategy=PruningStrategy.HYBRID,
                enable_summarization=True,
                logger=self.state.logger,
            )

        # Initialize graph integration if enabled
        if enable_graph:
            graph_state_config = graph_config or GraphConfig()
            self.state.graph_state = GraphState(config=graph_state_config)
            self.state.graph_persistence_adapter = GraphPersistenceAdapter(
                checkpoint_dir=graph_state_config.checkpoint_dir
            )

        self.state.logger.info(
            "CodeAgent initialized",
            model=model,
            streaming=enable_streaming,
            workflow=enable_workflow,
            context_management=enable_context_management,
            graph_integration=enable_graph,
            hierarchical=enable_hierarchical,
        )

        # Create PydanticAI agent (optionally with prepare_tools hook)
        if prepare_tools is None:
            self.agent = Agent(
                model,
                deps_type=CodeAgentState,
                system_prompt=SYSTEM_PROMPT,
            )
        else:
            self.agent = Agent(
                model,
                deps_type=CodeAgentState,
                system_prompt=SYSTEM_PROMPT,
                prepare_tools=prepare_tools,
            )

        # Register code analysis tools with retry support
        self._register_code_tools()

        # Register integration tools if available
        self._register_task_planning_tools()

        # Initialize hierarchical agent support if enabled
        self.hierarchical_agent: HierarchicalAgent | None = None
        if enable_hierarchical:
            from .hierarchical_agent import HierarchicalAgent

            self.hierarchical_agent = HierarchicalAgent(
                parent_agent=self,
                config=hierarchical_config,
            )

        # Initialize routing system if configured
        self.prompt_enhancer: PromptEnhancer | None = None
        self.request_classifier: RequestClassifier | None = None
        self.model_router: ModelRouter | None = None

        if self.config.routing_config and self.config.routing_config.enabled:
            from .model_router import ModelRouter
            from .prompt_enhancer import PromptEnhancer
            from .request_classifier import RequestClassifier

            self.prompt_enhancer = PromptEnhancer(
                config=self.config.routing_config.enhancement,
                logger=self.state.logger,
            )
            self.request_classifier = RequestClassifier(
                config=self.config.routing_config.classification,
                logger=self.state.logger,
            )
            self.model_router = ModelRouter(
                config=self.config.routing_config,
                logger=self.state.logger,
            )

            self.state.logger.info(
                "Routing system initialized",
                extra={
                    "enhancement_enabled": self.config.routing_config.enhancement.enabled,
                    "classification_enabled": self.config.routing_config.classification.enabled,
                    "available_models": self.model_router.get_available_models(),
                },
            )

        # Register graph tools if enabled
        if enable_graph:
            self._register_graph_tools()

        # Register workflow tools if enabled
        if enable_workflow:
            self._register_workflow_tools()

        # Optional external tool integrations
        if enable_filesystem_tools:
            self._register_filesystem_tools()
        if enable_file_editing_tools:
            self._register_file_editing_tools()

    def _register_code_tools(self) -> None:
        """Register code analysis and manipulation tools with retry support."""

        @self.agent.tool(retries=2)
        def analyze_python_code(
            ctx: RunContext[CodeAgentState],
            file_path: str,
            analysis_type: str = "full",
            include_metrics: bool = True,
            include_patterns: bool = True,
        ) -> str:
            """
            Analyze Python code structure, metrics, and patterns.

            Automatically retries up to 2 times on failure.

            Args:
                file_path: Path to Python file to analyze
                analysis_type: Type of analysis (structure, metrics, patterns, dependencies, full)
                include_metrics: Include code quality metrics
                include_patterns: Detect code smells and patterns
            """
            try:
                return analyze_code_with_retry(
                    AnalyzeCodeInput(
                        file_path=file_path,
                        analysis_type=analysis_type,  # type: ignore
                        include_metrics=include_metrics,
                        include_patterns=include_patterns,
                    ),
                    ctx.deps,
                    max_retries=2,
                )
            except Exception as e:
                raise ModelRetry(f"Analysis failed: {e}. Please try with a different file or parameters.") from e

        @self.agent.tool(retries=2)
        def validate_python_syntax(ctx: RunContext[CodeAgentState], file_path: str, strict: bool = False) -> str:
            """
            Validate Python file syntax.

            Automatically retries up to 2 times on failure.

            Args:
                file_path: Path to Python file to validate
                strict: Use strict validation mode
            """
            try:
                return validate_syntax_with_retry(
                    ValidateSyntaxInput(file_path=file_path, strict=strict), ctx.deps, max_retries=2
                )
            except Exception as e:
                raise ModelRetry(f"Validation failed: {e}. Please check the file path.") from e

        @self.agent.tool(retries=2)
        def detect_code_patterns(
            ctx: RunContext[CodeAgentState], file_path: str, severity_threshold: str = "low"
        ) -> str:
            """
            Detect code patterns, smells, and anti-patterns.

            Automatically retries up to 2 times on failure.

            Args:
                file_path: Path to Python file to analyze
                severity_threshold: Minimum severity level (low, medium, high)
            """
            try:
                return detect_patterns_with_retry(
                    DetectPatternsInput(
                        file_path=file_path,
                        severity_threshold=severity_threshold,  # type: ignore
                    ),
                    ctx.deps,
                    max_retries=2,
                )
            except Exception as e:
                raise ModelRetry(f"Pattern detection failed: {e}") from e

        @self.agent.tool
        def get_code_metrics(ctx: RunContext[CodeAgentState], file_path: str) -> str:
            """
            Calculate comprehensive code quality metrics.

            Args:
                file_path: Path to Python file
            """
            return _calculate_metrics(file_path, ctx.deps)

        @self.agent.tool
        def analyze_dependencies(ctx: RunContext[CodeAgentState], file_path: str) -> str:
            """
            Find and analyze code dependencies.

            Args:
                file_path: Path to Python file
            """
            return _find_dependencies(file_path, ctx.deps)

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
            return _suggest_refactoring(
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
            return _generate_code(
                GenerateCodeInput(
                    description=description,
                    code_type=code_type,  # type: ignore
                    include_docstrings=include_docstrings,
                    include_type_hints=include_type_hints,
                ),
                ctx.deps,
            )

        _ = (
            analyze_python_code,
            validate_python_syntax,
            detect_code_patterns,
            get_code_metrics,
            analyze_dependencies,
            get_refactoring_suggestions,
            generate_python_code,
        )

    def _register_task_planning_tools(self) -> None:
        """Register task planning tools."""
        try:
            from tools.task_planning_toolkit import (
                TaskListState,
                TodoItem,
                TodoWriteInput,
                todo_write,
            )
        except Exception:
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
                status=task_status,  # type: ignore[arg-type]
                id=task_id,
            )
            return todo_write(TodoWriteInput(todos=[todo_item]), ctx.deps.task_state)

        _ = manage_task_list

    def _register_graph_tools(self) -> None:
        """Register graph workflow tools."""

        @self.agent.tool
        def get_graph_statistics(ctx: RunContext[CodeAgentState]) -> str:
            """
            Get graph execution statistics and health status.

            Returns:
                JSON-formatted statistics about graph executions
            """
            import json

            if ctx.deps.graph_state is None:
                return "Graph integration is not enabled"

            stats = ctx.deps.get_graph_statistics()
            health = ctx.deps.get_graph_health()

            return json.dumps({"statistics": stats, "health_status": health}, indent=2)

        @self.agent.tool
        def list_graph_checkpoints(ctx: RunContext[CodeAgentState]) -> str:
            """
            List available graph checkpoints.

            Returns:
                List of graph checkpoint IDs
            """
            if ctx.deps.graph_persistence_adapter is None:
                return "Graph persistence is not enabled"

            checkpoints = ctx.deps.graph_persistence_adapter.list_checkpoints()

            if not checkpoints:
                return "No graph checkpoints found"

            return f"Available checkpoints: {', '.join(checkpoints)}"

        _ = (get_graph_statistics, list_graph_checkpoints)

    def _register_workflow_tools(self) -> None:
        """Expose workflow orchestration controls as tools."""

        @self.agent.tool
        def list_workflow_steps(ctx: RunContext[CodeAgentState]) -> str:
            """List registered workflow step names."""
            import json

            if ctx.deps.workflow_orchestrator is None:
                return "Workflow orchestrator is not enabled"
            steps = ctx.deps.workflow_orchestrator.list_steps()
            return json.dumps({"steps": steps}, indent=2)

        @self.agent.tool
        def run_workflow_sequence(ctx: RunContext[CodeAgentState], steps_json: str, context: str | None = None) -> str:
            """Run a sequence of workflow steps.

            steps_json: JSON array of {"name": str, "params": {}} items
            """
            import json

            if ctx.deps.workflow_orchestrator is None:
                return "Workflow orchestrator is not enabled"
            try:
                steps = json.loads(steps_json)
                results = ctx.deps.workflow_orchestrator.run(steps, context=context)
                return json.dumps({"results": results}, indent=2)
            except Exception as e:
                return f"Workflow execution failed: {e}"

        @self.agent.tool
        def get_workflow_status(ctx: RunContext[CodeAgentState]) -> str:
            """Get current workflow status as JSON."""
            import json

            if ctx.deps.workflow_orchestrator is None:
                return "Workflow orchestrator is not enabled"
            return json.dumps(ctx.deps.workflow_orchestrator.get_workflow_status(), indent=2)

        @self.agent.tool
        def create_workflow_checkpoint(ctx: RunContext[CodeAgentState], input_json: str) -> str:
            """Create a workflow checkpoint with input/output data."""
            import json

            if ctx.deps.workflow_orchestrator is None:
                return "Workflow orchestrator is not enabled"
            try:
                data = json.loads(input_json) if input_json else {}
            except Exception as e:
                return f"Invalid JSON: {e}"
            cp = ctx.deps.workflow_orchestrator.create_checkpoint(input_data=data)
            return str(cp.checkpoint_id)

        @self.agent.tool
        def rollback_workflow_checkpoint(ctx: RunContext[CodeAgentState], checkpoint_id: str) -> str:
            """Rollback workflow state to a specific checkpoint id."""
            cp = None
            if ctx.deps.workflow_orchestrator:
                cp = ctx.deps.workflow_orchestrator.rollback_to_checkpoint(checkpoint_id)
            return cp.checkpoint_id if cp else "Checkpoint not found"

        _ = (
            list_workflow_steps,
            run_workflow_sequence,
            get_workflow_status,
            create_workflow_checkpoint,
            rollback_workflow_checkpoint,
        )

    def _register_filesystem_tools(self) -> None:
        """Register filesystem tools (glob, grep, ls) as agent tools."""
        try:
            from tools.filesystem_tools import (
                DEFAULT_MAX_RESULTS,
                GlobInput,
                GrepInput,
                LSInput,
                glob_files,
                grep_search,
                ls_directory,
            )
        except Exception:
            return

        @self.agent.tool
        def fs_glob(
            _ctx: RunContext[CodeAgentState], pattern: str, path: str | None = None, max_results: int | None = None
        ) -> str:
            try:
                input_params = GlobInput(pattern=pattern, path=path, max_results=max_results or DEFAULT_MAX_RESULTS)
                res = glob_files(input_params)
                header = f"Found {res.total_count} files matching '{res.pattern}'"
                if res.search_path:
                    header += f" in '{res.search_path}'"
                body = "\n".join(f"  {p}" for p in res.files)
                tail = ""
                if res.truncated:
                    tail = f"\n(Showing {len(res.files)} of {res.total_count} results)"
                return f"{header}\n\n{body}{tail}".strip()
            except Exception as e:
                return f"glob failed: {e}"

        @self.agent.tool
        def fs_grep(
            _ctx: RunContext[CodeAgentState],
            pattern: str,
            path: str | None = None,
            glob: str | None = None,
            output_mode: str = "files_with_matches",
            line_number: bool = False,
            ignore_case: bool = False,
            head_limit: int | None = None,
        ) -> str:
            try:
                input_params = GrepInput(
                    pattern=pattern,
                    path=path,
                    glob=glob,
                    output_mode=output_mode,  # type: ignore
                    line_number=line_number,
                    ignore_case=ignore_case,
                    head_limit=head_limit,
                )
                res = grep_search(input_params)
                out = [f"Search results for pattern '{pattern}' using {res.output_mode} mode:\n"]
                if res.output_mode == "files_with_matches" and res.files:
                    out.extend(f"  {f}" for f in res.files)
                elif res.output_mode == "count" and res.counts:
                    out.extend(f"  {f}: {c}" for f, c in res.counts.items())
                elif res.output_mode == "content" and res.matches:
                    for m in res.matches:
                        prefix = f"{m.file_path}:{m.line_number}" if m.line_number else m.file_path
                        out.append(prefix)
                        out.append(f"  {m.line_content}")
                return "\n".join(out).strip()
            except Exception as e:
                return f"grep failed: {e}"

        @self.agent.tool
        def fs_ls(
            _ctx: RunContext[CodeAgentState], path: str, ignore: list[str] | None = None, max_results: int | None = None
        ) -> str:
            try:
                input_params = LSInput(path=path, ignore=ignore, max_results=max_results)
                res = ls_directory(input_params)
                lines = [f"Directory listing for '{res.directory_path}':\n"]
                dirs = [e for e in res.entries if e.is_directory]
                files = [e for e in res.entries if not e.is_directory]
                if dirs:
                    lines.append("Directories:")
                    lines.extend(f"  📁  {d.name}/" for d in dirs)
                    lines.append("")
                if files:
                    lines.append("Files:")
                    for f in files:
                        size = f"{f.size:,} bytes" if f.size is not None else "unknown size"
                        lines.append(f"  📄  {f.name} ({size})")
                if not dirs and not files:
                    lines.append("  (empty directory)")
                if res.truncated:
                    lines.append(f"\n(Showing {len(res.entries)} of {res.total_count} entries)")
                return "\n".join(lines).strip()
            except Exception as e:
                return f"ls failed: {e}"

        _ = (fs_glob, fs_grep, fs_ls)

    def _register_file_editing_tools(self) -> None:
        """Register file editing tools with strict validation."""
        try:
            from tools.file_editing_toolkit import (
                EditInput,
                FileEditState,
                MultiEditInput,
                NotebookEditInput,
                WriteInput,
                edit_file,
                multi_edit_file,
                notebook_edit,
                write_file,
            )
        except Exception:
            return

        @self.agent.tool
        def file_edit(
            ctx: RunContext[CodeAgentState], file_path: str, old_string: str, new_string: str, replace_all: bool = False
        ) -> str:
            if ctx.deps.edit_state is None:
                ctx.deps.edit_state = FileEditState()
            return edit_file(
                EditInput(
                    file_path=file_path,
                    old_string=old_string,
                    new_string=new_string,
                    replace_all=replace_all,
                ),
                ctx.deps.edit_state,
            )

        @self.agent.tool
        def file_multi_edit(ctx: RunContext[CodeAgentState], file_path: str, edits_json: str) -> str:
            import json

            if ctx.deps.edit_state is None:
                ctx.deps.edit_state = FileEditState()
            edits_list = json.loads(edits_json)
            input_params = MultiEditInput(file_path=file_path, edits=edits_list)
            return multi_edit_file(input_params, ctx.deps.edit_state)

        @self.agent.tool
        def file_write(ctx: RunContext[CodeAgentState], file_path: str, content: str) -> str:
            if ctx.deps.edit_state is None:
                ctx.deps.edit_state = FileEditState()
            return write_file(WriteInput(file_path=file_path, content=content), ctx.deps.edit_state)

        @self.agent.tool
        def notebook_cell_edit(
            ctx: RunContext[CodeAgentState],
            notebook_path: str,
            cell_id: str | None,
            new_source: str,
            cell_type: str | None = None,
            edit_mode: str = "replace",
        ) -> str:
            if ctx.deps.edit_state is None:
                ctx.deps.edit_state = FileEditState()
            input_params = NotebookEditInput(
                notebook_path=notebook_path,
                cell_id=cell_id,
                new_source=new_source,
                cell_type=cell_type,  # type: ignore
                edit_mode=edit_mode,  # type: ignore
            )
            return notebook_edit(input_params, ctx.deps.edit_state)

        _ = (file_edit, file_multi_edit, file_write, notebook_cell_edit)

    # ------------------------------------------------------------------
    # Public helpers for programmatic workflow usage
    # ------------------------------------------------------------------
    def register_workflow_step(self, name: str, func: Any) -> None:
        if self.state.workflow_orchestrator is None:
            self.state.workflow_orchestrator = WorkflowOrchestrator(operation_name="code_analysis")
        self.state.workflow_orchestrator.register_step(name, func)

    def run_workflow(self, steps: list[dict[str, Any]], *, context: Any | None = None) -> list[dict[str, Any]]:
        if self.state.workflow_orchestrator is None:
            self.state.workflow_orchestrator = WorkflowOrchestrator(operation_name="code_analysis")
        return self.state.workflow_orchestrator.run(steps, context=context)

    # ------------------------------------------------------------------
    # Routing System Pre-processing
    # ------------------------------------------------------------------
    def _preprocess_prompt(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """
        Pre-process prompt with routing system.

        Args:
            prompt: Original user prompt

        Returns:
            Tuple of (processed_prompt, routing_metadata)
        """
        routing_metadata: dict[str, Any] = {}

        # If routing is not enabled, return original prompt
        if not self.config.routing_config or not self.config.routing_config.enabled:
            return prompt, routing_metadata

        processed_prompt = prompt

        # Step 1: Enhance prompt if enabled
        if self.prompt_enhancer and self.config.routing_config.enhancement.enabled:
            enhancement = self.prompt_enhancer.enhance(prompt)
            if enhancement.confidence >= self.config.routing_config.enhancement.min_confidence:
                processed_prompt = enhancement.enhanced_prompt
                routing_metadata["enhancement"] = enhancement.to_dict()

                if self.state.logger:
                    self.state.logger.info(
                        "Prompt enhanced",
                        extra={
                            "strategy": enhancement.strategy.value,
                            "confidence": enhancement.confidence,
                            "improvements": enhancement.improvements,
                        },
                    )

        # Step 2: Classify request if enabled
        if self.request_classifier and self.config.routing_config.classification.enabled:
            classification = self.request_classifier.classify(processed_prompt)
            routing_metadata["classification"] = classification.to_dict()

            if self.state.logger:
                self.state.logger.info(
                    "Request classified",
                    extra={
                        "difficulty": classification.difficulty.value,
                        "mode": classification.mode.value,
                        "confidence": classification.confidence,
                    },
                )

            # Step 3: Route to appropriate model if enabled
            if self.model_router:
                routing_decision = self.model_router.route(
                    classification=classification,
                    default_model=self.config.model,
                )
                routing_metadata["routing"] = routing_decision.to_dict()

                # Update agent model if routing suggests a different model
                if routing_decision.selected_model != self.config.model:
                    if not self.config.routing_config.dry_run:
                        # In a real implementation, we would switch the model here
                        # For now, we just log the decision
                        if self.state.logger:
                            self.state.logger.info(
                                "Model routing decision",
                                extra={
                                    "original_model": self.config.model,
                                    "selected_model": routing_decision.selected_model,
                                    "reasoning": routing_decision.reasoning,
                                    "dry_run": self.config.routing_config.dry_run,
                                },
                            )
                    else:
                        if self.state.logger:
                            self.state.logger.info(
                                "Model routing decision (dry run)",
                                extra={
                                    "would_route_to": routing_decision.selected_model,
                                    "reasoning": routing_decision.reasoning,
                                },
                            )

        return processed_prompt, routing_metadata

    def run_sync(
        self, prompt: str, message_history: list[Any] | None = None, usage_limits: UsageLimits | None = None
    ) -> Any:
        """
        Run the agent synchronously.

        Args:
            prompt: User prompt
            message_history: Optional message history for multi-turn conversations
            usage_limits: Optional usage limits (overrides default)

        Returns:
            Agent run result

        Raises:
            UsageLimitExceeded: If usage limits are exceeded
        """
        limits = usage_limits or self.default_usage_limits

        # Pre-process prompt with routing system
        processed_prompt, routing_metadata = self._preprocess_prompt(prompt)

        try:
            result = self.agent.run_sync(
                processed_prompt, deps=self.state, message_history=message_history, usage_limits=limits
            )

            # Add routing metadata to result if available
            if routing_metadata and hasattr(result, "data") and not hasattr(result.data, "_routing_metadata"):
                result.data._routing_metadata = routing_metadata

            # Update usage tracking
            if hasattr(result, "usage"):
                usage = result.usage()
                self.state.update_usage(usage.input_tokens, usage.output_tokens)

            # Add to message history
            if hasattr(result, "new_messages"):
                for msg in result.new_messages():
                    self.state.add_message(msg)

            return result

        except UsageLimitExceeded as e:
            print(f"Usage limit exceeded: {e}")
            print(self.state.get_usage_summary())
            raise

    async def run(
        self, prompt: str, message_history: list[Any] | None = None, usage_limits: UsageLimits | None = None
    ) -> Any:
        """
        Run the agent asynchronously.

        Args:
            prompt: User prompt
            message_history: Optional message history for multi-turn conversations
            usage_limits: Optional usage limits (overrides default)

        Returns:
            Agent run result

        Raises:
            UsageLimitExceeded: If usage limits are exceeded
        """
        limits = usage_limits or self.default_usage_limits

        # Pre-process prompt with routing system
        processed_prompt, routing_metadata = self._preprocess_prompt(prompt)

        try:
            result = await self.agent.run(
                processed_prompt, deps=self.state, message_history=message_history, usage_limits=limits
            )

            # Add routing metadata to result if available
            if routing_metadata and hasattr(result, "data") and not hasattr(result.data, "_routing_metadata"):
                result.data._routing_metadata = routing_metadata

            # Update usage tracking
            if hasattr(result, "usage"):
                usage = result.usage()
                self.state.update_usage(usage.input_tokens, usage.output_tokens)

            # Add to message history
            if hasattr(result, "new_messages"):
                for msg in result.new_messages():
                    self.state.add_message(msg)

            return result

        except UsageLimitExceeded as e:
            print(f"Usage limit exceeded: {e}")
            print(self.state.get_usage_summary())
            raise

    async def run_stream(
        self,
        prompt: str,
        message_history: list[Any] | None = None,
        usage_limits: UsageLimits | None = None,
        event_handler: Callable[[AgentStreamEvent], Awaitable[None]] | None = None,
    ) -> AsyncIterable[str]:
        """
        Run the agent with streaming output.

        Args:
            prompt: User prompt
            message_history: Optional message history
            usage_limits: Optional usage limits
            event_handler: Optional event handler for stream events

        Yields:
            Streamed text output

        Raises:
            UsageLimitExceeded: If usage limits are exceeded
        """
        limits = usage_limits or self.default_usage_limits

        async def default_event_handler(
            _ctx: RunContext[CodeAgentState], event_stream: AsyncIterable[AgentStreamEvent]
        ) -> None:
            """Default event handler for streaming."""
            async for event in event_stream:
                if event_handler:
                    await event_handler(event)

        context_manager = getattr(self.agent, "run_stream", None)
        if context_manager is None:
            raise AttributeError("Underlying agent does not support streaming")

        run_context = context_manager(
            prompt,
            deps=self.state,
            message_history=message_history,
            usage_limits=limits,
            event_stream_handler=default_event_handler if event_handler else None,
        )

        try:
            async with run_context as run:
                async for text in run.stream_text():
                    yield text

                # Update usage after streaming completes
                if hasattr(run, "usage"):
                    usage = run.usage()
                    self.state.update_usage(usage.input_tokens, usage.output_tokens)

        except UsageLimitExceeded as e:
            print(f"Usage limit exceeded: {e}")
            print(self.state.get_usage_summary())
            raise

    async def iter_nodes(self, prompt: str) -> AsyncIterable[Any]:
        """
        Iterate over agent execution nodes.

        Provides access to intermediate execution steps.

        Args:
            prompt: User prompt

        Yields:
            Execution nodes
        """
        iterator: Any = self.agent.iter(prompt, deps=self.state)

        if inspect.isawaitable(iterator):
            iterator = await iterator

        if hasattr(iterator, "__aenter__") and hasattr(iterator, "__aexit__"):
            async with iterator as agent_run:
                async for node in agent_run:
                    yield node
            return

        if hasattr(iterator, "__aiter__"):
            async for node in iterator:
                yield node
            return

        raise TypeError("Underlying agent.iter did not return an async iterator or context manager")

    def get_usage_summary(self) -> str:
        """Get formatted usage summary."""
        return self.state.get_usage_summary()

    def get_error_summary(self) -> dict[str, Any]:
        """Return the structured error summary tracked by the agent state."""
        return self.state.get_error_summary()

    def get_state(self) -> dict[str, Any]:
        """Snapshot the current agent state for observability and debugging."""
        return {
            "config": self.config.to_dict(),
            "message_history": self.state.message_history.copy(),
            "total_usage": self.state.total_usage.copy(),
            "streaming_enabled": self.state.streaming_enabled,
            "errors": [error.to_dict() for error in self.state.error_history],
            "context_statistics": self.state.get_context_statistics(),
            "graph_statistics": self.state.get_graph_statistics(),
        }

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.state.clear_history()

    def get_message_history(self) -> list[Any]:
        """Get conversation message history."""
        return self.state.message_history.copy()

    # Hierarchical agent methods
    def register_sub_agent(
        self,
        name: str,
        description: str,
        agent: Any,
        capabilities: list[str] | None = None,
    ) -> Any:
        """
        Register a sub-agent for hierarchical delegation.

        Args:
            name: Sub-agent name
            description: Sub-agent description
            agent: CodeAgent instance
            capabilities: Agent capabilities

        Returns:
            Created sub-agent

        Raises:
            RuntimeError: If hierarchical mode not enabled
        """
        if not self.hierarchical_agent:
            raise RuntimeError("Hierarchical mode not enabled. Initialize with enable_hierarchical=True")
        return self.hierarchical_agent.register_sub_agent(name, description, agent, capabilities)

    async def delegate(
        self,
        prompt: str,
        required_capabilities: list[str] | None = None,
        agent_id: str | None = None,
    ) -> Any:
        """
        Delegate a task to a sub-agent.

        Args:
            prompt: Task prompt
            required_capabilities: Required capabilities
            agent_id: Specific agent ID

        Returns:
            SubAgentResult

        Raises:
            RuntimeError: If hierarchical mode not enabled
        """
        if not self.hierarchical_agent:
            raise RuntimeError("Hierarchical mode not enabled. Initialize with enable_hierarchical=True")
        return await self.hierarchical_agent.delegate(prompt, required_capabilities, agent_id)

    def get_sub_agents(self) -> list[Any]:
        """
        Get all registered sub-agents.

        Returns:
            List of sub-agents

        Raises:
            RuntimeError: If hierarchical mode not enabled
        """
        if not self.hierarchical_agent:
            raise RuntimeError("Hierarchical mode not enabled. Initialize with enable_hierarchical=True")
        return self.hierarchical_agent.get_sub_agents()

    def get_hierarchical_stats(self) -> dict[str, Any]:
        """
        Get hierarchical agent statistics.

        Returns:
            Statistics dictionary

        Raises:
            RuntimeError: If hierarchical mode not enabled
        """
        if not self.hierarchical_agent:
            raise RuntimeError("Hierarchical mode not enabled. Initialize with enable_hierarchical=True")
        return self.hierarchical_agent.get_stats()

    def to_a2a(
        self,
        *,
        storage: Any | None = None,
        broker: Any | None = None,
        name: str | None = None,
        url: str | None = None,
        version: str | None = None,
        description: str | None = None,
        provider: str | None = None,
        skills: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Convert this CodeAgent to an A2A ASGI application.

        This method exposes the agent as an Agent-to-Agent (A2A) protocol
        server, allowing it to communicate with other A2A-compliant agents.

        The resulting ASGI application can be run with any ASGI server:
            uvicorn my_module:app --host 0.0.0.0 --port 8000

        Args:
            storage: Optional custom storage implementation
            broker: Optional custom broker implementation
            name: Agent name for the agent card
            url: Agent URL for the agent card
            version: Agent version for the agent card
            description: Agent description for the agent card
            provider: Agent provider for the agent card
            skills: Agent skills for the agent card
            **kwargs: Additional arguments passed to FastA2A

        Returns:
            ASGI application (FastA2A instance)

        Raises:
            ImportError: If fasta2a is not installed

        Example:
            ```python
            agent = CodeAgent(model="openai:gpt-4")
            app = agent.to_a2a(
                name="Code Analysis Agent",
                description="Specialized in code analysis and refactoring",
                skills=["python", "analysis", "refactoring"]
            )
            # Run with: uvicorn module:app
            ```
        """
        to_a2a_handler = getattr(self.agent, "to_a2a", None)
        if callable(to_a2a_handler):
            return to_a2a_handler(
                storage=storage,
                broker=broker,
                name=name or "CodeAgent",
                url=url,
                version=version or "1.0.0",
                description=description or "Code analysis and manipulation agent",
                provider=provider,
                skills=skills,
                **kwargs,
            )

        # Fallback for older pydantic-ai versions
        from .a2a_integration import A2AConfig, A2AServer

        a2a_config = A2AConfig()
        a2a_server = A2AServer(cast(Any, self.agent), a2a_config)
        return a2a_server.get_app()


# ============================================================================
# Convenience Functions
# ============================================================================


def create_code_agent(
    model: str = "openai:gpt-4",
    usage_limits: UsageLimits | None = None,
    enable_streaming: bool = False,
    log_level: LogLevel = LogLevel.INFO,
    log_format: LogFormat = LogFormat.HUMAN,
    enable_workflow: bool = True,
    enable_context_management: bool = True,
    max_context_tokens: int = 100_000,
    enable_graph: bool = True,
    graph_config: GraphConfig | None = None,
    routing_config: RoutingConfig | None = None,
) -> CodeAgent:
    """
    Create an enhanced code agent instance.

    Args:
        model: Model to use for the agent (default: openai:gpt-4)
        usage_limits: Optional usage limits for token control
        enable_streaming: Enable streaming by default
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (JSON or HUMAN)
        enable_workflow: Enable workflow orchestration for debugging
        enable_context_management: Enable intelligent context management
        max_context_tokens: Maximum context tokens (default: 100,000)
        enable_graph: Enable graph workflow integration
        graph_config: Optional graph configuration
        routing_config: Optional routing configuration for intelligent model selection

    Returns:
        Enhanced CodeAgent instance
    """
    # Create agent config with routing

    log_level_value = log_level.name if isinstance(log_level, LogLevel) else str(log_level)
    log_format_value = log_format.value if isinstance(log_format, LogFormat) else str(log_format)

    config = AgentConfig(
        model=model,
        enable_streaming=enable_streaming,
        enable_logging=True,
        log_level=cast(AgentLogLevelLiteral, log_level_value),
        log_format=cast(AgentLogFormatLiteral, log_format_value),
        usage_limits=usage_limits,
        enable_error_recovery=enable_workflow,
        enable_context_management=enable_context_management,
        max_context_tokens=max_context_tokens,
        enable_graph_integration=enable_graph,
        routing_config=routing_config,
    )

    return CodeAgent(
        model=config,
        usage_limits=usage_limits,
        enable_streaming=enable_streaming,
        log_level=log_level,
        log_format=log_format,
        enable_workflow=enable_workflow,
        enable_context_management=enable_context_management,
        max_context_tokens=max_context_tokens,
        enable_graph=enable_graph,
        graph_config=graph_config,
    )


__all__ = ["CodeAgent", "create_code_agent", "SYSTEM_PROMPT"]
