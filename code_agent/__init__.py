"""
Code Agent Package - Enhanced Autonomous Code Analysis with Auto-Debugging

A sophisticated code analysis and manipulation package built on PydanticAI
with advanced auto-debugging capabilities:

- **Automatic Error Recovery**: Circuit breakers, intelligent retry with exponential backoff
- **Workflow Automation**: Multi-step debugging with validation and rollback
- **Structured Logging**: JSON-formatted logs with performance metrics
- **Error Diagnosis**: Automatic categorization and recovery suggestions
- **Streaming Support**: Real-time analysis feedback
- **Context Management**: Multi-turn conversations with history
- **Usage Tracking**: Token limits and comprehensive statistics

Quick Start:
    ```python
    from code_agent import CodeAgent

    # Create agent with auto-debugging
    agent = CodeAgent()

    # Analyze code with automatic error recovery
    result = agent.run_sync("Analyze tools/task_planning_toolkit.py")
    print(result.output)

    # Stream analysis with real-time progress
    async for text in agent.run_stream("Analyze my_module.py"):
        print(text, end="", flush=True)

    # Multi-turn conversation with context
    result1 = agent.run_sync("Analyze main.py")
    result2 = agent.run_sync(
        "What are the top 3 issues?",
        message_history=result1.new_messages()
    )

    # Check error history and metrics
    print(agent.state.get_error_summary())
    print(agent.state.logger.get_metrics_summary())
    ```

Advanced Features:
    ```python
    from code_agent import CodeAgent, UsageLimits, LogLevel, LogFormat

    # Create agent with full configuration
    agent = CodeAgent(
        model="openai:gpt-4",
        usage_limits=UsageLimits(response_tokens_limit=1000),
        enable_streaming=True,
        log_level=LogLevel.DEBUG,
        log_format=LogFormat.JSON,
        enable_workflow=True
    )

    # Get comprehensive usage and error summary
    result = agent.run_sync("Analyze code.py")
    print(agent.get_usage_summary())
    print(agent.state.get_error_summary())

    # Iterate over execution nodes
    async for node in agent.iter_nodes("Analyze module.py"):
        print(f"Node: {type(node).__name__}")
    ```

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .adapters.context import (
    ContextConfig,
    ContextManager,
    ImportanceLevel,
    PruningStrategy,
    create_context_manager,
)
from .adapters.graph import (
    GraphConfig,
    GraphPersistenceAdapter,
    GraphState,
)
from .adapters.workflow import (
    FixStrategy,
    WorkflowOrchestrator,
    WorkflowState,
)
from .config.execution import (
    ExecutionConfig,
    HookConfig,
    OutputConfig,
    ResourceLimits,
    SecurityConfig,
    ValidationConfig,
    VerificationConfig,
    create_full_config,
    create_restricted_config,
    create_safe_config,
)

# Import from new modular structure
from .config.logging import (
    LogFormat,
    LogLevel,
    StructuredLogger,
    create_logger,
)
from .core.agent import (
    SYSTEM_PROMPT,
    CodeAgent,
    create_code_agent,
)

# Import execution types and config
from .core.types import (
    ExecutionContext,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
)
from .tools.executor import (
    CodeExecutor,
    ExecutionCache,
    ExecutionError,
)
from .tools.toolkit import (
    CODE_SMELL_PATTERNS,
    COMPLEXITY_HIGH,
    COMPLEXITY_LOW,
    COMPLEXITY_MEDIUM,
    MAX_FILE_SIZE,
    AnalyzeCodeInput,
    CodeAgentError,
    CodeAgentState,
    CodeAnalysisError,
    CodeAnalysisResult,
    CodeGenerationError,
    CodeSmellType,
    DetectPatternsInput,
    FileSizeExceededError,
    GenerateCodeInput,
    PatternDetectionError,
    RefactoringError,
    StreamingAnalysisEvent,
    SuggestRefactoringInput,
    SyntaxValidationError,
    ValidateSyntaxInput,
    analyze_code_streaming,
    analyze_code_with_retry,
    detect_patterns_with_retry,
    # Code execution
    execute_code,
    execute_code_with_retry,
    validate_code_for_execution,
    validate_syntax_with_retry,
)
from .tools.validators import (
    ExecutionValidator,
    ExecutionVerifier,
    SecurityValidationError,
    ValidationError,
    ValidationResult,
)
from .utils.errors import (
    CircuitBreaker,
    CircuitBreakerError,
    ErrorCategory,
    ErrorContext,
    ErrorDiagnosisEngine,
    ErrorSeverity,
    RetryStrategy,
)

# Import PydanticAI utilities for convenience
if TYPE_CHECKING:
    from pydantic_ai import ModelRetry, UsageLimitExceeded, UsageLimits

    PYDANTIC_AI_AVAILABLE = True
else:
    try:
        from pydantic_ai import ModelRetry, UsageLimitExceeded, UsageLimits

        PYDANTIC_AI_AVAILABLE = True
    except ImportError:
        PYDANTIC_AI_AVAILABLE = False
        UsageLimits = None  # type: ignore
        UsageLimitExceeded = None  # type: ignore
        ModelRetry = None  # type: ignore


# ============================================================================
# Package Metadata
# ============================================================================

__version__ = "2.0.0"
__author__ = "The Augster"
__description__ = "Enhanced autonomous code analysis agent with PydanticAI"


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main classes
    "CodeAgent",
    "create_code_agent",
    # State
    "CodeAgentState",
    # Exceptions
    "CodeAgentError",
    "CodeAnalysisError",
    "SyntaxValidationError",
    "PatternDetectionError",
    "CodeGenerationError",
    "RefactoringError",
    "FileSizeExceededError",
    # Models
    "AnalyzeCodeInput",
    "ValidateSyntaxInput",
    "DetectPatternsInput",
    "SuggestRefactoringInput",
    "GenerateCodeInput",
    "CodeAnalysisResult",
    "CodeSmellType",
    # Enhanced functions
    "analyze_code_with_retry",
    "validate_syntax_with_retry",
    "detect_patterns_with_retry",
    "analyze_code_streaming",
    "StreamingAnalysisEvent",
    # Constants
    "MAX_FILE_SIZE",
    "COMPLEXITY_LOW",
    "COMPLEXITY_MEDIUM",
    "COMPLEXITY_HIGH",
    "CODE_SMELL_PATTERNS",
    # Logging
    "StructuredLogger",
    "LogLevel",
    "LogFormat",
    "create_logger",
    # Error Handling
    "ErrorContext",
    "ErrorCategory",
    "ErrorSeverity",
    "CircuitBreaker",
    "CircuitBreakerError",
    "RetryStrategy",
    "ErrorDiagnosisEngine",
    # Workflow
    "WorkflowOrchestrator",
    "WorkflowState",
    "FixStrategy",
    # Context Management
    "ContextManager",
    "ContextConfig",
    "PruningStrategy",
    "ImportanceLevel",
    "create_context_manager",
    # Graph Integration
    "GraphState",
    "GraphConfig",
    "GraphPersistenceAdapter",
    # Code Execution
    "execute_code",
    "validate_code_for_execution",
    "execute_code_with_retry",
    "ExecutionContext",
    "ExecutionMode",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionConfig",
    "ValidationConfig",
    "VerificationConfig",
    "SecurityConfig",
    "ResourceLimits",
    "OutputConfig",
    "HookConfig",
    "create_safe_config",
    "create_restricted_config",
    "create_full_config",
    "CodeExecutor",
    "ExecutionCache",
    "ExecutionError",
    "ExecutionValidator",
    "ExecutionVerifier",
    "ValidationResult",
    "ValidationError",
    "SecurityValidationError",
    # System prompt
    "SYSTEM_PROMPT",
    # PydanticAI utilities
    "UsageLimits",
    "UsageLimitExceeded",
    "ModelRetry",
    # Metadata
    "__version__",
    "__author__",
    "__description__",
]


# ============================================================================
# Package-level convenience functions
# ============================================================================


def quick_analyze(file_path: str, model: str = "openai:gpt-4") -> Any:
    """
    Quick code analysis without creating an agent instance.

    Args:
        file_path: Path to Python file to analyze
        model: Model to use

    Returns:
        Analysis result

    Example:
        ```python
        from code_agent import quick_analyze

        result = quick_analyze("my_module.py")
        print(result)
        ```
    """
    agent = CodeAgent(model=model)
    result = agent.run_sync(f"Analyze the code in {file_path}")
    return result.output


def quick_refactor(file_path: str, model: str = "openai:gpt-4") -> Any:
    """
    Quick refactoring suggestions without creating an agent instance.

    Args:
        file_path: Path to Python file to analyze
        model: Model to use

    Returns:
        Refactoring suggestions

    Example:
        ```python
        from code_agent import quick_refactor

        suggestions = quick_refactor("legacy_code.py")
        print(suggestions)
        ```
    """
    agent = CodeAgent(model=model)
    result = agent.run_sync(f"Suggest refactoring for {file_path}")
    return result.output


# Terminal UI imports (moved to ui/ module)
# Config module
from .config import (
    CommandValidationConfig,
    ConfigManager,
    FilesystemAccessConfig,
    MCPConfig,
    MCPConfigLoader,
    MCPTransportType,
    ResourceLimitConfig,
    TerminalSecurityConfig,
    create_development_terminal_config,
    create_safe_terminal_config,
    load_config,
)
from .core import (
    A2AClient,
    A2AConfig,
    # A2A integration
    A2AServer,
    AgentConfig,
    AgentRegistry,
    AgentState,
    AnalysisResult,
    CodeGenerationResult,
    CodeSubAgent,
    DelegatedTask,
    # Hierarchical agent system
    HierarchicalAgent,
    HierarchicalAgentConfig,
    RefactoringResult,
    SubAgent,
    SubAgentInfo,
    SubAgentResult,
    SubAgentStatus,
    TaskDelegator,
    TaskStatus,
)

# ============================================================================
# New Refactored Modules (v2.1+)
# ============================================================================
# Core module
from .core import (
    CodeAgent as CoreCodeAgent,
)
from .core import (
    create_code_agent as create_core_agent,
)

# Tools module
from .tools import (
    CodeAnalyzer,
    CodeGenerator,
    CommandValidator,
    RateLimiter,
    RealTimeTerminalSession,
    RefactoringEngine,
    SessionInfo,
    SessionState,
    TerminalSandbox,
    TerminalSessionManager,
)
from .ui import (
    CodeAgentTerminal,
    TerminalUIComponents,
    TerminalUIState,
    launch_terminal,
    launch_terminal_async,
)

# Custom tools (optional - may not be available)
try:
    from .tools.custom import (
        CodeFormatter,
        CodeLinter,
        DependencyAnalyzer,
        DocumentationAnalyzer,
    )
    from .workflows import (
        QualityReport,
        QualityWorkflow,
    )

    _CUSTOM_FEATURES_AVAILABLE = True
except ImportError:
    _CUSTOM_FEATURES_AVAILABLE = False

# Utils module (additional exports)
from .utils import (
    PerformanceMetrics,
)

# Adapters module - already imported above in the main imports section
# These are now available as ContextManager, WorkflowOrchestrator, GraphConfig

# Add convenience functions to exports
__all__.extend(
    [
        "quick_analyze",
        "quick_refactor",
        # Terminal UI
        "CodeAgentTerminal",
        "TerminalUIState",
        "TerminalUIComponents",
        "launch_terminal",
        "launch_terminal_async",
        # New refactored modules
        "CoreCodeAgent",
        "AgentConfig",
        "create_core_agent",
        "AgentState",
        "AnalysisResult",
        "RefactoringResult",
        "CodeGenerationResult",
        "ConfigManager",
        "load_config",
        "MCPConfigLoader",
        "MCPConfig",
        "MCPTransportType",
        "CodeAnalyzer",
        "RefactoringEngine",
        "CodeGenerator",
        "PerformanceMetrics",
        # Adapters (migrated from legacy)
        "ContextManager",
        "WorkflowOrchestrator",
        "GraphConfig",
        "GraphState",
        "GraphPersistenceAdapter",
        # Hierarchical agent system
        "HierarchicalAgent",
        "HierarchicalAgentConfig",
        "SubAgent",
        "CodeSubAgent",
        "SubAgentInfo",
        "SubAgentStatus",
        "DelegatedTask",
        "SubAgentResult",
        "TaskStatus",
        "AgentRegistry",
        "TaskDelegator",
        # A2A integration
        "A2AServer",
        "A2AClient",
        "A2AConfig",
        # Terminal sandbox and session
        "TerminalSandbox",
        "CommandValidator",
        "RateLimiter",
        "RealTimeTerminalSession",
        "TerminalSessionManager",
        "SessionState",
        "SessionInfo",
        # Terminal security configuration
        "TerminalSecurityConfig",
        "CommandValidationConfig",
        "ResourceLimitConfig",
        "FilesystemAccessConfig",
        "create_safe_terminal_config",
        "create_development_terminal_config",
    ]
)

# Add custom features to exports if available
if _CUSTOM_FEATURES_AVAILABLE:
    __all__.extend(
        [
            # Custom tools
            "CodeFormatter",
            "CodeLinter",
            "DependencyAnalyzer",
            "DocumentationAnalyzer",
            # Workflows
            "QualityWorkflow",
            "QualityReport",
        ]
    )
