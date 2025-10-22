"""
Code Agent Toolkit - Enhanced Version

Comprehensive code analysis and manipulation tools with advanced features:
- Streaming support for real-time feedback
- Retry logic for robust tool execution
- Usage tracking and limits
- Conversation history management

This is an enhanced version of the original toolkit with PydanticAI advanced patterns.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

# Import execution modules (use TYPE_CHECKING to avoid circular import)
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from ..adapters.graph import (
    GraphConfig,
    GraphPersistenceAdapter,
    GraphState,
)
from ..adapters.workflow import (
    FixStrategy,
    WorkflowOrchestrator,
    WorkflowState,
)

# Import new modules
from ..config.logging import LogFormat, LogLevel, StructuredLogger, create_logger
from ..utils.errors import (
    CircuitBreaker,
    CircuitBreakerError,
    ErrorCategory,
    ErrorContext,
    ErrorDiagnosisEngine,
    ErrorSeverity,
    RetryStrategy,
)

if TYPE_CHECKING:
    from ..adapters.context import (
        ContextConfig,
        ContextManager,
        ImportanceLevel,
        PruningStrategy,
        create_context_manager,
    )
    from ..config.execution import (
        ExecutionConfig,
        HookConfig,
        OutputConfig,
        ResourceLimits,
        SecurityConfig,
        ValidationConfig,
        VerificationConfig,
    )

from ..core.types import (
    ExecutionContext,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
)
from .executor import CodeExecutor, ExecutionError
from .validators import (
    ExecutionValidator,
    ExecutionVerifier,
    SecurityValidationError,
    ValidationResult,
)

# Import the original toolkit - use absolute import from tools package
# This avoids circular imports and sys.path manipulation
try:
    from tools.code_agent_toolkit import (
        CODE_SMELL_PATTERNS,
        COMPLEXITY_HIGH,
        COMPLEXITY_LOW,
        COMPLEXITY_MEDIUM,
        # Constants
        MAX_FILE_SIZE,
        # Models
        AnalyzeCodeInput,
        # Exceptions
        CodeAgentError,
        CodeAnalysisError,
        CodeAnalysisResult,
        CodeGenerationError,
        CodeSmellType,
        DetectPatternsInput,
        FileSizeExceededError,
        GenerateCodeInput,
        PatternDetectionError,
        RefactoringError,
        SuggestRefactoringInput,
        SyntaxValidationError,
        ValidateSyntaxInput,
    )
    from tools.code_agent_toolkit import (
        # State
        CodeAgentState as BaseCodeAgentState,
    )
    from tools.code_agent_toolkit import (
        # Core functions
        analyze_code as _analyze_code,
    )
    from tools.code_agent_toolkit import (
        calculate_metrics as _calculate_metrics,
    )
    from tools.code_agent_toolkit import (
        detect_patterns as _detect_patterns,
    )
    from tools.code_agent_toolkit import (
        find_dependencies as _find_dependencies,
    )
except ImportError:
    # Fallback: try with sys.path manipulation if direct import fails
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tools.code_agent_toolkit import (
        CODE_SMELL_PATTERNS,
        COMPLEXITY_HIGH,
        COMPLEXITY_LOW,
        COMPLEXITY_MEDIUM,
        MAX_FILE_SIZE,
        AnalyzeCodeInput,
        CodeAgentError,
        CodeAnalysisError,
        CodeAnalysisResult,
        CodeGenerationError,
        CodeSmellType,
        DetectPatternsInput,
        FileSizeExceededError,
        GenerateCodeInput,
        PatternDetectionError,
        RefactoringError,
        SuggestRefactoringInput,
        SyntaxValidationError,
        ValidateSyntaxInput,
    )
    from tools.code_agent_toolkit import (
        CodeAgentState as BaseCodeAgentState,
    )
    from tools.code_agent_toolkit import (
        analyze_code as _analyze_code,
    )
    from tools.code_agent_toolkit import (
        calculate_metrics as _calculate_metrics,
    )
    from tools.code_agent_toolkit import (
        detect_patterns as _detect_patterns,
    )
    from tools.code_agent_toolkit import (
        find_dependencies as _find_dependencies,
    )
from tools.code_agent_toolkit import (
    generate_code as _generate_code,
)
from tools.code_agent_toolkit import (
    suggest_refactoring as _suggest_refactoring,
)
from tools.code_agent_toolkit import (
    validate_syntax as _validate_syntax,
)

# ============================================================================
# Enhanced State with History and Usage Tracking
# ============================================================================


@dataclass
class CodeAgentState(BaseCodeAgentState):
    """
    Enhanced code agent state with conversation history and usage tracking.

    Extends the base state with:
    - Message history for multi-turn conversations
    - Usage tracking for token limits
    - Streaming event handlers
    - Error handling and recovery
    - Workflow orchestration
    - Logging infrastructure
    - Context management for intelligent pruning/summarization

    Attributes:
        analysis_cache: Cache of analysis results
        task_state: Task planning state (if available)
        edit_state: File editing state (if available)
        current_context: Current code context
        message_history: Conversation message history
        total_usage: Total token usage across all requests
        streaming_enabled: Whether streaming is enabled
        logger: Structured logger instance
        error_history: History of errors encountered
        circuit_breakers: Circuit breakers for operations
        workflow_orchestrator: Workflow orchestration manager
        retry_strategy: Retry strategy configuration
        context_manager: Context management for intelligent pruning
    """

    message_history: list[Any] = field(default_factory=list)
    total_usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0})
    streaming_enabled: bool = field(default=False)

    # New enhanced features
    logger: StructuredLogger = field(default_factory=lambda: create_logger("code_agent", LogLevel.INFO))
    error_history: list[ErrorContext] = field(default_factory=list)
    circuit_breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    workflow_orchestrator: WorkflowOrchestrator | None = field(default=None)
    retry_strategy: RetryStrategy = field(default_factory=RetryStrategy)
    context_manager: ContextManager | None = field(default=None)

    # Graph integration
    graph_state: GraphState | None = field(default=None)
    graph_persistence_adapter: GraphPersistenceAdapter | None = field(default=None)

    # Execution tracking
    execution_history: list[ExecutionResult] = field(default_factory=list)

    def add_message(self, message: Any) -> None:
        """Add a message to conversation history."""
        self.message_history.append(message)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.message_history.clear()

    def update_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update usage statistics."""
        self.total_usage["input_tokens"] += input_tokens
        self.total_usage["output_tokens"] += output_tokens
        self.total_usage["requests"] += 1

    def get_usage_summary(self) -> str:
        """Get formatted usage summary."""
        return (
            f"Total Usage:\n"
            f"  Input Tokens: {self.total_usage['input_tokens']}\n"
            f"  Output Tokens: {self.total_usage['output_tokens']}\n"
            f"  Requests: {self.total_usage['requests']}"
        )

    def add_error(self, error_context: ErrorContext) -> None:
        """Add error to history."""
        self.error_history.append(error_context)
        self.logger.error(
            f"Error recorded: {error_context.error_type}",
            category=error_context.category.value,
            severity=error_context.severity.value,
        )

    def get_or_create_circuit_breaker(
        self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Get or create circuit breaker for an operation."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name, failure_threshold=failure_threshold, recovery_timeout=recovery_timeout
            )
        return self.circuit_breakers[name]

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of errors encountered."""
        if not self.error_history:
            return {"total_errors": 0}

        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}

        for error in self.error_history:
            cat = error.category.value
            sev = error.severity.value
            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_errors": [
                {
                    "type": e.error_type,
                    "message": e.error_message,
                    "category": e.category.value,
                }
                for e in self.error_history[-5:]
            ],
        }

    def get_context_statistics(self) -> dict[str, Any]:
        """Get context management statistics."""
        if self.context_manager is None:
            return {"context_management": "disabled"}

        return self.context_manager.get_statistics()

    def get_context_health(self) -> str:
        """Get context health status."""
        if self.context_manager is None:
            return "disabled"

        return self.context_manager.get_health_status()

    def get_graph_statistics(self) -> dict[str, Any]:
        """Get graph execution statistics."""
        if self.graph_state is None:
            return {"graph_integration": "disabled"}

        return self.graph_state.get_statistics()

    def get_graph_health(self) -> str:
        """Get graph health status."""
        if self.graph_state is None:
            return "disabled"

        return self.graph_state.get_health_status()


# ============================================================================
# Enhanced Tool Wrappers with Retry Support
# ============================================================================


def analyze_code_with_retry(input_params: AnalyzeCodeInput, state: CodeAgentState, max_retries: int = 2) -> str:
    """
    Analyze code with enhanced retry, circuit breaker, and error handling.

    Args:
        input_params: Analysis input parameters
        state: Enhanced code agent state
        max_retries: Maximum number of retries

    Returns:
        Analysis result

    Raises:
        CodeAnalysisError: If analysis fails after retries
        CircuitBreakerError: If circuit breaker is open
    """
    operation_name = "analyze_code"
    circuit_breaker = state.get_or_create_circuit_breaker(operation_name)
    metrics = state.logger.start_operation(operation_name, file_path=input_params.file_path)

    for attempt in range(max_retries + 1):
        try:
            # Execute through circuit breaker
            result: str = str(circuit_breaker.call(_analyze_code, input_params, state))
            state.logger.complete_operation(metrics, success=True)
            return result

        except CircuitBreakerError:
            state.logger.complete_operation(metrics, success=False, error="Circuit breaker open")
            raise

        except CodeAnalysisError as e:
            # Create error context
            error_context = ErrorContext.from_exception(
                e,
                category=ErrorCategory.TRANSIENT,
                severity=ErrorSeverity.MEDIUM,
                input_parameters={"file_path": input_params.file_path},
            )
            error_context.retry_count = attempt

            # Diagnose error
            ErrorDiagnosisEngine.diagnose(error_context)
            state.add_error(error_context)

            if attempt == max_retries:
                state.logger.complete_operation(metrics, success=False, error=str(e))
                raise

            # Calculate backoff delay
            delay = state.retry_strategy.calculate_delay(attempt)
            state.logger.warning(
                f"Analysis failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.2f}s",
                error=str(e),
                suggestions=error_context.recovery_suggestions,
            )

            import time

            time.sleep(delay)
            continue

    raise CodeAnalysisError("Analysis failed after all retries")


def validate_syntax_with_retry(input_params: ValidateSyntaxInput, state: CodeAgentState, max_retries: int = 2) -> str:
    """Validate syntax with automatic retry on failure."""
    for attempt in range(max_retries + 1):
        try:
            return _validate_syntax(input_params, state)
        except SyntaxValidationError as e:
            if attempt == max_retries:
                raise
            print(f"Validation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            continue

    raise SyntaxValidationError("Validation failed after all retries")


def detect_patterns_with_retry(input_params: DetectPatternsInput, state: CodeAgentState, max_retries: int = 2) -> str:
    """Detect patterns with automatic retry on failure."""
    for attempt in range(max_retries + 1):
        try:
            return _detect_patterns(input_params, state)
        except PatternDetectionError as e:
            if attempt == max_retries:
                raise
            print(f"Pattern detection failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            continue

    raise PatternDetectionError("Pattern detection failed after all retries")


# ============================================================================
# Streaming Support
# ============================================================================


class StreamingAnalysisEvent(BaseModel):
    """Event emitted during streaming analysis."""

    event_type: Literal["start", "progress", "complete", "error"]
    message: str
    data: dict[str, Any] | None = None

    model_config = {"extra": "forbid"}


async def analyze_code_streaming(
    input_params: AnalyzeCodeInput,
    state: CodeAgentState,
    event_callback: Callable[[StreamingAnalysisEvent], Awaitable[None]] | None = None,
) -> str:
    """
    Analyze code with streaming progress updates.

    Args:
        input_params: Analysis input parameters
        state: Enhanced code agent state
        event_callback: Optional callback for streaming events

    Returns:
        Analysis result
    """
    if event_callback:
        await event_callback(
            StreamingAnalysisEvent(event_type="start", message=f"Starting analysis of {input_params.file_path}")
        )

    try:
        # Perform analysis
        result = _analyze_code(input_params, state)

        if event_callback:
            await event_callback(
                StreamingAnalysisEvent(
                    event_type="complete", message="Analysis complete", data={"result_length": len(result)}
                )
            )

        return result

    except Exception as e:
        if event_callback:
            await event_callback(StreamingAnalysisEvent(event_type="error", message=f"Analysis failed: {e}"))
        raise


# ============================================================================
# Code Execution Functions
# ============================================================================


def execute_code(
    code: str, state: CodeAgentState, config: ExecutionConfig | None = None, context: ExecutionContext | None = None
) -> str:
    """
    Execute Python code with validation and monitoring.

    Args:
        code: Python code to execute
        state: Code agent state
        config: Execution configuration
        context: Execution context

    Returns:
        Formatted execution result

    Raises:
        ExecutionError: If execution fails
    """
    # Import at runtime to avoid circular import
    if config is None:
        from ..config.execution import create_safe_config

        config = create_safe_config()

    # Create executor
    executor = CodeExecutor(config)

    # Execute code
    result = executor.execute(code, context)

    # Store in state
    if not hasattr(state, "execution_history"):
        state.execution_history = []
    state.execution_history.append(result)

    # Format result
    if result.is_success():
        output = "✓ Execution successful\n\n"
        output += f"Output:\n{result.output}\n"
        if result.execution_time:
            output += f"\nExecution time: {result.execution_time:.3f}s"
        return output
    output = "✗ Execution failed\n\n"
    if result.validation_errors:
        output += "Validation errors:\n"
        for error in result.validation_errors:
            output += f"  - {error}\n"
    if result.error:
        output += f"\nError: {result.error}\n"
    if result.verification_errors:
        output += "\nVerification errors:\n"
        for error in result.verification_errors:
            output += f"  - {error}\n"
    return output


def validate_code_for_execution(code: str, state: CodeAgentState, config: ValidationConfig | None = None) -> str:
    """
    Validate code before execution.

    Args:
        code: Python code to validate
        state: Code agent state
        config: Validation configuration

    Returns:
        Validation result message
    """
    validator = ExecutionValidator(config)
    result = validator.validate(code)

    if result.is_valid:
        output = "✓ Code validation passed\n"
        if result.warnings:
            output += "\nWarnings:\n"
            for warning in result.warnings:
                output += f"  - {warning}\n"
        return output
    output = "✗ Code validation failed\n\n"
    output += "Errors:\n"
    for error in result.errors:
        output += f"  - {error}\n"
    if result.warnings:
        output += "\nWarnings:\n"
        for warning in result.warnings:
            output += f"  - {warning}\n"
    return output


def execute_code_with_retry(
    code: str, state: CodeAgentState, config: ExecutionConfig | None = None, max_retries: int = 2
) -> str:
    """
    Execute code with automatic retry on transient failures.

    Args:
        code: Python code to execute
        state: Code agent state
        config: Execution configuration
        max_retries: Maximum retry attempts

    Returns:
        Formatted execution result
    """
    operation_name = "execute_code"
    circuit_breaker = state.get_or_create_circuit_breaker(operation_name)
    metrics = state.logger.start_operation(operation_name, code_length=len(code))

    for attempt in range(max_retries + 1):
        try:
            # Execute through circuit breaker
            result: str = str(circuit_breaker.call(execute_code, code, state, config))
            state.logger.complete_operation(metrics, success=True)
            return result

        except CircuitBreakerError:
            state.logger.complete_operation(metrics, success=False, error="Circuit breaker open")
            raise

        except ExecutionError as e:
            if attempt == max_retries:
                state.logger.complete_operation(metrics, success=False, error=str(e))
                raise

            # Log retry
            state.logger.warning(
                f"Execution failed (attempt {attempt + 1}/{max_retries + 1}): {e}",
                operation=operation_name,
                attempt=attempt + 1,
            )
            continue

    raise ExecutionError("Execution failed after all retries")


# ============================================================================
# Runtime imports for exports (to avoid circular import at module level)
# ============================================================================


# These are imported here for re-export, but not at module level to avoid circular imports
def __getattr__(name: str) -> Any:
    """Lazy import for execution config and context classes to avoid circular imports."""
    # Context management imports (lazy to avoid circular import)
    if name in (
        "ContextManager",
        "ContextConfig",
        "PruningStrategy",
        "ImportanceLevel",
        "create_context_manager",
    ):
        from ..adapters.context import (
            ContextConfig as _ContextConfig,
        )
        from ..adapters.context import (
            ContextManager as _ContextManager,
        )
        from ..adapters.context import (
            ImportanceLevel as _ImportanceLevel,
        )
        from ..adapters.context import (
            PruningStrategy as _PruningStrategy,
        )
        from ..adapters.context import (
            create_context_manager as _create_context_manager,
        )

        _context_exports = {
            "ContextManager": _ContextManager,
            "ContextConfig": _ContextConfig,
            "PruningStrategy": _PruningStrategy,
            "ImportanceLevel": _ImportanceLevel,
            "create_context_manager": _create_context_manager,
        }
        return _context_exports[name]

    # Execution config imports (lazy to avoid circular import)
    if name in (
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
    ):
        from ..config.execution import (
            ExecutionConfig as _ExecutionConfig,
        )
        from ..config.execution import (
            HookConfig as _HookConfig,
        )
        from ..config.execution import (
            OutputConfig as _OutputConfig,
        )
        from ..config.execution import (
            ResourceLimits as _ResourceLimits,
        )
        from ..config.execution import (
            SecurityConfig as _SecurityConfig,
        )
        from ..config.execution import (
            ValidationConfig as _ValidationConfig,
        )
        from ..config.execution import (
            VerificationConfig as _VerificationConfig,
        )
        from ..config.execution import (
            create_full_config as _create_full_config,
        )
        from ..config.execution import (
            create_restricted_config as _create_restricted_config,
        )
        from ..config.execution import (
            create_safe_config as _create_safe_config,
        )

        _execution_exports = {
            "ExecutionConfig": _ExecutionConfig,
            "ValidationConfig": _ValidationConfig,
            "VerificationConfig": _VerificationConfig,
            "SecurityConfig": _SecurityConfig,
            "ResourceLimits": _ResourceLimits,
            "OutputConfig": _OutputConfig,
            "HookConfig": _HookConfig,
            "create_safe_config": _create_safe_config,
            "create_restricted_config": _create_restricted_config,
            "create_full_config": _create_full_config,
        }
        return _execution_exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enhanced State
    "CodeAgentState",
    # Original exports
    "CodeAgentError",
    "CodeAnalysisError",
    "SyntaxValidationError",
    "PatternDetectionError",
    "CodeGenerationError",
    "RefactoringError",
    "FileSizeExceededError",
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
    # Original functions (re-exported)
    "_analyze_code",
    "_validate_syntax",
    "_detect_patterns",
    "_calculate_metrics",
    "_find_dependencies",
    "_suggest_refactoring",
    "_generate_code",
    # Constants
    "MAX_FILE_SIZE",
    "COMPLEXITY_LOW",
    "COMPLEXITY_MEDIUM",
    "COMPLEXITY_HIGH",
    "CODE_SMELL_PATTERNS",
    # New modules (re-exported for convenience)
    "StructuredLogger",
    "LogLevel",
    "LogFormat",
    "create_logger",
    "ErrorContext",
    "ErrorCategory",
    "ErrorSeverity",
    "CircuitBreaker",
    "CircuitBreakerError",
    "RetryStrategy",
    "ErrorDiagnosisEngine",
    "WorkflowOrchestrator",
    "WorkflowState",
    "FixStrategy",
    "ContextManager",
    "ContextConfig",
    "PruningStrategy",
    "ImportanceLevel",
    "create_context_manager",
    # Graph integration
    "GraphState",
    "GraphConfig",
    "GraphPersistenceAdapter",
    # Code execution
    "execute_code",
    "validate_code_for_execution",
    "execute_code_with_retry",
    "ExecutionConfig",
    "ExecutionContext",
    "ExecutionMode",
    "ExecutionResult",
    "ExecutionStatus",
    "CodeExecutor",
    "ExecutionValidator",
    "ExecutionVerifier",
    "ValidationResult",
    "ExecutionError",
    "SecurityValidationError",
    "ValidationConfig",
    "VerificationConfig",
    "SecurityConfig",
    "ResourceLimits",
    "OutputConfig",
    "HookConfig",
]
