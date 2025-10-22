"""
Tools Module

Code analysis and manipulation tools.

Exports:
    - CodeAnalyzer: Main code analysis tool
    - RefactoringEngine: Refactoring suggestions
    - CodeGenerator: Code generation tool
    - Custom tools: CodeFormatter, CodeLinter, DependencyAnalyzer, DocumentationAnalyzer
    - Terminal sandbox and session management
"""

from __future__ import annotations

from .analyzer import CodeAnalyzer
from .executor import CodeExecutor, ExecutionCache, ExecutionError
from .generator import CodeGenerator
from .refactoring import RefactoringEngine
from .terminal_sandbox import (
    CommandValidator,
    RateLimiter,
    TerminalSandbox,
)
from .terminal_sandbox import (
    ValidationResult as TerminalValidationResult,
)
from .terminal_session import (
    RealTimeTerminalSession,
    SessionInfo,
    SessionState,
    TerminalSessionManager,
)
from .validators import (
    ExecutionValidator,
    ExecutionVerifier,
    SecurityValidationError,
    ValidationError,
    ValidationResult,
)

# Custom tools (optional imports - may not be available in all environments)
try:
    from .custom import (  # noqa: F401
        CodeFormatter,
        CodeLinter,
        DependencyAnalysis,
        DependencyAnalyzer,
        DocumentationAnalysis,
        DocumentationAnalyzer,
        FormattedCode,
        LintIssue,
        LintResult,
    )

    _CUSTOM_TOOLS_AVAILABLE = True
except ImportError:
    _CUSTOM_TOOLS_AVAILABLE = False

__all__ = [
    # Core tools
    "CodeAnalyzer",
    "RefactoringEngine",
    "CodeGenerator",
    "CodeExecutor",
    "ExecutionCache",
    "ExecutionError",
    "ExecutionValidator",
    "ExecutionVerifier",
    "ValidationResult",
    "ValidationError",
    "SecurityValidationError",
    # Terminal sandbox
    "TerminalSandbox",
    "CommandValidator",
    "RateLimiter",
    "TerminalValidationResult",
    # Terminal session
    "RealTimeTerminalSession",
    "TerminalSessionManager",
    "SessionState",
    "SessionInfo",
]

# Add custom tools to __all__ if available
if _CUSTOM_TOOLS_AVAILABLE:
    __all__.extend(
        [
            "CodeFormatter",
            "CodeLinter",
            "DependencyAnalyzer",
            "DocumentationAnalyzer",
            "FormattedCode",
            "LintResult",
            "LintIssue",
            "DependencyAnalysis",
            "DocumentationAnalysis",
        ]
    )
