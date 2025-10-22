"""
Core Type Definitions

Defines all core types and data structures used throughout the code agent.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


@dataclass
class AnalysisResult:
    """Result of code analysis operation."""

    code: str
    """The analyzed code"""

    complexity: Literal["low", "medium", "high"]
    """Complexity level"""

    issues: list[str] = field(default_factory=list)
    """Identified issues"""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Analysis metrics"""

    suggestions: list[str] = field(default_factory=list)
    """Improvement suggestions"""


@dataclass
class RefactoringResult:
    """Result of refactoring operation."""

    original_code: str
    """Original code"""

    refactored_code: str
    """Refactored code"""

    changes: list[str] = field(default_factory=list)
    """List of changes made"""

    rationale: str = ""
    """Explanation of refactoring"""


@dataclass
class CodeGenerationResult:
    """Result of code generation operation."""

    generated_code: str
    """Generated code"""

    description: str = ""
    """Description of generated code"""

    dependencies: list[str] = field(default_factory=list)
    """Required dependencies"""


@dataclass
class AgentState:
    """State of the code agent."""

    message_history: list[Any] = field(default_factory=list)
    """Conversation history"""

    total_usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0})
    """Token usage tracking"""

    error_history: list[dict[str, Any]] = field(default_factory=list)
    """Error history"""

    analysis_cache: dict[str, Any] = field(default_factory=dict)
    """Cache of analysis results"""

    current_context: str = ""
    """Current code context"""

    streaming_enabled: bool = False
    """Whether streaming is enabled"""


# ============================================================================
# Code Execution Types
# ============================================================================


class ExecutionMode(str, Enum):
    """Code execution modes with different security levels."""

    SAFE = "safe"  # Read-only, no side effects
    RESTRICTED = "restricted"  # Limited file/network access
    FULL = "full"  # Full execution capabilities


class ExecutionStatus(str, Enum):
    """Execution status states."""

    PENDING = "pending"
    VALIDATING = "validating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of code execution operation."""

    code: str
    """The executed code"""

    status: ExecutionStatus
    """Execution status"""

    output: str = ""
    """Standard output"""

    error: str = ""
    """Standard error"""

    exit_code: int = 0
    """Exit code (0 = success)"""

    execution_time: float = 0.0
    """Execution time in seconds"""

    stdout: str = ""
    """Captured stdout"""

    stderr: str = ""
    """Captured stderr"""

    return_value: Any = None
    """Return value from execution"""

    side_effects: list[str] = field(default_factory=list)
    """Detected side effects (file writes, network calls, etc.)"""

    validation_errors: list[str] = field(default_factory=list)
    """Pre-execution validation errors"""

    verification_errors: list[str] = field(default_factory=list)
    """Post-execution verification errors"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return (
            self.status == ExecutionStatus.COMPLETED
            and self.exit_code == 0
            and not self.validation_errors
            and not self.verification_errors
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "execution_time": self.execution_time,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_value": self.return_value,
            "side_effects": self.side_effects,
            "validation_errors": self.validation_errors,
            "verification_errors": self.verification_errors,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionContext:
    """Context for code execution."""

    mode: ExecutionMode = ExecutionMode.SAFE
    """Execution mode"""

    timeout: float = 30.0
    """Timeout in seconds"""

    max_memory: int | None = None
    """Maximum memory in bytes"""

    allowed_imports: list[str] | None = None
    """Allowed import modules (None = all allowed)"""

    blocked_imports: list[str] = field(
        default_factory=lambda: ["os", "subprocess", "sys", "socket", "urllib", "requests"]
    )
    """Blocked import modules"""

    allowed_builtins: list[str] | None = None
    """Allowed builtin functions (None = all allowed)"""

    blocked_builtins: list[str] = field(default_factory=lambda: ["eval", "exec", "compile", "__import__", "open"])
    """Blocked builtin functions"""

    working_directory: str | None = None
    """Working directory for execution"""

    environment_vars: dict[str, str] = field(default_factory=dict)
    """Environment variables"""

    capture_output: bool = True
    """Capture stdout/stderr"""

    dry_run: bool = False
    """Dry run mode (validate only, don't execute)"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""
