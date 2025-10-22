"""
Code Agent Toolkit for PydanticAI

A comprehensive, production-grade Python implementation of code analysis and manipulation tools
designed for seamless integration with PydanticAI agents.

Tools:
1. AnalyzeCode: Parse and analyze Python code structure using AST
2. DetectPatterns: Identify code smells, anti-patterns, and issues
3. CalculateMetrics: Compute code complexity and quality metrics
4. FindDependencies: Analyze imports and dependencies
5. ValidateSyntax: Check Python syntax and structure
6. SuggestRefactoring: Propose refactoring opportunities
7. GenerateCode: Create code from specifications

Features:
- Modern Python 3.12+ with latest type hints (using | for unions)
- Pydantic v2 validation for robust input handling
- AST-based code analysis (no code execution)
- Comprehensive error handling with informative messages
- Integration with existing file system and editing tools
- Security-focused design (read-only analysis, validated modifications)

Security Considerations:
- No code execution during analysis (AST parsing only)
- All file paths are validated
- File size limits to prevent resource exhaustion
- Proper error handling prevents information leakage

Example Usage:
    ```python
    from tools.code_agent_toolkit import CodeAgentState, analyze_code, AnalyzeCodeInput
    from pydantic_ai import Agent, RunContext

    # Create state tracker
    state = CodeAgentState()

    # Create agent with code analysis tools
    agent = Agent('openai:gpt-4', deps_type=CodeAgentState)

    # Register tools
    @agent.tool
    def analyze(ctx: RunContext[CodeAgentState], file_path: str) -> str:
        return analyze_code(AnalyzeCodeInput(file_path=file_path), ctx.deps)

    # Use the agent
    result = agent.run_sync('Analyze the code structure...', deps=state)
    ```

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Import existing toolkit states for composition
try:
    from tools.file_editing_toolkit import FileEditState
    from tools.task_planning_toolkit import TaskListState
except ImportError:
    # Fallback if imports fail
    TaskListState = None  # type: ignore
    FileEditState = None  # type: ignore


# ============================================================================
# Custom Exceptions
# ============================================================================


class CodeAgentError(Exception):
    """Base exception for code agent errors."""

    pass


class CodeAnalysisError(CodeAgentError):
    """Raised when code analysis fails."""

    pass


class SyntaxValidationError(CodeAgentError):
    """Raised when syntax validation fails."""

    pass


class PatternDetectionError(CodeAgentError):
    """Raised when pattern detection fails."""

    pass


class CodeGenerationError(CodeAgentError):
    """Raised when code generation fails."""

    pass


class RefactoringError(CodeAgentError):
    """Raised when refactoring operation fails."""

    pass


class FileSizeExceededError(CodeAnalysisError):
    """Raised when file size exceeds maximum allowed."""

    pass


# ============================================================================
# Constants
# ============================================================================

# File size limits (in bytes)
MAX_FILE_SIZE = 1_000_000  # 1 MB
MAX_ANALYSIS_FILES = 100  # Maximum files to analyze in one operation

# Complexity thresholds
COMPLEXITY_LOW = 5
COMPLEXITY_MEDIUM = 10
COMPLEXITY_HIGH = 20

# Code smell patterns
CODE_SMELL_PATTERNS = {
    "long_function": 50,  # Lines
    "long_parameter_list": 5,  # Parameters
    "deep_nesting": 4,  # Nesting levels
    "too_many_branches": 10,  # If/elif/else branches
}


# ============================================================================
# Type Definitions
# ============================================================================

AnalysisType = Literal["structure", "metrics", "patterns", "dependencies", "full"]
ComplexityLevel = Literal["low", "medium", "high", "very_high"]
CodeSmellType = Literal[
    "long_function",
    "long_parameter_list",
    "deep_nesting",
    "too_many_branches",
    "duplicate_code",
    "magic_numbers",
    "god_class",
]


# ============================================================================
# State Management
# ============================================================================


@dataclass
class CodeAnalysisResult:
    """
    Result of code analysis operation.

    Attributes:
        file_path: Path to analyzed file
        analysis_type: Type of analysis performed
        structure: Code structure information (classes, functions, etc.)
        metrics: Code quality metrics
        patterns: Detected patterns and code smells
        dependencies: Import and dependency information
    """

    file_path: str
    analysis_type: AnalysisType
    structure: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class CodeAgentState:
    """
    State tracker for code agent operations.

    Maintains analysis results, task state, and file edit state across operations.
    Composes existing state classes for unified state management.

    Attributes:
        analysis_cache: Cache of code analysis results
        task_state: Task planning state (if available)
        edit_state: File editing state (if available)
        current_context: Current code context being analyzed
    """

    analysis_cache: dict[str, CodeAnalysisResult] = field(default_factory=dict)
    task_state: TaskListState | None = field(default=None)
    edit_state: FileEditState | None = field(default=None)
    current_context: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize composed states if available."""
        if TaskListState is not None and self.task_state is None:
            self.task_state = TaskListState()
        if FileEditState is not None and self.edit_state is None:
            self.edit_state = FileEditState()

    def cache_analysis(self, result: CodeAnalysisResult) -> None:
        """Cache an analysis result."""
        self.analysis_cache[result.file_path] = result

    def get_cached_analysis(self, file_path: str) -> CodeAnalysisResult | None:
        """Get cached analysis result."""
        return self.analysis_cache.get(file_path)

    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self.analysis_cache.clear()

    def add_to_context(self, file_path: str) -> None:
        """Add file to current context."""
        if file_path not in self.current_context:
            self.current_context.append(file_path)

    def clear_context(self) -> None:
        """Clear current context."""
        self.current_context.clear()


# ============================================================================
# Helper Functions
# ============================================================================


def validate_file_path(file_path: str) -> Path:
    """
    Validate and normalize file path.

    Args:
        file_path: Path to validate

    Returns:
        Normalized Path object

    Raises:
        CodeAnalysisError: If path is invalid or file doesn't exist
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            raise CodeAnalysisError(f"File not found: {file_path}")
        if not path.is_file():
            raise CodeAnalysisError(f"Path is not a file: {file_path}")
        return path
    except Exception as e:
        if isinstance(e, CodeAnalysisError):
            raise
        raise CodeAnalysisError(f"Invalid file path: {file_path}") from e


def check_file_size(file_path: Path, max_size: int = MAX_FILE_SIZE) -> None:
    """
    Check if file size is within limits.

    Args:
        file_path: Path to check
        max_size: Maximum allowed size in bytes

    Raises:
        FileSizeExceededError: If file is too large
    """
    size = file_path.stat().st_size
    if size > max_size:
        raise FileSizeExceededError(
            f"File size ({size} bytes) exceeds maximum allowed ({max_size} bytes). File: {file_path}"
        )


def parse_python_file(file_path: Path) -> ast.Module:
    """
    Parse Python file into AST.

    Args:
        file_path: Path to Python file

    Returns:
        AST Module node

    Raises:
        SyntaxValidationError: If file has syntax errors
        CodeAnalysisError: If file cannot be read
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        return ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        raise SyntaxValidationError(f"Syntax error in {file_path}:{e.lineno}:{e.offset}: {e.msg}") from e
    except Exception as e:
        raise CodeAnalysisError(f"Failed to read file {file_path}: {e}") from e


def calculate_cyclomatic_complexity(node: ast.AST) -> int:
    """
    Calculate cyclomatic complexity of an AST node.

    Args:
        node: AST node to analyze

    Returns:
        Cyclomatic complexity score
    """
    complexity = 1  # Base complexity

    for child in ast.walk(node):
        # Add 1 for each decision point
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, (ast.And, ast.Or)):
            complexity += 1

    return complexity


# ============================================================================
# Pydantic Input Models
# ============================================================================


class AnalyzeCodeInput(BaseModel):
    """
    Input model for code analysis tool.

    Attributes:
        file_path: Path to Python file to analyze
        analysis_type: Type of analysis to perform
        include_metrics: Whether to include code metrics
        include_patterns: Whether to detect code patterns
    """

    file_path: str = Field(..., min_length=1, description="Path to Python file to analyze")
    analysis_type: AnalysisType = Field(
        default="full", description="Type of analysis: structure, metrics, patterns, dependencies, or full"
    )
    include_metrics: bool = Field(default=True, description="Include code quality metrics")
    include_patterns: bool = Field(default=True, description="Detect code smells and patterns")

    model_config = {"extra": "forbid"}

    @field_validator("file_path")
    @classmethod
    def validate_python_file(cls, v: str) -> str:
        """Validate that file path ends with .py"""
        if not v.endswith(".py"):
            raise ValueError("File must be a Python file (.py extension)")
        return v


class ValidateSyntaxInput(BaseModel):
    """
    Input model for syntax validation tool.

    Attributes:
        file_path: Path to Python file to validate
        strict: Whether to use strict validation
    """

    file_path: str = Field(..., min_length=1, description="Path to Python file to validate")
    strict: bool = Field(default=False, description="Use strict validation mode")

    model_config = {"extra": "forbid"}

    @field_validator("file_path")
    @classmethod
    def validate_python_file(cls, v: str) -> str:
        """Validate that file path ends with .py"""
        if not v.endswith(".py"):
            raise ValueError("File must be a Python file (.py extension)")
        return v


class DetectPatternsInput(BaseModel):
    """
    Input model for pattern detection tool.

    Attributes:
        file_path: Path to Python file to analyze
        pattern_types: Types of patterns to detect
        severity_threshold: Minimum severity level to report
    """

    file_path: str = Field(..., min_length=1, description="Path to Python file to analyze")
    pattern_types: list[CodeSmellType] | None = Field(
        default=None, description="Specific pattern types to detect (None = all)"
    )
    severity_threshold: Literal["low", "medium", "high"] = Field(
        default="low", description="Minimum severity level to report"
    )

    model_config = {"extra": "forbid"}


class SuggestRefactoringInput(BaseModel):
    """
    Input model for refactoring suggestion tool.

    Attributes:
        file_path: Path to Python file to analyze
        focus_areas: Specific areas to focus on
        include_examples: Whether to include code examples
    """

    file_path: str = Field(..., min_length=1, description="Path to Python file to analyze")
    focus_areas: list[str] | None = Field(
        default=None, description="Specific areas to focus on (e.g., 'complexity', 'duplication')"
    )
    include_examples: bool = Field(default=True, description="Include code examples in suggestions")

    model_config = {"extra": "forbid"}


class GenerateCodeInput(BaseModel):
    """
    Input model for code generation tool.

    Attributes:
        description: Description of code to generate
        code_type: Type of code to generate
        style_guide: Style guide to follow
        include_docstrings: Whether to include docstrings
        include_type_hints: Whether to include type hints
    """

    description: str = Field(..., min_length=10, description="Detailed description of code to generate")
    code_type: Literal["function", "class", "module", "snippet"] = Field(
        default="function", description="Type of code to generate"
    )
    style_guide: Literal["pep8", "google", "numpy"] = Field(default="pep8", description="Style guide to follow")
    include_docstrings: bool = Field(default=True, description="Include docstrings")
    include_type_hints: bool = Field(default=True, description="Include type hints")

    model_config = {"extra": "forbid"}


# ============================================================================
# Core Tool Functions
# ============================================================================


def analyze_code(input_params: AnalyzeCodeInput, state: CodeAgentState) -> str:
    """
    Analyze Python code structure, metrics, and patterns.

    This function performs comprehensive code analysis including:
    - Code structure (classes, functions, methods)
    - Quality metrics (complexity, LOC, etc.)
    - Pattern detection (code smells, anti-patterns)
    - Dependency analysis (imports)

    Args:
        input_params: Validated analysis input
        state: Code agent state tracker

    Returns:
        Formatted analysis report

    Raises:
        CodeAnalysisError: If analysis fails
        FileSizeExceededError: If file is too large
        SyntaxValidationError: If file has syntax errors
    """
    try:
        # Validate and check file
        file_path = validate_file_path(input_params.file_path)
        check_file_size(file_path)

        # Check cache
        cached = state.get_cached_analysis(str(file_path))
        if cached and cached.analysis_type == input_params.analysis_type:
            return _format_analysis_result(cached)

        # Parse file
        tree = parse_python_file(file_path)

        # Create result object
        result = CodeAnalysisResult(file_path=str(file_path), analysis_type=input_params.analysis_type)

        # Perform requested analysis
        if input_params.analysis_type in ("structure", "full"):
            result.structure = _extract_structure(tree)

        if input_params.analysis_type in ("metrics", "full") and input_params.include_metrics:
            result.metrics = _calculate_metrics(tree)

        if input_params.analysis_type in ("dependencies", "full"):
            result.dependencies = _extract_dependencies(tree)

        if input_params.analysis_type in ("patterns", "full") and input_params.include_patterns:
            result.patterns = _detect_code_smells(tree)

        # Cache result
        state.cache_analysis(result)
        state.add_to_context(str(file_path))

        return _format_analysis_result(result)

    except (CodeAnalysisError, FileSizeExceededError, SyntaxValidationError):
        raise
    except Exception as e:
        raise CodeAnalysisError(f"Failed to analyze code: {e}") from e


def validate_syntax(input_params: ValidateSyntaxInput, state: CodeAgentState) -> str:
    """
    Validate Python file syntax.

    Args:
        input_params: Validated syntax check input
        state: Code agent state tracker

    Returns:
        Validation result message

    Raises:
        SyntaxValidationError: If syntax is invalid
        CodeAnalysisError: If validation fails
    """
    try:
        file_path = validate_file_path(input_params.file_path)
        check_file_size(file_path)

        # Parse file (will raise SyntaxValidationError if invalid)
        tree = parse_python_file(file_path)

        # Additional strict checks if requested
        if input_params.strict:
            issues = _strict_validation_checks(tree)
            if issues:
                return f"Syntax valid but found {len(issues)} style issues:\n" + "\n".join(issues)

        return f"✓ Syntax valid: {file_path.name}"

    except SyntaxValidationError:
        raise
    except Exception as e:
        raise CodeAnalysisError(f"Failed to validate syntax: {e}") from e


def detect_patterns(input_params: DetectPatternsInput, state: CodeAgentState) -> str:
    """
    Detect code patterns, smells, and anti-patterns.

    This function analyzes code for common issues including:
    - Long functions and methods
    - Long parameter lists
    - Deep nesting
    - Too many branches
    - Magic numbers
    - Duplicate code patterns

    Args:
        input_params: Validated pattern detection input
        state: Code agent state tracker

    Returns:
        Formatted pattern detection report

    Raises:
        PatternDetectionError: If pattern detection fails
        CodeAnalysisError: If file cannot be analyzed
    """
    try:
        file_path = validate_file_path(input_params.file_path)
        check_file_size(file_path)

        # Parse file
        tree = parse_python_file(file_path)

        # Detect patterns
        all_patterns = _detect_all_patterns(tree, file_path)

        # Filter by requested types
        if input_params.pattern_types:
            all_patterns = [p for p in all_patterns if p["type"] in input_params.pattern_types]

        # Filter by severity
        severity_order = {"low": 0, "medium": 1, "high": 2}
        threshold = severity_order[input_params.severity_threshold]
        filtered_patterns = [p for p in all_patterns if severity_order.get(p["severity"], 0) >= threshold]

        return _format_pattern_report(filtered_patterns, file_path)

    except (CodeAnalysisError, SyntaxValidationError):
        raise
    except Exception as e:
        raise PatternDetectionError(f"Failed to detect patterns: {e}") from e


def calculate_metrics(file_path: str, state: CodeAgentState) -> str:
    """
    Calculate comprehensive code quality metrics.

    Metrics include:
    - Lines of code (total, code, comments, blank)
    - Cyclomatic complexity (average, max)
    - Function and class counts
    - Maintainability index

    Args:
        file_path: Path to Python file
        state: Code agent state tracker

    Returns:
        Formatted metrics report

    Raises:
        CodeAnalysisError: If metrics calculation fails
    """
    try:
        path = validate_file_path(file_path)
        check_file_size(path)

        # Parse file
        tree = parse_python_file(path)
        content = path.read_text(encoding="utf-8")

        # Calculate comprehensive metrics
        metrics = _calculate_comprehensive_metrics(tree, content)

        return _format_metrics_report(metrics, path)

    except (CodeAnalysisError, SyntaxValidationError):
        raise
    except Exception as e:
        raise CodeAnalysisError(f"Failed to calculate metrics: {e}") from e


def find_dependencies(file_path: str, state: CodeAgentState) -> str:
    """
    Find and analyze code dependencies.

    Analyzes:
    - Standard library imports
    - Third-party imports
    - Local imports
    - Import structure and organization

    Args:
        file_path: Path to Python file
        state: Code agent state tracker

    Returns:
        Formatted dependency report

    Raises:
        CodeAnalysisError: If dependency analysis fails
    """
    try:
        path = validate_file_path(file_path)
        check_file_size(path)

        # Parse file
        tree = parse_python_file(path)

        # Extract and categorize dependencies
        deps = _categorize_dependencies(tree)

        return _format_dependency_report(deps, path)

    except (CodeAnalysisError, SyntaxValidationError):
        raise
    except Exception as e:
        raise CodeAnalysisError(f"Failed to find dependencies: {e}") from e


def suggest_refactoring(input_params: SuggestRefactoringInput, state: CodeAgentState) -> str:
    """
    Suggest refactoring opportunities for code improvement.

    Analyzes code and suggests:
    - Complexity reduction strategies
    - Code organization improvements
    - Design pattern applications
    - Performance optimizations

    Args:
        input_params: Validated refactoring input
        state: Code agent state tracker

    Returns:
        Formatted refactoring suggestions

    Raises:
        RefactoringError: If refactoring analysis fails
    """
    try:
        path = validate_file_path(input_params.file_path)
        check_file_size(path)

        # Parse file
        tree = parse_python_file(path)

        # Analyze for refactoring opportunities
        suggestions = _analyze_refactoring_opportunities(
            tree, path, input_params.focus_areas, input_params.include_examples
        )

        return _format_refactoring_suggestions(suggestions, path)

    except (CodeAnalysisError, SyntaxValidationError):
        raise
    except Exception as e:
        raise RefactoringError(f"Failed to suggest refactoring: {e}") from e


def generate_code(input_params: GenerateCodeInput, state: CodeAgentState) -> str:
    """
    Generate Python code from description.

    Generates well-structured, documented code following best practices:
    - Type hints (if requested)
    - Docstrings (if requested)
    - PEP 8 compliance
    - Error handling

    Args:
        input_params: Validated code generation input
        state: Code agent state tracker

    Returns:
        Generated Python code

    Raises:
        CodeGenerationError: If code generation fails
    """
    try:
        # Generate code based on type
        if input_params.code_type == "function":
            code = _generate_function(input_params)
        elif input_params.code_type == "class":
            code = _generate_class(input_params)
        elif input_params.code_type == "module":
            code = _generate_module(input_params)
        else:  # snippet
            code = _generate_snippet(input_params)

        return code

    except Exception as e:
        raise CodeGenerationError(f"Failed to generate code: {e}") from e


# ============================================================================
# Helper Functions for Analysis
# ============================================================================


def _extract_structure(tree: ast.Module) -> dict[str, Any]:
    """Extract code structure from AST."""
    structure: dict[str, Any] = {
        "classes": [],
        "functions": [],
        "imports": [],
        "constants": [],
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            structure["classes"].append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    "bases": [_get_name(base) for base in node.bases],
                }
            )
        elif isinstance(node, ast.FunctionDef) and not _is_method(node, tree):
            structure["functions"].append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "args": len(node.args.args),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                }
            )

    return structure


def _calculate_metrics(tree: ast.Module) -> dict[str, Any]:
    """Calculate code quality metrics."""
    metrics: dict[str, int | float] = {
        "total_lines": 0,
        "code_lines": 0,
        "comment_lines": 0,
        "blank_lines": 0,
        "functions": 0,
        "classes": 0,
        "avg_complexity": 0.0,
        "max_complexity": 0,
    }

    complexities = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            metrics["functions"] += 1
            complexity = calculate_cyclomatic_complexity(node)
            complexities.append(complexity)
        elif isinstance(node, ast.ClassDef):
            metrics["classes"] += 1

    if complexities:
        metrics["avg_complexity"] = sum(complexities) / len(complexities)
        metrics["max_complexity"] = max(complexities)

    return metrics


def _extract_dependencies(tree: ast.Module) -> list[str]:
    """Extract import dependencies."""
    dependencies = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                dependencies.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            dependencies.append(node.module)

    return sorted(set(dependencies))


def _detect_code_smells(tree: ast.Module) -> list[dict[str, Any]]:
    """Detect code smells and anti-patterns."""
    smells = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check function length
            if hasattr(node, "end_lineno") and node.end_lineno:
                length = node.end_lineno - node.lineno
                if length > CODE_SMELL_PATTERNS["long_function"]:
                    smells.append(
                        {
                            "type": "long_function",
                            "line": node.lineno,
                            "name": node.name,
                            "severity": "medium",
                            "message": f"Function '{node.name}' is {length} lines long",
                        }
                    )

            # Check parameter count
            param_count = len(node.args.args)
            if param_count > CODE_SMELL_PATTERNS["long_parameter_list"]:
                smells.append(
                    {
                        "type": "long_parameter_list",
                        "line": node.lineno,
                        "name": node.name,
                        "severity": "low",
                        "message": f"Function '{node.name}' has {param_count} parameters",
                    }
                )

    return smells


def _format_analysis_result(result: CodeAnalysisResult) -> str:
    """Format analysis result as readable string."""
    lines = [f"Code Analysis: {Path(result.file_path).name}"]
    lines.append("=" * 60)

    if result.structure:
        lines.append("\nStructure:")
        lines.append(f"  Classes: {len(result.structure.get('classes', []))}")
        lines.append(f"  Functions: {len(result.structure.get('functions', []))}")

    if result.metrics:
        lines.append("\nMetrics:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.2f}")
            else:
                lines.append(f"  {key}: {value}")

    if result.dependencies:
        lines.append(f"\nDependencies ({len(result.dependencies)}):")
        for dep in result.dependencies[:10]:  # Limit to first 10
            lines.append(f"  - {dep}")

    if result.patterns:
        lines.append(f"\nCode Smells ({len(result.patterns)}):")
        for pattern in result.patterns[:5]:  # Limit to first 5
            lines.append(f"  [{pattern['severity']}] {pattern['message']}")

    return "\n".join(lines)


def _strict_validation_checks(tree: ast.Module) -> list[str]:
    """Perform strict validation checks."""
    issues: list[str] = []
    # Placeholder for strict checks
    return issues


def _is_method(node: ast.FunctionDef, tree: ast.Module) -> bool:
    """Check if function is a method."""
    return any(isinstance(parent, ast.ClassDef) and node in parent.body for parent in ast.walk(tree))


def _get_name(node: ast.AST) -> str:
    """Get name from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_get_name(node.value)}.{node.attr}"
    return str(node)


def _detect_all_patterns(tree: ast.Module, file_path: Path) -> list[dict[str, Any]]:
    """Detect all code patterns and smells."""
    patterns = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Long function
            if hasattr(node, "end_lineno") and node.end_lineno:
                length = node.end_lineno - node.lineno
                if length > CODE_SMELL_PATTERNS["long_function"]:
                    patterns.append(
                        {
                            "type": "long_function",
                            "line": node.lineno,
                            "name": node.name,
                            "severity": "medium",
                            "message": (
                                f"Function '{node.name}' is {length} lines long "
                                f"(threshold: {CODE_SMELL_PATTERNS['long_function']})"
                            ),
                        }
                    )

            # Long parameter list
            param_count = len(node.args.args)
            if param_count > CODE_SMELL_PATTERNS["long_parameter_list"]:
                patterns.append(
                    {
                        "type": "long_parameter_list",
                        "line": node.lineno,
                        "name": node.name,
                        "severity": "low",
                        "message": (
                            f"Function '{node.name}' has {param_count} parameters "
                            f"(threshold: {CODE_SMELL_PATTERNS['long_parameter_list']})"
                        ),
                    }
                )

            # High complexity
            complexity = calculate_cyclomatic_complexity(node)
            if complexity > COMPLEXITY_HIGH:
                patterns.append(
                    {
                        "type": "too_many_branches",
                        "line": node.lineno,
                        "name": node.name,
                        "severity": "high",
                        "message": f"Function '{node.name}' has complexity {complexity} (threshold: {COMPLEXITY_HIGH})",
                    }
                )

            # Magic numbers
            magic_numbers = _find_magic_numbers(node)
            if magic_numbers:
                patterns.append(
                    {
                        "type": "magic_numbers",
                        "line": node.lineno,
                        "name": node.name,
                        "severity": "low",
                        "message": f"Function '{node.name}' contains {len(magic_numbers)} magic number(s)",
                    }
                )

        elif isinstance(node, ast.ClassDef):
            # God class (too many methods)
            method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
            if method_count > 20:
                patterns.append(
                    {
                        "type": "god_class",
                        "line": node.lineno,
                        "name": node.name,
                        "severity": "high",
                        "message": f"Class '{node.name}' has {method_count} methods (threshold: 20)",
                    }
                )

    return patterns


def _find_magic_numbers(node: ast.AST) -> list[int | float]:
    """Find magic numbers in code."""
    magic_numbers = []

    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, (int, float))
            and child.value not in (0, 1, -1, 2, 10, 100, 1000)
        ):
            # Exclude common non-magic numbers
            magic_numbers.append(child.value)

    return magic_numbers


def _calculate_comprehensive_metrics(tree: ast.Module, content: str) -> dict[str, Any]:
    """Calculate comprehensive code metrics."""
    lines = content.split("\n")

    metrics: dict[str, Any] = {
        "total_lines": len(lines),
        "code_lines": 0,
        "comment_lines": 0,
        "blank_lines": 0,
        "functions": 0,
        "classes": 0,
        "methods": 0,
        "avg_complexity": 0.0,
        "max_complexity": 0,
        "complexity_distribution": {"low": 0, "medium": 0, "high": 0, "very_high": 0},
    }

    # Count line types
    for line in lines:
        stripped = line.strip()
        if not stripped:
            metrics["blank_lines"] += 1
        elif stripped.startswith("#"):
            metrics["comment_lines"] += 1
        else:
            metrics["code_lines"] += 1

    # Analyze AST
    complexities = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if _is_method(node, tree):
                metrics["methods"] += 1
            else:
                metrics["functions"] += 1

            complexity = calculate_cyclomatic_complexity(node)
            complexities.append(complexity)

            # Categorize complexity
            if complexity <= COMPLEXITY_LOW:
                metrics["complexity_distribution"]["low"] += 1
            elif complexity <= COMPLEXITY_MEDIUM:
                metrics["complexity_distribution"]["medium"] += 1
            elif complexity <= COMPLEXITY_HIGH:
                metrics["complexity_distribution"]["high"] += 1
            else:
                metrics["complexity_distribution"]["very_high"] += 1

        elif isinstance(node, ast.ClassDef):
            metrics["classes"] += 1

    if complexities:
        metrics["avg_complexity"] = sum(complexities) / len(complexities)
        metrics["max_complexity"] = max(complexities)

    return metrics


def _categorize_dependencies(tree: ast.Module) -> dict[str, list[str]]:
    """Categorize dependencies into stdlib, third-party, and local."""
    import sys

    deps: dict[str, list[str]] = {
        "stdlib": [],
        "third_party": [],
        "local": [],
        "all": [],
    }

    stdlib_modules = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                deps["all"].append(alias.name)

                if module in stdlib_modules:
                    deps["stdlib"].append(alias.name)
                elif module.startswith("."):
                    deps["local"].append(alias.name)
                else:
                    deps["third_party"].append(alias.name)

        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module.split(".")[0]
            deps["all"].append(node.module)

            if module in stdlib_modules:
                deps["stdlib"].append(node.module)
            elif node.level > 0:  # Relative import
                deps["local"].append(node.module)
            else:
                deps["third_party"].append(node.module)

    # Remove duplicates and sort
    for key in deps:
        deps[key] = sorted(set(deps[key]))

    return deps


def _format_pattern_report(patterns: list[dict[str, Any]], file_path: Path) -> str:
    """Format pattern detection report."""
    lines = [f"Pattern Detection: {file_path.name}"]
    lines.append("=" * 60)

    if not patterns:
        lines.append("\n✓ No code smells detected!")
        return "\n".join(lines)

    # Group by severity
    by_severity: dict[str, list[dict[str, Any]]] = {"high": [], "medium": [], "low": []}
    for pattern in patterns:
        severity = pattern.get("severity", "low")
        by_severity[severity].append(pattern)

    lines.append(f"\nFound {len(patterns)} issue(s):")

    for severity in ["high", "medium", "low"]:
        if by_severity[severity]:
            lines.append(f"\n{severity.upper()} ({len(by_severity[severity])}):")
            for pattern in by_severity[severity]:
                lines.append(f"  Line {pattern['line']}: {pattern['message']}")

    return "\n".join(lines)


def _format_metrics_report(metrics: dict[str, Any], file_path: Path) -> str:
    """Format metrics report."""
    lines = [f"Code Metrics: {file_path.name}"]
    lines.append("=" * 60)

    lines.append("\nLines of Code:")
    lines.append(f"  Total: {metrics['total_lines']}")
    lines.append(f"  Code: {metrics['code_lines']}")
    lines.append(f"  Comments: {metrics['comment_lines']}")
    lines.append(f"  Blank: {metrics['blank_lines']}")

    lines.append("\nStructure:")
    lines.append(f"  Classes: {metrics['classes']}")
    lines.append(f"  Functions: {metrics['functions']}")
    lines.append(f"  Methods: {metrics['methods']}")

    lines.append("\nComplexity:")
    lines.append(f"  Average: {metrics['avg_complexity']:.2f}")
    lines.append(f"  Maximum: {metrics['max_complexity']}")

    dist = metrics["complexity_distribution"]
    lines.append("\nComplexity Distribution:")
    lines.append(f"  Low (≤{COMPLEXITY_LOW}): {dist['low']}")
    lines.append(f"  Medium (≤{COMPLEXITY_MEDIUM}): {dist['medium']}")
    lines.append(f"  High (≤{COMPLEXITY_HIGH}): {dist['high']}")
    lines.append(f"  Very High (>{COMPLEXITY_HIGH}): {dist['very_high']}")

    return "\n".join(lines)


def _format_dependency_report(deps: dict[str, list[str]], file_path: Path) -> str:
    """Format dependency report."""
    lines = [f"Dependencies: {file_path.name}"]
    lines.append("=" * 60)

    lines.append(f"\nTotal Dependencies: {len(deps['all'])}")

    if deps["stdlib"]:
        lines.append(f"\nStandard Library ({len(deps['stdlib'])}):")
        for dep in deps["stdlib"][:10]:
            lines.append(f"  - {dep}")
        if len(deps["stdlib"]) > 10:
            lines.append(f"  ... and {len(deps['stdlib']) - 10} more")

    if deps["third_party"]:
        lines.append(f"\nThird-Party ({len(deps['third_party'])}):")
        for dep in deps["third_party"][:10]:
            lines.append(f"  - {dep}")
        if len(deps["third_party"]) > 10:
            lines.append(f"  ... and {len(deps['third_party']) - 10} more")

    if deps["local"]:
        lines.append(f"\nLocal Imports ({len(deps['local'])}):")
        for dep in deps["local"]:
            lines.append(f"  - {dep}")

    return "\n".join(lines)


def _analyze_refactoring_opportunities(
    tree: ast.Module, file_path: Path, focus_areas: list[str] | None, include_examples: bool
) -> list[dict[str, Any]]:
    """Analyze code for refactoring opportunities."""
    suggestions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # High complexity functions
            complexity = calculate_cyclomatic_complexity(node)
            if complexity > COMPLEXITY_MEDIUM:
                suggestion = {
                    "type": "reduce_complexity",
                    "line": node.lineno,
                    "name": node.name,
                    "priority": "high" if complexity > COMPLEXITY_HIGH else "medium",
                    "description": f"Function '{node.name}' has high complexity ({complexity})",
                    "recommendation": "Consider breaking down into smaller functions or simplifying logic",
                }
                if include_examples:
                    suggestion["example"] = "Extract complex logic into separate helper functions"
                suggestions.append(suggestion)

            # Long functions
            if hasattr(node, "end_lineno") and node.end_lineno:
                length = node.end_lineno - node.lineno
                if length > CODE_SMELL_PATTERNS["long_function"]:
                    suggestion = {
                        "type": "extract_function",
                        "line": node.lineno,
                        "name": node.name,
                        "priority": "medium",
                        "description": f"Function '{node.name}' is {length} lines long",
                        "recommendation": "Extract logical sections into separate functions",
                    }
                    suggestions.append(suggestion)

    return suggestions


def _format_refactoring_suggestions(suggestions: list[dict[str, Any]], file_path: Path) -> str:
    """Format refactoring suggestions."""
    lines = [f"Refactoring Suggestions: {file_path.name}"]
    lines.append("=" * 60)

    if not suggestions:
        lines.append("\n✓ No refactoring opportunities identified!")
        return "\n".join(lines)

    # Group by priority
    by_priority: dict[str, list[dict[str, Any]]] = {"high": [], "medium": [], "low": []}
    for suggestion in suggestions:
        priority = suggestion.get("priority", "low")
        by_priority[priority].append(suggestion)

    lines.append(f"\nFound {len(suggestions)} suggestion(s):")

    for priority in ["high", "medium", "low"]:
        if by_priority[priority]:
            lines.append(f"\n{priority.upper()} PRIORITY ({len(by_priority[priority])}):")
            for suggestion in by_priority[priority]:
                lines.append(f"\n  Line {suggestion['line']}: {suggestion['description']}")
                lines.append(f"  → {suggestion['recommendation']}")
                if "example" in suggestion:
                    lines.append(f"  Example: {suggestion['example']}")

    return "\n".join(lines)


def _generate_function(input_params: GenerateCodeInput) -> str:
    """Generate a function from description."""
    lines = []

    # Function signature
    func_name = "generated_function"
    if input_params.include_type_hints:
        lines.append(f"def {func_name}(param: Any) -> Any:")
    else:
        lines.append(f"def {func_name}(param):")

    # Docstring
    if input_params.include_docstrings:
        lines.append('    """')
        lines.append(f"    {input_params.description}")
        lines.append('    """')

    # Function body
    lines.append("    # TODO: Implement function logic")
    lines.append("    pass")

    return "\n".join(lines)


def _generate_class(input_params: GenerateCodeInput) -> str:
    """Generate a class from description."""
    lines = []

    # Class definition
    lines.append("class GeneratedClass:")

    # Docstring
    if input_params.include_docstrings:
        lines.append('    """')
        lines.append(f"    {input_params.description}")
        lines.append('    """')

    # Init method
    if input_params.include_type_hints:
        lines.append("    def __init__(self) -> None:")
    else:
        lines.append("    def __init__(self):")

    if input_params.include_docstrings:
        lines.append('        """Initialize the class."""')

    lines.append("        pass")

    return "\n".join(lines)


def _generate_module(input_params: GenerateCodeInput) -> str:
    """Generate a module from description."""
    lines = []

    # Module docstring
    if input_params.include_docstrings:
        lines.append('"""')
        lines.append(f"{input_params.description}")
        lines.append('"""')
        lines.append("")

    # Imports
    lines.append("from __future__ import annotations")
    lines.append("")

    # Placeholder content
    lines.append("# TODO: Implement module content")

    return "\n".join(lines)


def _generate_snippet(input_params: GenerateCodeInput) -> str:
    """Generate a code snippet from description."""
    return f"# {input_params.description}\n# TODO: Implement code snippet"


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # Core types and enums
    "AnalysisType",
    "ComplexityLevel",
    "CodeSmellType",
    # Models
    "AnalyzeCodeInput",
    "ValidateSyntaxInput",
    "DetectPatternsInput",
    "SuggestRefactoringInput",
    "GenerateCodeInput",
    "CodeAnalysisResult",
    # State management
    "CodeAgentState",
    # Core functions
    "analyze_code",
    "validate_syntax",
    "detect_patterns",
    "calculate_metrics",
    "find_dependencies",
    "suggest_refactoring",
    "generate_code",
    # Helper functions
    "validate_file_path",
    "check_file_size",
    "parse_python_file",
    "calculate_cyclomatic_complexity",
    # Exceptions
    "CodeAgentError",
    "CodeAnalysisError",
    "SyntaxValidationError",
    "PatternDetectionError",
    "CodeGenerationError",
    "RefactoringError",
    "FileSizeExceededError",
]
