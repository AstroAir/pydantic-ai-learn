"""
Custom Tools Configuration

Configuration dataclasses for custom tools (formatter, linter, etc.).

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ============================================================================
# Enums
# ============================================================================


class FormatterBackend(str, Enum):
    """Code formatter backend."""

    BLACK = "black"
    RUFF = "ruff"
    AUTO = "auto"  # Choose best available


class LintSeverity(str, Enum):
    """Lint issue severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


# ============================================================================
# Formatter Configuration
# ============================================================================


@dataclass
class FormatterConfig:
    """Configuration for code formatter."""

    backend: FormatterBackend = FormatterBackend.AUTO
    """Formatter backend to use"""

    line_length: int = 88
    """Maximum line length (black default: 88, PEP 8: 79)"""

    target_version: str = "py311"
    """Python version target (e.g., 'py311', 'py312')"""

    skip_string_normalization: bool = False
    """Skip normalizing string quotes"""

    skip_magic_trailing_comma: bool = False
    """Skip adding trailing commas"""

    preview: bool = False
    """Enable preview features"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend.value,
            "line_length": self.line_length,
            "target_version": self.target_version,
            "skip_string_normalization": self.skip_string_normalization,
            "skip_magic_trailing_comma": self.skip_magic_trailing_comma,
            "preview": self.preview,
        }


# ============================================================================
# Linter Configuration
# ============================================================================


@dataclass
class LinterConfig:
    """Configuration for code linter."""

    line_length: int = 88
    """Maximum line length"""

    target_version: str = "py311"
    """Python version target"""

    select_rules: list[str] = field(default_factory=lambda: ["E", "F", "W", "C90", "I", "N"])
    """Lint rules to enable (E=pycodestyle errors, F=pyflakes, W=warnings, C90=complexity, I=isort, N=naming)"""

    ignore_rules: list[str] = field(default_factory=list)
    """Lint rules to ignore"""

    max_complexity: int = 10
    """Maximum cyclomatic complexity"""

    min_severity: LintSeverity = LintSeverity.WARNING
    """Minimum severity to report"""

    auto_fix: bool = False
    """Automatically fix issues when possible"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "line_length": self.line_length,
            "target_version": self.target_version,
            "select_rules": self.select_rules,
            "ignore_rules": self.ignore_rules,
            "max_complexity": self.max_complexity,
            "min_severity": self.min_severity.value,
            "auto_fix": self.auto_fix,
        }


# ============================================================================
# Dependency Analyzer Configuration
# ============================================================================


@dataclass
class DependencyConfig:
    """Configuration for dependency analyzer."""

    check_circular: bool = True
    """Check for circular dependencies"""

    check_unused: bool = True
    """Check for unused imports"""

    check_stdlib: bool = True
    """Include standard library in analysis"""

    check_third_party: bool = True
    """Include third-party packages in analysis"""

    suggest_optimizations: bool = True
    """Suggest import optimizations"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_circular": self.check_circular,
            "check_unused": self.check_unused,
            "check_stdlib": self.check_stdlib,
            "check_third_party": self.check_third_party,
            "suggest_optimizations": self.suggest_optimizations,
        }


# ============================================================================
# Documentation Analyzer Configuration
# ============================================================================


@dataclass
class DocumentationConfig:
    """Configuration for documentation analyzer."""

    min_coverage: float = 0.8
    """Minimum documentation coverage (0.0-1.0)"""

    require_module_docstring: bool = True
    """Require module-level docstrings"""

    require_class_docstring: bool = True
    """Require class docstrings"""

    require_function_docstring: bool = True
    """Require function docstrings"""

    require_param_docs: bool = True
    """Require parameter documentation"""

    require_return_docs: bool = True
    """Require return value documentation"""

    docstring_style: str = "google"
    """Docstring style (google, numpy, sphinx)"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_coverage": self.min_coverage,
            "require_module_docstring": self.require_module_docstring,
            "require_class_docstring": self.require_class_docstring,
            "require_function_docstring": self.require_function_docstring,
            "require_param_docs": self.require_param_docs,
            "require_return_docs": self.require_return_docs,
            "docstring_style": self.docstring_style,
        }


# ============================================================================
# Quality Workflow Configuration
# ============================================================================


@dataclass
class QualityWorkflowConfig:
    """Configuration for quality workflow."""

    enable_analysis: bool = True
    """Enable code analysis step"""

    enable_linting: bool = True
    """Enable linting step"""

    enable_formatting: bool = True
    """Enable formatting step"""

    enable_refactoring: bool = True
    """Enable refactoring suggestions step"""

    enable_documentation: bool = True
    """Enable documentation check step"""

    fail_on_errors: bool = False
    """Fail workflow on errors"""

    auto_fix: bool = False
    """Automatically fix issues when possible"""

    formatter_config: FormatterConfig = field(default_factory=FormatterConfig)
    """Formatter configuration"""

    linter_config: LinterConfig = field(default_factory=LinterConfig)
    """Linter configuration"""

    dependency_config: DependencyConfig = field(default_factory=DependencyConfig)
    """Dependency analyzer configuration"""

    documentation_config: DocumentationConfig = field(default_factory=DocumentationConfig)
    """Documentation analyzer configuration"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_analysis": self.enable_analysis,
            "enable_linting": self.enable_linting,
            "enable_formatting": self.enable_formatting,
            "enable_refactoring": self.enable_refactoring,
            "enable_documentation": self.enable_documentation,
            "fail_on_errors": self.fail_on_errors,
            "auto_fix": self.auto_fix,
            "formatter_config": self.formatter_config.to_dict(),
            "linter_config": self.linter_config.to_dict(),
            "dependency_config": self.dependency_config.to_dict(),
            "documentation_config": self.documentation_config.to_dict(),
        }
