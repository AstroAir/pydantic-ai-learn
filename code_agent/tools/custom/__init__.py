"""
Custom Tools Module

Extended tools for code formatting, linting, dependency analysis, and documentation.

This module provides custom extensions to the Code Agent toolkit:
- CodeFormatter: Format code using black/ruff
- CodeLinter: Lint code using ruff
- DependencyAnalyzer: Analyze imports and dependencies
- DocumentationAnalyzer: Analyze docstring coverage and quality

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

from .dependencies import DependencyAnalysis, DependencyAnalyzer, DependencyIssue, ImportInfo
from .documentation import DocstringInfo, DocumentationAnalysis, DocumentationAnalyzer
from .formatter import CodeFormatter, FormattedCode
from .linter import CodeLinter, LintIssue, LintResult
from .validators import (
    create_documentation_validator,
    create_quality_validator,
    validate_best_practices,
    validate_code_smells,
    validate_enhanced_complexity,
    validate_naming_conventions,
)

__all__ = [
    # Formatter
    "CodeFormatter",
    "FormattedCode",
    # Linter
    "CodeLinter",
    "LintResult",
    "LintIssue",
    # Dependencies
    "DependencyAnalyzer",
    "DependencyAnalysis",
    "ImportInfo",
    "DependencyIssue",
    # Documentation
    "DocumentationAnalyzer",
    "DocumentationAnalysis",
    "DocstringInfo",
    # Validators
    "validate_naming_conventions",
    "create_documentation_validator",
    "validate_enhanced_complexity",
    "validate_code_smells",
    "validate_best_practices",
    "create_quality_validator",
]
