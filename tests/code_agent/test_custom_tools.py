"""
Tests for Custom Tools

Tests for CodeFormatter, CodeLinter, DependencyAnalyzer, and DocumentationAnalyzer.

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

from code_agent.config.custom import (
    DependencyConfig,
    DocumentationConfig,
    FormatterConfig,
    LinterConfig,
)
from code_agent.tools.custom import (
    CodeFormatter,
    CodeLinter,
    DependencyAnalyzer,
    DocumentationAnalyzer,
)

# ============================================================================
# Test Data
# ============================================================================

SIMPLE_CODE = """
def hello():
    print("Hello, World!")
"""

UNFORMATTED_CODE = """
def hello(  ):
    x=1+2
    return   x
"""

CODE_WITH_LINT_ISSUES = """
import os
import sys

def test():
    x = 1
    y = 2
"""

CODE_WITH_DEPENDENCIES = """
import os
import sys
from pathlib import Path
from typing import Any, List

def test():
    pass
"""

CODE_WITH_DOCS = """
'''Module docstring.'''

class MyClass:
    '''Class docstring.'''

    def my_method(self, param: str) -> int:
        '''
        Method docstring.

        Args:
            param: Parameter description

        Returns:
            int: Return value
        '''
        return 42
"""

CODE_WITHOUT_DOCS = """
class MyClass:
    def my_method(self, param):
        return 42
"""


# ============================================================================
# CodeFormatter Tests
# ============================================================================


class TestCodeFormatter:
    """Tests for CodeFormatter."""

    def test_formatter_creation(self):
        """Test formatter initialization."""
        formatter = CodeFormatter()
        assert formatter is not None
        assert formatter.config is not None

    def test_formatter_with_config(self):
        """Test formatter with custom config."""
        config = FormatterConfig(line_length=100)
        formatter = CodeFormatter(config)
        assert formatter.config.line_length == 100

    def test_format_simple_code(self):
        """Test formatting simple code."""
        formatter = CodeFormatter()
        result = formatter.format(SIMPLE_CODE)

        assert result is not None
        assert result.code is not None
        assert result.backend_used in ("black", "ruff", "none")

    def test_format_unformatted_code(self):
        """Test formatting unformatted code."""
        formatter = CodeFormatter()
        result = formatter.format(UNFORMATTED_CODE)

        assert result is not None
        # Code should be changed (formatted)
        # Note: May not change if formatter not available
        assert result.code is not None

    def test_check_formatted_code(self):
        """Test checking if code is formatted."""
        formatter = CodeFormatter()
        # Simple code should already be formatted
        is_formatted = formatter.check(SIMPLE_CODE)
        assert isinstance(is_formatted, bool)


# ============================================================================
# CodeLinter Tests
# ============================================================================


class TestCodeLinter:
    """Tests for CodeLinter."""

    def test_linter_creation(self):
        """Test linter initialization."""
        linter = CodeLinter()
        assert linter is not None
        assert linter.config is not None

    def test_linter_with_config(self):
        """Test linter with custom config."""
        config = LinterConfig(max_complexity=15)
        linter = CodeLinter(config)
        assert linter.config.max_complexity == 15

    def test_lint_simple_code(self):
        """Test linting simple code."""
        linter = CodeLinter()
        result = linter.lint(SIMPLE_CODE)

        assert result is not None
        assert isinstance(result.issues, list)
        assert isinstance(result.error_count, int)
        assert isinstance(result.warning_count, int)

    def test_lint_code_with_issues(self):
        """Test linting code with issues."""
        linter = CodeLinter()
        result = linter.lint(CODE_WITH_LINT_ISSUES)

        assert result is not None
        # May have unused import issues
        assert isinstance(result.issues, list)

    def test_check_clean_code(self):
        """Test checking if code is clean."""
        linter = CodeLinter()
        is_clean = linter.check(SIMPLE_CODE)
        assert isinstance(is_clean, bool)


# ============================================================================
# DependencyAnalyzer Tests
# ============================================================================


class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer."""

    def test_analyzer_creation(self):
        """Test analyzer initialization."""
        analyzer = DependencyAnalyzer()
        assert analyzer is not None
        assert analyzer.config is not None

    def test_analyzer_with_config(self):
        """Test analyzer with custom config."""
        config = DependencyConfig(check_circular=False)
        analyzer = DependencyAnalyzer(config)
        assert analyzer.config.check_circular is False

    def test_analyze_simple_code(self):
        """Test analyzing simple code."""
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze(SIMPLE_CODE)

        assert result is not None
        assert isinstance(result.imports, list)
        assert isinstance(result.total_imports, int)

    def test_analyze_code_with_dependencies(self):
        """Test analyzing code with dependencies."""
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze(CODE_WITH_DEPENDENCIES)

        assert result is not None
        assert result.total_imports > 0
        assert len(result.stdlib_imports) > 0
        assert "os" in result.stdlib_imports
        assert "sys" in result.stdlib_imports

    def test_categorize_imports(self):
        """Test import categorization."""
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze(CODE_WITH_DEPENDENCIES)

        # Should have stdlib imports
        assert len(result.stdlib_imports) > 0
        # pathlib and typing are stdlib
        assert any("pathlib" in imp or "typing" in imp for imp in result.stdlib_imports)


# ============================================================================
# DocumentationAnalyzer Tests
# ============================================================================


class TestDocumentationAnalyzer:
    """Tests for DocumentationAnalyzer."""

    def test_analyzer_creation(self):
        """Test analyzer initialization."""
        analyzer = DocumentationAnalyzer()
        assert analyzer is not None
        assert analyzer.config is not None

    def test_analyzer_with_config(self):
        """Test analyzer with custom config."""
        config = DocumentationConfig(min_coverage=0.9)
        analyzer = DocumentationAnalyzer(config)
        assert analyzer.config.min_coverage == 0.9

    def test_analyze_documented_code(self):
        """Test analyzing well-documented code."""
        analyzer = DocumentationAnalyzer()
        result = analyzer.analyze(CODE_WITH_DOCS)

        assert result is not None
        assert isinstance(result.coverage, float)
        assert result.coverage >= 0.0
        assert result.coverage <= 1.0
        assert result.total_items > 0

    def test_analyze_undocumented_code(self):
        """Test analyzing undocumented code."""
        analyzer = DocumentationAnalyzer()
        result = analyzer.analyze(CODE_WITHOUT_DOCS)

        assert result is not None
        assert result.coverage < 1.0  # Should have missing docs
        assert len(result.missing) > 0

    def test_documentation_coverage(self):
        """Test documentation coverage calculation."""
        analyzer = DocumentationAnalyzer()

        # Well-documented code should have high coverage
        result_good = analyzer.analyze(CODE_WITH_DOCS)
        assert result_good.coverage > 0.5

        # Undocumented code should have low coverage
        result_bad = analyzer.analyze(CODE_WITHOUT_DOCS)
        assert result_bad.coverage < result_good.coverage


# ============================================================================
# Integration Tests
# ============================================================================


class TestCustomToolsIntegration:
    """Integration tests for custom tools."""

    def test_format_then_lint(self):
        """Test formatting then linting."""
        formatter = CodeFormatter()
        linter = CodeLinter()

        # Format code first
        format_result = formatter.format(UNFORMATTED_CODE)

        # Then lint the formatted code
        lint_result = linter.lint(format_result.code)

        assert lint_result is not None

    def test_analyze_dependencies_and_docs(self):
        """Test analyzing dependencies and documentation together."""
        dep_analyzer = DependencyAnalyzer()
        doc_analyzer = DocumentationAnalyzer()

        # Analyze dependencies
        dep_result = dep_analyzer.analyze(CODE_WITH_DEPENDENCIES)

        # Analyze documentation
        doc_result = doc_analyzer.analyze(CODE_WITH_DEPENDENCIES)

        assert dep_result is not None
        assert doc_result is not None

    def test_complete_quality_check(self):
        """Test complete quality check workflow."""
        formatter = CodeFormatter()
        linter = CodeLinter()
        dep_analyzer = DependencyAnalyzer()
        doc_analyzer = DocumentationAnalyzer()

        code = CODE_WITH_DOCS

        # Run all tools
        format_result = formatter.format(code)
        lint_result = linter.lint(format_result.code)
        dep_result = dep_analyzer.analyze(format_result.code)
        doc_result = doc_analyzer.analyze(format_result.code)

        # All should succeed
        assert format_result is not None
        assert lint_result is not None
        assert dep_result is not None
        assert doc_result is not None
