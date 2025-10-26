"""
Documentation Analyzer Tool

Analyzes docstring coverage and quality in Python code.

Features:
- Docstring coverage analysis
- Style validation (Google, NumPy, Sphinx)
- Missing documentation detection
- Parameter/return documentation checks
- Documentation quality scoring

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from ...config.tools import DocumentationConfig

# ============================================================================
# Result Types
# ============================================================================


@dataclass
class DocstringInfo:
    """Information about a docstring."""

    name: str
    """Function/class/module name"""

    type: str
    """Type (module, class, function, method)"""

    has_docstring: bool
    """Whether docstring exists"""

    docstring: str | None = None
    """Docstring content"""

    line: int = 0
    """Line number"""

    has_params: bool = False
    """Whether parameters are documented"""

    has_returns: bool = False
    """Whether return value is documented"""

    issues: list[str] = field(default_factory=list)
    """Documentation issues"""


@dataclass
class DocumentationAnalysis:
    """Result of documentation analysis."""

    coverage: float = 0.0
    """Documentation coverage (0.0-1.0)"""

    total_items: int = 0
    """Total documentable items"""

    documented_items: int = 0
    """Number of documented items"""

    docstrings: list[DocstringInfo] = field(default_factory=list)
    """All docstrings found"""

    missing: list[str] = field(default_factory=list)
    """Items missing documentation"""

    issues: list[str] = field(default_factory=list)
    """Documentation issues"""

    suggestions: list[str] = field(default_factory=list)
    """Improvement suggestions"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    @property
    def meets_threshold(self) -> bool:
        """Check if coverage meets threshold."""
        return self.coverage >= 0.8  # Default threshold

    @property
    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return len(self.issues) > 0


# ============================================================================
# Documentation Analyzer
# ============================================================================


class DocumentationAnalyzer:
    """
    Documentation analyzer for Python code.

    Analyzes docstring coverage and quality.
    """

    def __init__(self, config: DocumentationConfig | None = None) -> None:
        """
        Initialize documentation analyzer.

        Args:
            config: Documentation analyzer configuration
        """
        self.config = config or DocumentationConfig()

    def analyze(self, code: str) -> DocumentationAnalysis:
        """
        Analyze code documentation.

        Args:
            code: Python code to analyze

        Returns:
            Documentation analysis result
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return DocumentationAnalysis(
                metadata={"error": f"Syntax error: {e}"},
            )

        # Analyze module docstring
        docstrings = []
        module_doc = ast.get_docstring(tree)

        if self.config.require_module_docstring:
            docstrings.append(
                DocstringInfo(
                    name="<module>",
                    type="module",
                    has_docstring=module_doc is not None,
                    docstring=module_doc,
                    line=1,
                )
            )

        # Analyze classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                doc_info = self._analyze_class(node)
                docstrings.append(doc_info)

            elif isinstance(node, ast.FunctionDef):
                doc_info = self._analyze_function(node)
                docstrings.append(doc_info)

        # Calculate coverage
        total_items = len(docstrings)
        documented_items = sum(1 for d in docstrings if d.has_docstring)
        coverage = documented_items / total_items if total_items > 0 else 1.0

        # Find missing documentation
        missing = [d.name for d in docstrings if not d.has_docstring]

        # Collect issues
        issues = []
        for doc in docstrings:
            issues.extend(doc.issues)

        # Generate suggestions
        suggestions = self._generate_suggestions(docstrings, coverage)

        return DocumentationAnalysis(
            coverage=coverage,
            total_items=total_items,
            documented_items=documented_items,
            docstrings=docstrings,
            missing=missing,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "style": self.config.docstring_style,
                "threshold": self.config.min_coverage,
            },
        )

    def _analyze_class(self, node: ast.ClassDef) -> DocstringInfo:
        """Analyze class documentation."""
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None

        issues = []
        if self.config.require_class_docstring and not has_docstring:
            issues.append(f"Class '{node.name}' missing docstring")

        return DocstringInfo(
            name=node.name,
            type="class",
            has_docstring=has_docstring,
            docstring=docstring,
            line=node.lineno,
            issues=issues,
        )

    def _analyze_function(self, node: ast.FunctionDef) -> DocstringInfo:
        """Analyze function documentation."""
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None

        # Check for parameters
        has_params = False
        has_returns = False

        if docstring:
            has_params = self._check_param_docs(docstring, node)
            has_returns = self._check_return_docs(docstring, node)

        # Collect issues
        issues = []

        if self.config.require_function_docstring and not has_docstring:
            issues.append(f"Function '{node.name}' missing docstring")

        if has_docstring:
            if self.config.require_param_docs and node.args.args and not has_params:
                issues.append(f"Function '{node.name}' missing parameter documentation")

            if self.config.require_return_docs and node.returns and not has_returns:
                issues.append(f"Function '{node.name}' missing return documentation")

        return DocstringInfo(
            name=node.name,
            type="function",
            has_docstring=has_docstring,
            docstring=docstring,
            line=node.lineno,
            has_params=has_params,
            has_returns=has_returns,
            issues=issues,
        )

    def _check_param_docs(self, docstring: str, node: ast.FunctionDef) -> bool:
        """Check if parameters are documented."""
        if not node.args.args:
            return True  # No params to document

        # Simple heuristic: check for "Args:" or "Parameters:" section
        if self.config.docstring_style == "google":
            return "Args:" in docstring or "Arguments:" in docstring
        if self.config.docstring_style == "numpy":
            return "Parameters" in docstring
        if self.config.docstring_style == "sphinx":
            return ":param" in docstring

        return False

    def _check_return_docs(self, docstring: str, node: ast.FunctionDef) -> bool:
        """Check if return value is documented."""
        if not node.returns:
            return True  # No return to document

        # Simple heuristic: check for "Returns:" section
        if self.config.docstring_style == "google":
            return "Returns:" in docstring
        if self.config.docstring_style == "numpy":
            return "Returns" in docstring
        if self.config.docstring_style == "sphinx":
            return ":return" in docstring or ":returns:" in docstring

        return False

    def _generate_suggestions(
        self,
        docstrings: list[DocstringInfo],
        coverage: float,
    ) -> list[str]:
        """Generate improvement suggestions."""
        suggestions = []

        # Coverage suggestions
        if coverage < self.config.min_coverage:
            suggestions.append(
                f"Documentation coverage ({coverage:.1%}) is below threshold ({self.config.min_coverage:.1%})"
            )

        # Missing docstrings
        missing_count = sum(1 for d in docstrings if not d.has_docstring)
        if missing_count > 0:
            suggestions.append(f"Add docstrings to {missing_count} undocumented items")

        # Parameter documentation
        missing_params = sum(1 for d in docstrings if d.type == "function" and d.has_docstring and not d.has_params)
        if missing_params > 0:
            suggestions.append(f"Add parameter documentation to {missing_params} functions")

        # Return documentation
        missing_returns = sum(1 for d in docstrings if d.type == "function" and d.has_docstring and not d.has_returns)
        if missing_returns > 0:
            suggestions.append(f"Add return documentation to {missing_returns} functions")

        return suggestions
