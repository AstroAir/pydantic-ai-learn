"""
Code Analyzer Tool

Comprehensive code analysis with metrics, patterns, and quality assessment.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass
class CodeMetrics:
    """Code metrics and statistics."""

    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    number_of_functions: int = 0
    number_of_classes: int = 0
    number_of_imports: int = 0
    average_function_length: float = 0.0


class CodeAnalyzer:
    """
    Comprehensive code analysis tool.

    Provides:
    - Syntax validation
    - Metrics calculation
    - Pattern detection
    - Dependency analysis
    """

    def __init__(self) -> None:
        """Initialize code analyzer."""
        pass

    def analyze(self, code: str) -> dict[str, Any]:
        """
        Analyze code and return comprehensive metrics.

        Args:
            code: Python code to analyze

        Returns:
            Dictionary with analysis results
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
            }

        metrics = self._calculate_metrics(code, tree)
        patterns = self._detect_patterns(tree)
        dependencies = self._analyze_dependencies(tree)

        return {
            "valid": True,
            "metrics": metrics,
            "patterns": patterns,
            "dependencies": dependencies,
        }

    def _calculate_metrics(self, code: str, tree: ast.AST) -> dict[str, Any]:
        """Calculate code metrics."""
        lines = code.split("\n")
        loc = len([line for line in lines if line.strip() and not line.strip().startswith("#")])

        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]

        avg_func_length = loc / len(functions) if functions else 0

        return {
            "lines_of_code": loc,
            "number_of_functions": len(functions),
            "number_of_classes": len(classes),
            "number_of_imports": len(imports),
            "average_function_length": avg_func_length,
        }

    def _detect_patterns(self, tree: ast.AST) -> list[str]:
        """Detect code patterns and potential issues."""
        patterns = []

        for node in ast.walk(tree):
            # Detect long functions
            if isinstance(node, ast.FunctionDef) and len(node.body) > 20:
                patterns.append(f"Long function: {node.name} ({len(node.body)} lines)")

            # Detect nested loops
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        patterns.append("Nested loop detected")
                        break

        return patterns

    def _analyze_dependencies(self, tree: ast.AST) -> dict[str, list[str]]:
        """Analyze code dependencies."""
        imports = []
        from_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    from_imports.append(f"{module}.{alias.name}")

        return {
            "imports": imports,
            "from_imports": from_imports,
        }

    def validate_syntax(self, code: str) -> tuple[bool, str | None]:
        """
        Validate Python syntax.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def get_complexity(self, code: str) -> str:
        """
        Estimate code complexity.

        Args:
            code: Python code

        Returns:
            Complexity level: "low", "medium", or "high"
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "unknown"

        metrics = self._calculate_metrics(code, tree)
        loc = metrics["lines_of_code"]

        if loc < 50:
            return "low"
        if loc < 200:
            return "medium"
        return "high"


__all__ = ["CodeAnalyzer", "CodeMetrics"]
