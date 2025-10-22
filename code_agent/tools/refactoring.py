"""
Refactoring Engine

Provides refactoring suggestions and improvements for Python code.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass
class RefactoringSuggestion:
    """A single refactoring suggestion."""

    title: str
    description: str
    severity: str  # "low", "medium", "high"
    location: str | None = None
    suggested_fix: str | None = None


class RefactoringEngine:
    """
    Code refactoring suggestion engine.

    Provides:
    - Code smell detection
    - Refactoring suggestions
    - Best practice recommendations
    """

    def __init__(self) -> None:
        """Initialize refactoring engine."""
        pass

    def suggest_refactoring(self, code: str) -> list[RefactoringSuggestion]:
        """
        Suggest refactoring improvements.

        Args:
            code: Python code to analyze

        Returns:
            List of refactoring suggestions
        """
        suggestions: list[RefactoringSuggestion] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return suggestions

        # Check for various code smells
        suggestions.extend(self._check_long_functions(tree))
        suggestions.extend(self._check_complex_conditions(tree))
        suggestions.extend(self._check_naming_conventions(tree))
        suggestions.extend(self._check_code_duplication(code))

        return suggestions

    def _check_long_functions(self, tree: ast.AST) -> list[RefactoringSuggestion]:
        """Check for functions that are too long."""
        suggestions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and len(node.body) > 20:
                suggestions.append(
                    RefactoringSuggestion(
                        title="Long Function",
                        description=f"Function '{node.name}' is too long ({len(node.body)} lines)",
                        severity="medium",
                        location=f"Line {node.lineno}",
                        suggested_fix="Consider breaking this function into smaller, more focused functions",
                    )
                )

        return suggestions

    def _check_complex_conditions(self, tree: ast.AST) -> list[RefactoringSuggestion]:
        """Check for overly complex conditions."""
        suggestions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Count boolean operations
                bool_ops = len([n for n in ast.walk(node.test) if isinstance(n, ast.BoolOp)])
                if bool_ops > 3:
                    suggestions.append(
                        RefactoringSuggestion(
                            title="Complex Condition",
                            description=f"Condition has {bool_ops} boolean operations",
                            severity="medium",
                            location=f"Line {node.lineno}",
                            suggested_fix="Extract condition into a named variable or helper function",
                        )
                    )

        return suggestions

    def _check_naming_conventions(self, tree: ast.AST) -> list[RefactoringSuggestion]:
        """Check for naming convention violations."""
        suggestions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_") and node.name.count("_") > 2:
                suggestions.append(
                    RefactoringSuggestion(
                        title="Unusual Naming",
                        description=f"Function name '{node.name}' has unusual underscores",
                        severity="low",
                        location=f"Line {node.lineno}",
                        suggested_fix="Consider using a more conventional name",
                    )
                )

        return suggestions

    def _check_code_duplication(self, code: str) -> list[RefactoringSuggestion]:
        """Check for code duplication."""
        suggestions = []
        lines = code.split("\n")

        # Simple duplication check
        seen_lines = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                if stripped in seen_lines:
                    suggestions.append(
                        RefactoringSuggestion(
                            title="Duplicated Code",
                            description=f"Line {i + 1} appears to be duplicated",
                            severity="low",
                            location=f"Line {i + 1}",
                            suggested_fix="Extract duplicated code into a helper function",
                        )
                    )
                else:
                    seen_lines[stripped] = i

        return suggestions

    def get_improvement_score(self, code: str) -> float:
        """
        Calculate code improvement score (0-100).

        Args:
            code: Python code

        Returns:
            Improvement score
        """
        suggestions = self.suggest_refactoring(code)

        if not suggestions:
            return 100.0

        # Calculate score based on severity
        severity_weights = {"low": 5, "medium": 15, "high": 30}
        total_deduction = sum(severity_weights.get(s.severity, 10) for s in suggestions)

        return max(0.0, 100.0 - total_deduction)


__all__ = ["RefactoringEngine", "RefactoringSuggestion"]
