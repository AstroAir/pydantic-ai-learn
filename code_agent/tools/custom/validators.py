"""
Custom Validators

Additional validation rules for code quality beyond standard validators.

Features:
- Naming convention validation (PEP 8)
- Documentation coverage validation
- Enhanced complexity validation
- Code smell detection
- Best practices enforcement

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable

from ...config.tools import DocumentationConfig

# ============================================================================
# Naming Convention Validator
# ============================================================================


def validate_naming_conventions(code: str) -> list[str]:
    """
    Validate PEP 8 naming conventions.

    Args:
        code: Python code to validate

    Returns:
        List of validation errors
    """
    errors = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["Syntax error prevents naming validation"]

    for node in ast.walk(tree):
        # Class names should be CamelCase
        if isinstance(node, ast.ClassDef):
            if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                errors.append(f"Class '{node.name}' should use CamelCase (line {node.lineno})")

        # Function names should be snake_case
        elif isinstance(node, ast.FunctionDef):
            if not re.match(r"^[a-z_][a-z0-9_]*$", node.name) and not node.name.startswith("__"):
                errors.append(f"Function '{node.name}' should use snake_case (line {node.lineno})")

        # Constants should be UPPER_CASE
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and isinstance(node.value, (ast.Constant, ast.Num, ast.Str))
                    and target.id.isupper()
                    and "_" not in target.id
                    and len(target.id) > 1
                ):
                    # Single uppercase letter is OK, but multiple should have underscores
                    errors.append(f"Constant '{target.id}' should use UPPER_CASE_WITH_UNDERSCORES (line {node.lineno})")

    return errors


# ============================================================================
# Documentation Coverage Validator
# ============================================================================


def create_documentation_validator(
    min_coverage: float = 0.8,
    config: DocumentationConfig | None = None,
) -> Callable[[str], list[str]]:
    """
    Create a documentation coverage validator.

    Args:
        min_coverage: Minimum required coverage (0.0-1.0)
        config: Documentation configuration

    Returns:
        Validator function
    """
    from .documentation import DocumentationAnalyzer

    def validator(code: str) -> list[str]:
        """Validate documentation coverage."""
        analyzer = DocumentationAnalyzer(config)
        result = analyzer.analyze(code)

        errors = []

        if result.coverage < min_coverage:
            errors.append(
                f"Documentation coverage ({result.coverage:.1%}) is below required threshold ({min_coverage:.1%})"
            )

        # Add specific missing documentation errors
        for missing in result.missing:
            errors.append(f"Missing docstring for: {missing}")

        return errors

    return validator


# ============================================================================
# Enhanced Complexity Validator
# ============================================================================


def validate_enhanced_complexity(
    code: str,
    max_complexity: int = 10,
    max_nesting: int = 4,
    max_function_length: int = 50,
) -> list[str]:
    """
    Validate code complexity beyond cyclomatic complexity.

    Args:
        code: Python code to validate
        max_complexity: Maximum cyclomatic complexity
        max_nesting: Maximum nesting depth
        max_function_length: Maximum function length in lines

    Returns:
        List of validation errors
    """
    errors = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["Syntax error prevents complexity validation"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check function length
            func_length = len(node.body)
            if func_length > max_function_length:
                errors.append(
                    f"Function '{node.name}' is too long ({func_length} statements, "
                    f"max {max_function_length}) (line {node.lineno})"
                )

            # Check nesting depth
            max_depth = _calculate_nesting_depth(node)
            if max_depth > max_nesting:
                errors.append(
                    f"Function '{node.name}' has excessive nesting (depth {max_depth}, "
                    f"max {max_nesting}) (line {node.lineno})"
                )

            # Check cyclomatic complexity
            complexity = _calculate_complexity(node)
            if complexity > max_complexity:
                errors.append(
                    f"Function '{node.name}' is too complex (complexity {complexity}, "
                    f"max {max_complexity}) (line {node.lineno})"
                )

    return errors


def _calculate_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Calculate maximum nesting depth in a node."""
    max_depth = current_depth

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            child_depth = _calculate_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _calculate_nesting_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)

    return max_depth


def _calculate_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity of a node."""
    complexity = 1  # Base complexity

    for child in ast.walk(node):
        # Each decision point adds 1 to complexity
        if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler)):
            complexity += 1

    return complexity


# ============================================================================
# Code Smell Detector
# ============================================================================


def validate_code_smells(code: str) -> list[str]:
    """
    Detect common code smells.

    Args:
        code: Python code to validate

    Returns:
        List of detected code smells
    """
    errors = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["Syntax error prevents code smell detection"]

    for node in ast.walk(tree):
        # Too many parameters
        if isinstance(node, ast.FunctionDef):
            param_count = len(node.args.args)
            if param_count > 5:
                errors.append(
                    f"Function '{node.name}' has too many parameters ({param_count}, max 5) (line {node.lineno})"
                )

        # Too many attributes in class
        elif isinstance(node, ast.ClassDef):
            attr_count = sum(1 for n in node.body if isinstance(n, ast.Assign))
            if attr_count > 10:
                errors.append(
                    f"Class '{node.name}' has too many attributes ({attr_count}, max 10) (line {node.lineno})"
                )

        # Bare except
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                errors.append(f"Bare 'except:' clause detected (line {node.lineno}). Specify exception type.")

        # Multiple statements on one line (detected via semicolons in source)
        # This would require source code analysis, skipping for now

    # Check for long lines (simple heuristic)
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        if len(line) > 100:
            errors.append(f"Line {i} is too long ({len(line)} characters, max 100)")

    return errors


# ============================================================================
# Best Practices Validator
# ============================================================================


def validate_best_practices(code: str) -> list[str]:
    """
    Validate Python best practices.

    Args:
        code: Python code to validate

    Returns:
        List of best practice violations
    """
    errors = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["Syntax error prevents best practices validation"]

    for node in ast.walk(tree):
        # Check for mutable default arguments
        if isinstance(node, ast.FunctionDef):
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    errors.append(f"Function '{node.name}' uses mutable default argument (line {node.lineno})")

        # Check for comparison to True/False/None
        elif isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value in (True, False, None):
                    errors.append(f"Comparison to {comparator.value!r} should use 'is' (line {node.lineno})")

        # Check for type() comparison instead of isinstance()
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "type":
            # Check if it's in a comparison
            errors.append(f"Use isinstance() instead of type() for type checking (line {node.lineno})")

    return errors


# ============================================================================
# Composite Validator
# ============================================================================


def create_quality_validator(
    check_naming: bool = True,
    check_documentation: bool = True,
    check_complexity: bool = True,
    check_smells: bool = True,
    check_best_practices: bool = True,
    min_doc_coverage: float = 0.8,
    max_complexity: int = 10,
) -> Callable[[str], list[str]]:
    """
    Create a composite quality validator.

    Args:
        check_naming: Enable naming convention checks
        check_documentation: Enable documentation checks
        check_complexity: Enable complexity checks
        check_smells: Enable code smell detection
        check_best_practices: Enable best practices checks
        min_doc_coverage: Minimum documentation coverage
        max_complexity: Maximum cyclomatic complexity

    Returns:
        Composite validator function
    """

    def validator(code: str) -> list[str]:
        """Validate code quality."""
        errors = []

        if check_naming:
            errors.extend(validate_naming_conventions(code))

        if check_documentation:
            doc_validator = create_documentation_validator(min_doc_coverage)
            errors.extend(doc_validator(code))

        if check_complexity:
            errors.extend(validate_enhanced_complexity(code, max_complexity=max_complexity))

        if check_smells:
            errors.extend(validate_code_smells(code))

        if check_best_practices:
            errors.extend(validate_best_practices(code))

        return errors

    return validator
