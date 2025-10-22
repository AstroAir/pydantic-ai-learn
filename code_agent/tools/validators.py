"""
Code Execution Validators

Pre-execution validation and post-execution verification for code execution.

Features:
- Syntax validation
- Security validation (imports, builtins, dangerous patterns)
- Complexity validation
- Import validation
- Output verification
- Side effect detection

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.execution import ValidationConfig, VerificationConfig

from ..core.types import ExecutionResult

# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(Exception):
    """Base class for validation errors."""

    pass


class SyntaxValidationError(ValidationError):
    """Raised when syntax validation fails."""

    pass


class SecurityValidationError(ValidationError):
    """Raised when security validation fails."""

    pass


class ComplexityValidationError(ValidationError):
    """Raised when complexity validation fails."""

    pass


class ImportValidationError(ValidationError):
    """Raised when import validation fails."""

    pass


# ============================================================================
# Validation Results
# ============================================================================


@dataclass
class ValidationResult:
    """Result of code validation."""

    is_valid: bool
    """Whether code is valid"""

    errors: list[str] = field(default_factory=list)
    """Validation errors"""

    warnings: list[str] = field(default_factory=list)
    """Validation warnings"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""


# ============================================================================
# Pre-Execution Validator
# ============================================================================


class ExecutionValidator:
    """
    Pre-execution code validator.

    Validates code before execution to ensure safety and correctness.
    """

    def __init__(self, config: ValidationConfig | None = None):
        """
        Initialize validator.

        Args:
            config: Validation configuration
        """
        # Import at runtime to avoid circular import
        if config is None:
            from ..config.execution import ValidationConfig

            config = ValidationConfig()

        self.config = config

    def validate(self, code: str) -> ValidationResult:
        """
        Validate code before execution.

        Args:
            code: Code to validate

        Returns:
            Validation result
        """
        errors: list[str] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        # Syntax validation
        if self.config.enable_syntax_check:
            syntax_errors = self._validate_syntax(code)
            errors.extend(syntax_errors)

        # Security validation
        if self.config.enable_security_check:
            security_errors = self._validate_security(code)
            errors.extend(security_errors)

        # Import validation
        if self.config.enable_import_check:
            import_errors = self._validate_imports(code)
            errors.extend(import_errors)

        # Complexity validation
        if self.config.enable_complexity_check:
            complexity_warnings = self._validate_complexity(code)
            warnings.extend(complexity_warnings)

        # Custom validators
        for validator in self.config.custom_validators:
            try:
                custom_errors = validator(code)
                errors.extend(custom_errors)
            except Exception as e:
                errors.append(f"Custom validator error: {e}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings, metadata=metadata)

    def _validate_syntax(self, code: str) -> list[str]:
        """Validate Python syntax."""
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Syntax validation error: {e}")
        return errors

    def _validate_security(self, code: str) -> list[str]:
        """Validate security concerns."""
        errors = []

        # Check for dangerous patterns
        dangerous_patterns = [
            (r"\beval\s*\(", "Use of 'eval' is not allowed"),
            (r"\bexec\s*\(", "Use of 'exec' is not allowed"),
            (r"\bcompile\s*\(", "Use of 'compile' is not allowed"),
            (r"\b__import__\s*\(", "Use of '__import__' is not allowed"),
            (r"\bopen\s*\(", "Use of 'open' may be restricted"),
            (r"\bos\.system\s*\(", "Use of 'os.system' is not allowed"),
            (r"\bsubprocess\.", "Use of 'subprocess' is not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                errors.append(message)

        return errors

    def _validate_imports(self, code: str) -> list[str]:
        """Validate imports."""
        errors = []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._is_blocked_import(alias.name):
                            errors.append(f"Import '{alias.name}' is not allowed")
                elif isinstance(node, ast.ImportFrom) and node.module and self._is_blocked_import(node.module):
                    errors.append(f"Import from '{node.module}' is not allowed")
        except Exception as e:
            errors.append(f"Import validation error: {e}")

        return errors

    def _is_blocked_import(self, module_name: str) -> bool:
        """Check if import is blocked."""
        # Default blocked imports
        blocked = [
            "os",
            "subprocess",
            "sys",
            "socket",
            "urllib",
            "requests",
            "http",
            "ftplib",
            "telnetlib",
            "smtplib",
            "poplib",
            "imaplib",
        ]
        return module_name in blocked or module_name.split(".")[0] in blocked

    def _validate_complexity(self, code: str) -> list[str]:
        """Validate code complexity."""
        warnings = []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function length
                    func_length = len(node.body)
                    if func_length > self.config.max_function_length:
                        warnings.append(
                            f"Function '{node.name}' is too long "
                            f"({func_length} > {self.config.max_function_length} lines)"
                        )

                    # Check complexity (simplified)
                    complexity = self._calculate_complexity(node)
                    if complexity > self.config.max_complexity:
                        warnings.append(
                            f"Function '{node.name}' is too complex "
                            f"(complexity {complexity} > {self.config.max_complexity})"
                        )
        except Exception as e:
            warnings.append(f"Complexity validation error: {e}")

        return warnings

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity


# ============================================================================
# Post-Execution Verifier
# ============================================================================


class ExecutionVerifier:
    """
    Post-execution result verifier.

    Verifies execution results for correctness and safety.
    """

    def __init__(self, config: VerificationConfig | None = None):
        """
        Initialize verifier.

        Args:
            config: Verification configuration
        """
        # Import at runtime to avoid circular import
        if config is None:
            from ..config.execution import VerificationConfig

            config = VerificationConfig()

        self.config = config

    def verify(self, result: ExecutionResult) -> ValidationResult:
        """
        Verify execution result.

        Args:
            result: Execution result to verify

        Returns:
            Verification result
        """
        errors: list[str] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        # Output validation
        if self.config.enable_output_check:
            output_errors = self._verify_output(result)
            errors.extend(output_errors)

        # Side effect detection
        if self.config.enable_side_effect_check:
            side_effect_warnings = self._verify_side_effects(result)
            warnings.extend(side_effect_warnings)

        # Custom verifiers
        for verifier in self.config.custom_verifiers:
            try:
                custom_errors = verifier(result)
                errors.extend(custom_errors)
            except Exception as e:
                errors.append(f"Custom verifier error: {e}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings, metadata=metadata)

    def _verify_output(self, result: ExecutionResult) -> list[str]:
        """Verify output size and content."""
        errors = []

        # Check output size
        total_output = len(result.output) + len(result.stdout) + len(result.stderr)
        if total_output > self.config.max_output_size:
            errors.append(f"Output size ({total_output} bytes) exceeds limit ({self.config.max_output_size} bytes)")

        return errors

    def _verify_side_effects(self, result: ExecutionResult) -> list[str]:
        """Verify side effects."""
        warnings = []

        # Check for unexpected side effects
        for side_effect in result.side_effects:
            if side_effect not in self.config.allowed_side_effects:
                warnings.append(f"Unexpected side effect detected: {side_effect}")

        return warnings
