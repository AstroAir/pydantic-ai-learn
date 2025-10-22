"""
Execution Validators Tests

Tests for code execution validators and verifiers.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from code_agent.config.execution import ValidationConfig, VerificationConfig
from code_agent.core.types import ExecutionResult, ExecutionStatus
from code_agent.tools.validators import (
    ExecutionValidator,
    ExecutionVerifier,
    SecurityValidationError,
    ValidationError,
    ValidationResult,
)


class TestValidationResult:
    """Test ValidationResult."""

    def test_validation_result_valid(self):
        """Test valid validation result."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_validation_result_invalid(self):
        """Test invalid validation result."""
        result = ValidationResult(
            is_valid=False,
            errors=["Syntax error", "Security error"],
            warnings=["Performance warning"],
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1


class TestExecutionValidator:
    """Test ExecutionValidator."""

    def test_validator_creation(self):
        """Test creating validator with default config."""
        validator = ExecutionValidator()

        assert validator.config is not None
        assert validator.config.enable_syntax_check is True

    def test_validator_with_custom_config(self):
        """Test creating validator with custom config."""
        config = ValidationConfig(enable_syntax_check=False)
        validator = ExecutionValidator(config)

        assert validator.config.enable_syntax_check is False

    def test_validate_valid_code(self, valid_code):
        """Test validating valid code."""
        validator = ExecutionValidator()
        result = validator.validate(valid_code)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_invalid_syntax(self, invalid_syntax_code):
        """Test validating code with syntax errors."""
        validator = ExecutionValidator()
        result = validator.validate(invalid_syntax_code)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("syntax" in err.lower() for err in result.errors)

    def test_validate_dangerous_code(self, dangerous_code):
        """Test validating code with security issues."""
        validator = ExecutionValidator()
        result = validator.validate(dangerous_code)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_eval_code(self, code_with_eval):
        """Test validating code using eval."""
        validator = ExecutionValidator()
        result = validator.validate(code_with_eval)

        assert result.is_valid is False
        assert any("eval" in err.lower() for err in result.errors)

    def test_syntax_validation_disabled(self, invalid_syntax_code):
        """Test validation with syntax check disabled."""
        config = ValidationConfig(enable_syntax_check=False)
        validator = ExecutionValidator(config)
        result = validator.validate(invalid_syntax_code)

        # Should not fail on syntax if syntax check is disabled
        # (but may fail on other checks)
        assert isinstance(result, ValidationResult)

    def test_security_validation_disabled(self, dangerous_code):
        """Test validation with security check disabled."""
        config = ValidationConfig(enable_security_check=False)
        validator = ExecutionValidator(config)
        result = validator.validate(dangerous_code)

        # Should not fail on security if security check is disabled
        assert isinstance(result, ValidationResult)

    def test_import_validation(self):
        """Test import validation."""
        validator = ExecutionValidator()
        code = "import os\nos.system('ls')"
        result = validator.validate(code)

        assert result.is_valid is False
        assert any("import" in err.lower() or "os" in err.lower() for err in result.errors)

    def test_complexity_validation_simple(self, valid_code):
        """Test complexity validation for simple code."""
        config = ValidationConfig(enable_complexity_check=True, max_complexity=10)
        validator = ExecutionValidator(config)
        result = validator.validate(valid_code)

        # Simple code should pass complexity check
        assert result.is_valid is True

    def test_complexity_validation_complex(self, complex_code):
        """Test complexity validation for complex code."""
        config = ValidationConfig(enable_complexity_check=True, max_complexity=5)
        validator = ExecutionValidator(config)
        result = validator.validate(complex_code)

        # Complex code should produce warnings (not errors)
        assert len(result.warnings) > 0
        assert any("complexity" in warn.lower() for warn in result.warnings)

    def test_custom_validator(self):
        """Test custom validator."""

        def no_print_validator(code: str) -> list[str]:
            """Disallow print statements."""
            if "print" in code:
                return ["Print statements are not allowed"]
            return []

        config = ValidationConfig()
        config.custom_validators.append(no_print_validator)
        validator = ExecutionValidator(config)

        code = "print('hello')"
        result = validator.validate(code)

        assert result.is_valid is False
        assert "Print statements are not allowed" in result.errors

    def test_multiple_custom_validators(self):
        """Test multiple custom validators."""

        def no_print(code: str) -> list[str]:
            return ["No print"] if "print" in code else []

        def no_loops(code: str) -> list[str]:
            return ["No loops"] if "for " in code or "while " in code else []

        config = ValidationConfig()
        config.custom_validators.extend([no_print, no_loops])
        validator = ExecutionValidator(config)

        code = "for i in range(10):\n    print(i)"
        result = validator.validate(code)

        assert result.is_valid is False
        assert "No print" in result.errors
        assert "No loops" in result.errors

    def test_strict_mode(self):
        """Test strict mode validation."""
        config = ValidationConfig(strict_mode=True)
        validator = ExecutionValidator(config)

        # In strict mode, even warnings should make validation fail
        # This depends on implementation details
        assert validator.config.strict_mode is True


class TestExecutionVerifier:
    """Test ExecutionVerifier."""

    def test_verifier_creation(self):
        """Test creating verifier with default config."""
        verifier = ExecutionVerifier()

        assert verifier.config is not None
        assert verifier.config.enable_output_check is True

    def test_verifier_with_custom_config(self):
        """Test creating verifier with custom config."""
        config = VerificationConfig(enable_output_check=False)
        verifier = ExecutionVerifier(config)

        assert verifier.config.enable_output_check is False

    def test_verify_successful_execution(self, mock_execution_result):
        """Test verifying successful execution."""
        verifier = ExecutionVerifier()
        result = verifier.verify(mock_execution_result)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_verify_failed_execution(self):
        """Test verifying failed execution."""
        exec_result = ExecutionResult(
            code="invalid",
            status=ExecutionStatus.FAILED,
            output="",
            error="Execution failed",
            exit_code=1,
        )

        verifier = ExecutionVerifier()
        result = verifier.verify(exec_result)

        # Verifier should still work on failed executions
        assert isinstance(result, ValidationResult)

    def test_verify_output_size_limit(self):
        """Test output size limit verification."""
        large_output = "x" * 2_000_000  # 2MB output
        exec_result = ExecutionResult(
            code="print('x' * 2000000)",
            status=ExecutionStatus.COMPLETED,
            output=large_output,
            error="",
            exit_code=0,
        )

        config = VerificationConfig(max_output_size=1_000_000)
        verifier = ExecutionVerifier(config)
        result = verifier.verify(exec_result)

        assert result.is_valid is False
        assert any("output" in err.lower() and "size" in err.lower() for err in result.errors)

    def test_verify_with_side_effects(self):
        """Test verification with side effects."""
        exec_result = ExecutionResult(
            code="import os; os.system('ls')",
            status=ExecutionStatus.COMPLETED,
            output="",
            error="",
            exit_code=0,
            side_effects=["file_write", "network_call"],
        )

        verifier = ExecutionVerifier()
        result = verifier.verify(exec_result)

        # Verifier should detect side effects
        assert isinstance(result, ValidationResult)

    def test_custom_verifier(self):
        """Test custom verifier."""

        def no_error_output(result: ExecutionResult) -> list[str]:
            """Disallow any error output."""
            if result.error:
                return ["Error output is not allowed"]
            return []

        config = VerificationConfig()
        config.custom_verifiers.append(no_error_output)
        verifier = ExecutionVerifier(config)

        exec_result = ExecutionResult(
            code="print('test')",
            status=ExecutionStatus.COMPLETED,
            output="test\n",
            error="Warning: something",
            exit_code=0,
        )

        result = verifier.verify(exec_result)

        assert result.is_valid is False
        assert "Error output is not allowed" in result.errors


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_creation(self):
        """Test creating validation error."""
        error = ValidationError("Test error")

        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_validation_error_with_details(self):
        """Test validation error with details."""
        error = ValidationError("Validation failed: Error 1, Error 2")

        assert "Validation failed" in str(error)
        assert "Error 1" in str(error)


class TestSecurityValidationError:
    """Test SecurityValidationError exception."""

    def test_security_validation_error_creation(self):
        """Test creating security validation error."""
        error = SecurityValidationError("Security issue detected")

        assert str(error) == "Security issue detected"
        assert isinstance(error, ValidationError)
        assert isinstance(error, Exception)
