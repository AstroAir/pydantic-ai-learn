"""
End-to-End User Workflow Tests

Simulates real user interactions with the code agent to validate
complete workflows from a user perspective.

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

from code_agent.config import ConfigManager
from code_agent.tools import (
    CodeAnalyzer,
    CodeExecutor,
    CodeGenerator,
    ExecutionValidator,
    RefactoringEngine,
)
from code_agent.utils import LogFormat, LogLevel, StructuredLogger


class TestUserWorkflow1_CodeAnalysis:  # noqa: N801
    """Test Workflow 1: User analyzes code for quality and metrics."""

    def test_analyze_simple_function(self):
        """User analyzes a simple function."""
        analyzer = CodeAnalyzer()

        code = """
def calculate_sum(numbers):
    '''Calculate sum of numbers.'''
    total = 0
    for num in numbers:
        total += num
    return total
"""

        result = analyzer.analyze(code)

        assert result["valid"] is True
        assert "metrics" in result
        assert result["metrics"]["number_of_functions"] == 1
        assert result["metrics"]["lines_of_code"] > 0

    def test_analyze_complex_code(self):
        """User analyzes complex code with classes and functions."""
        analyzer = CodeAnalyzer()

        code = """
import math
from typing import List

class Calculator:
    '''Advanced calculator.'''

    def __init__(self):
        self.history = []

    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(('add', a, b, result))
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self.history.append(('multiply', a, b, result))
        return result

def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

        result = analyzer.analyze(code)

        assert result["valid"] is True
        assert result["metrics"]["number_of_classes"] == 1
        assert result["metrics"]["number_of_functions"] >= 3
        assert result["metrics"]["number_of_imports"] == 2
        assert "dependencies" in result

    def test_analyze_invalid_syntax(self):
        """User analyzes code with syntax errors."""
        analyzer = CodeAnalyzer()

        code = """
def broken_function(
    # Missing closing parenthesis
    return "broken"
"""

        result = analyzer.analyze(code)

        assert result["valid"] is False
        assert "error" in result
        assert "line" in result

    def test_check_complexity(self):
        """User checks code complexity."""
        analyzer = CodeAnalyzer()

        simple_code = "def test(): pass"
        medium_code = "\n".join([f"def func{i}(): pass" for i in range(30)])
        complex_code = "\n".join([f"def func{i}(): pass" for i in range(100)])

        assert analyzer.get_complexity(simple_code) == "low"
        assert analyzer.get_complexity(medium_code) in ["low", "medium"]
        assert analyzer.get_complexity(complex_code) in ["medium", "high"]


class TestUserWorkflow2_Refactoring:  # noqa: N801
    """Test Workflow 2: User gets refactoring suggestions."""

    def test_get_refactoring_suggestions(self):
        """User gets refactoring suggestions for code."""
        engine = RefactoringEngine()

        code = """
def process_data(data):
    # Long function that does too much
    result = []
    for item in data:
        if item > 0:
            if item < 100:
                if item % 2 == 0:
                    result.append(item * 2)
    return result
"""

        suggestions = engine.suggest_refactoring(code)

        # Should return a list (may be empty if no issues detected)
        assert isinstance(suggestions, list)

    def test_detect_code_smells(self):
        """User detects code smells."""
        engine = RefactoringEngine()

        # Code with potential issues
        code = """
def long_function():
    x = 1
    y = 2
    z = 3
    a = 4
    b = 5
    c = 6
    d = 7
    e = 8
    f = 9
    g = 10
    h = 11
    i = 12
    j = 13
    k = 14
    l = 15
    m = 16
    n = 17
    o = 18
    p = 19
    q = 20
    return x + y + z + a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q
"""

        # suggest_refactoring is the correct method
        suggestions = engine.suggest_refactoring(code)

        assert isinstance(suggestions, list)


class TestUserWorkflow3_CodeGeneration:  # noqa: N801
    """Test Workflow 3: User generates code templates."""

    def test_generate_function_stub(self):
        """User generates a function stub."""
        generator = CodeGenerator()

        result = generator.generate_function(
            name="calculate_average",
            parameters=["numbers"],
            return_type="float",
            docstring="Calculate average of numbers.",
        )

        assert "def calculate_average" in result.code
        assert "numbers" in result.code
        assert "float" in result.code

    def test_generate_class_template(self):
        """User generates a class template."""
        generator = CodeGenerator()

        result = generator.generate_class(
            name="DataProcessor", methods=["process", "validate"], docstring="Process and validate data."
        )

        assert "class DataProcessor" in result.code
        assert "def process" in result.code
        assert "def validate" in result.code

    def test_generate_test_template(self):
        """User generates a test template."""
        generator = CodeGenerator()

        # Correct method name is generate_test_template
        result = generator.generate_test_template(
            function_name="calculate_sum", test_cases=["test_positive_numbers", "test_negative_numbers"]
        )

        # The generator creates test methods with pattern test_{function_name}_case_{n}
        assert "test_calculate_sum_case_1" in result.code
        assert "test_calculate_sum_case_2" in result.code
        assert "calculate_sum" in result.code
        assert "pytest" in result.code


class TestUserWorkflow4_CodeExecution:  # noqa: N801
    """Test Workflow 4: User executes code safely."""

    def test_execute_simple_code(self):
        """User executes simple code."""
        executor = CodeExecutor()

        code = "x = 1 + 1"
        result = executor.execute(code)

        assert result.is_success()
        assert result.exit_code == 0

    def test_execute_code_with_output(self):
        """User executes code that produces output."""
        executor = CodeExecutor()

        code = "result = 2 + 2"
        result = executor.execute(code)

        assert result.is_success()

    def test_execute_code_with_error(self):
        """User executes code with errors."""
        executor = CodeExecutor()

        code = "x = 1 / 0"
        result = executor.execute(code)

        assert not result.is_success()
        assert result.error != ""


class TestUserWorkflow5_Validation:  # noqa: N801
    """Test Workflow 5: User validates code before execution."""

    def test_validate_safe_code(self):
        """User validates safe code."""
        validator = ExecutionValidator()

        code = "x = 1 + 1"
        result = validator.validate(code)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_unsafe_code(self):
        """User validates potentially unsafe code."""
        validator = ExecutionValidator()

        # Code with potential security issues
        code = "import os; os.system('rm -rf /')"
        result = validator.validate(code)

        # Should detect security issues
        assert not result.is_valid or len(result.errors) > 0


class TestUserWorkflow6_CompleteWorkflow:  # noqa: N801
    """Test Workflow 6: Complete user workflow combining multiple tools."""

    def test_analyze_refactor_generate_workflow(self):
        """User analyzes code, gets suggestions, and generates improved version."""
        # Step 1: Analyze existing code
        analyzer = CodeAnalyzer()
        code = """
def process(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""

        analysis = analyzer.analyze(code)
        assert analysis["valid"]

        # Step 2: Get refactoring suggestions
        engine = RefactoringEngine()
        suggestions = engine.suggest_refactoring(code)
        assert isinstance(suggestions, list)

        # Step 3: Generate improved version
        generator = CodeGenerator()
        improved = generator.generate_function(
            name="process_data", parameters=["data"], return_type="list", docstring="Process data efficiently."
        )
        assert "def process_data" in improved.code

    def test_validate_execute_workflow(self):
        """User validates code before executing it."""
        # Step 1: Validate code
        validator = ExecutionValidator()
        code = "result = sum([1, 2, 3, 4, 5])"

        validation = validator.validate(code)
        assert validation.is_valid

        # Step 2: Execute validated code
        executor = CodeExecutor()
        result = executor.execute(code)
        assert result.is_success()


class TestUserWorkflow7_Configuration:  # noqa: N801
    """Test Workflow 7: User configures the agent."""

    def test_load_configuration(self):
        """User loads configuration."""
        config = ConfigManager()

        # Load from dict
        config.load_from_dict({"model": "openai:gpt-4", "log_level": "INFO"})

        assert config.get("model") == "openai:gpt-4"
        assert config.get("log_level") == "INFO"

    def test_update_configuration(self):
        """User updates configuration."""
        config = ConfigManager()

        config.set("custom_setting", "value")
        assert config.get("custom_setting") == "value"


class TestUserWorkflow8_Logging:  # noqa: N801
    """Test Workflow 8: User configures logging."""

    def test_structured_logging(self):
        """User uses structured logging."""
        # Correct parameter name is format_type, not format
        logger = StructuredLogger(name="test_logger", level=LogLevel.INFO, format_type=LogFormat.HUMAN)

        # Should not raise errors
        logger.info("Test message")
        logger.debug("Debug message")
        logger.warning("Warning message")

    def test_json_logging(self):
        """User uses JSON logging."""
        # Correct parameter name is format_type, not format
        logger = StructuredLogger(name="test_logger", level=LogLevel.INFO, format_type=LogFormat.JSON)

        # Should not raise errors
        logger.info("Test message", extra={"key": "value"})
