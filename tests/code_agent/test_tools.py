"""
Tools Tests

Tests for code analysis and manipulation tools.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import pytest

from code_agent.tools import CodeAnalyzer, CodeGenerator, RefactoringEngine


class TestCodeAnalyzer:
    """Test CodeAnalyzer."""

    def test_analyze_valid_code(self):
        """Test analyzing valid code."""
        analyzer = CodeAnalyzer()
        code = """
def hello(name):
    return f"Hello, {name}!"

class Greeter:
    def greet(self, name):
        return hello(name)
"""
        result = analyzer.analyze(code)

        assert result["valid"] is True
        assert "metrics" in result
        assert "patterns" in result
        assert "dependencies" in result

    def test_analyze_invalid_code(self):
        """Test analyzing invalid code."""
        analyzer = CodeAnalyzer()
        code = "def invalid syntax here"
        result = analyzer.analyze(code)

        assert result["valid"] is False
        assert "error" in result

    def test_validate_syntax_valid(self):
        """Test syntax validation for valid code."""
        analyzer = CodeAnalyzer()
        code = "x = 1 + 2"

        is_valid, error = analyzer.validate_syntax(code)

        assert is_valid is True
        assert error is None

    def test_validate_syntax_invalid(self):
        """Test syntax validation for invalid code."""
        analyzer = CodeAnalyzer()
        code = "x = 1 +"

        is_valid, error = analyzer.validate_syntax(code)

        assert is_valid is False
        assert error is not None

    def test_get_complexity_low(self):
        """Test complexity detection for simple code."""
        analyzer = CodeAnalyzer()
        code = "x = 1"

        complexity = analyzer.get_complexity(code)

        assert complexity == "low"

    def test_get_complexity_high(self):
        """Test complexity detection for complex code."""
        analyzer = CodeAnalyzer()
        code = "\n".join([f"x{i} = {i}" for i in range(100)])

        complexity = analyzer.get_complexity(code)

        assert complexity in ["medium", "high"]


class TestRefactoringEngine:
    """Test RefactoringEngine."""

    def test_suggest_refactoring_empty(self):
        """Test refactoring suggestions for simple code."""
        engine = RefactoringEngine()
        code = "x = 1"

        suggestions = engine.suggest_refactoring(code)

        assert isinstance(suggestions, list)

    def test_suggest_refactoring_long_function(self):
        """Test detecting long functions."""
        engine = RefactoringEngine()
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
    r = 21
    return r
"""
        suggestions = engine.suggest_refactoring(code)

        # Should detect long function
        assert any("Long Function" in s.title for s in suggestions)

    def test_improvement_score(self):
        """Test improvement score calculation."""
        engine = RefactoringEngine()
        code = "x = 1"

        score = engine.get_improvement_score(code)

        assert 0 <= score <= 100


class TestCodeGenerator:
    """Test CodeGenerator."""

    def test_generate_function(self):
        """Test function generation."""
        generator = CodeGenerator()

        result = generator.generate_function(
            name="test_func", parameters=["x", "y"], return_type="int", docstring="Test function"
        )

        assert "def test_func" in result.code
        assert "x, y" in result.code
        assert "-> int" in result.code

    def test_generate_class(self):
        """Test class generation."""
        generator = CodeGenerator()

        result = generator.generate_class(
            name="TestClass",
            attributes={"name": "str", "value": "int"},
            methods=["get_name", "set_value"],
            docstring="Test class",
        )

        assert "class TestClass" in result.code
        assert "self.name" in result.code
        assert "def get_name" in result.code

    def test_generate_test_template(self):
        """Test test template generation."""
        generator = CodeGenerator()

        result = generator.generate_test_template(function_name="test_func", test_cases=["case1", "case2"])

        assert "import pytest" in result.code
        assert "def test_test_func" in result.code
        assert "pytest" in result.dependencies

    def test_generate_docstring(self):
        """Test docstring generation."""
        generator = CodeGenerator()

        docstring = generator.generate_docstring(
            name="test_func", description="Test function", parameters=["x", "y"], returns="int"
        )

        assert "Test function" in docstring
        assert "Args:" in docstring
        assert "Returns:" in docstring


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
