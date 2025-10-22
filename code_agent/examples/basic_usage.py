"""
Basic Usage Examples

Demonstrates basic usage of the code agent.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from code_agent.core import create_code_agent
from code_agent.tools import CodeAnalyzer, CodeGenerator, RefactoringEngine
from code_agent.utils import LogFormat, LogLevel, StructuredLogger


def example_basic_agent() -> None:
    """Example: Create and use a basic code agent."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Code Agent")
    print("=" * 60)

    # Create agent
    agent = create_code_agent(model="openai:gpt-4")

    # Get agent state
    state = agent.get_state()
    print(f"Agent state: {state}")

    # Get usage summary
    usage = agent.get_usage_summary()
    print(f"Usage: {usage}")


def example_code_analyzer() -> None:
    """Example: Analyze code using CodeAnalyzer."""
    print("\n" + "=" * 60)
    print("Example 2: Code Analysis")
    print("=" * 60)

    analyzer = CodeAnalyzer()

    code = """
def calculate_sum(numbers):
    '''Calculate sum of numbers.'''
    total = 0
    for num in numbers:
        total += num
    return total

class Calculator:
    def add(self, a, b):
        return a + b
"""

    result = analyzer.analyze(code)

    print(f"Valid: {result['valid']}")
    print(f"Metrics: {result['metrics']}")
    print(f"Patterns: {result['patterns']}")
    print(f"Dependencies: {result['dependencies']}")

    # Check complexity
    complexity = analyzer.get_complexity(code)
    print(f"Complexity: {complexity}")


def example_refactoring_suggestions() -> None:
    """Example: Get refactoring suggestions."""
    print("\n" + "=" * 60)
    print("Example 3: Refactoring Suggestions")
    print("=" * 60)

    engine = RefactoringEngine()

    code = """
def process_data(data):
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

    print(f"Found {len(suggestions)} suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion.title}: {suggestion.description}")

    # Get improvement score
    score = engine.get_improvement_score(code)
    print(f"Improvement score: {score:.1f}/100")


def example_code_generation() -> None:
    """Example: Generate code."""
    print("\n" + "=" * 60)
    print("Example 4: Code Generation")
    print("=" * 60)

    generator = CodeGenerator()

    # Generate function
    func_result = generator.generate_function(
        name="calculate_average", parameters=["numbers"], return_type="float", docstring="Calculate average of numbers"
    )

    print("Generated function:")
    print(func_result.code)

    # Generate class
    class_result = generator.generate_class(
        name="DataProcessor",
        attributes={"data": "list", "result": "Any"},
        methods=["process", "validate"],
        docstring="Process data",
    )

    print("\nGenerated class:")
    print(class_result.code)

    # Generate test template
    test_result = generator.generate_test_template(
        function_name="calculate_average", test_cases=["empty list", "single element", "multiple elements"]
    )

    print("\nGenerated test template:")
    print(test_result.code)


def example_logging() -> None:
    """Example: Use structured logging."""
    print("\n" + "=" * 60)
    print("Example 5: Structured Logging")
    print("=" * 60)

    # Create logger
    logger = StructuredLogger(
        name="example",
        level=LogLevel.INFO,
        format_type=LogFormat.HUMAN,
    )

    # Log messages
    logger.info("Application started")
    logger.debug("Debug information", user_id=123)
    logger.warning("Warning message", severity="high")

    # Track operation
    metrics = logger.start_operation("data_processing", items=1000)
    # ... do work ...
    metrics.input_tokens = 500
    metrics.output_tokens = 300
    logger.complete_operation(metrics, success=True)

    # Get metrics summary
    summary = logger.get_metrics_summary()
    print(f"\nMetrics summary: {summary}")


def example_error_handling() -> None:
    """Example: Error handling and recovery."""
    print("\n" + "=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)

    from code_agent.utils import (
        CircuitBreaker,
        ErrorCategory,
        ErrorContext,
        ErrorDiagnosisEngine,
        ErrorSeverity,
    )

    # Create error context
    context = ErrorContext(
        error_type="FileNotFoundError",
        error_message="File not found: config.json",
        category=ErrorCategory.PERMANENT,
        severity=ErrorSeverity.HIGH,
    )

    # Diagnose error
    suggestions = ErrorDiagnosisEngine.diagnose(context)
    print(f"Error: {context.error_type}")
    print("Suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")

    # Circuit breaker example
    breaker = CircuitBreaker(name="api_call", failure_threshold=3)

    def api_call():
        return "success"

    try:
        result = breaker.call(api_call)
        print(f"\nAPI call result: {result}")
    except Exception as e:
        print(f"API call failed: {e}")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Code Agent - Basic Usage Examples")
    print("=" * 60)

    example_basic_agent()
    example_code_analyzer()
    example_refactoring_suggestions()
    example_code_generation()
    example_logging()
    example_error_handling()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
