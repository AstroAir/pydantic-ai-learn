"""
Custom Workflows Example

Demonstrates how to use custom tools and workflows for code quality analysis.

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations


def example_1_code_formatter() -> None:
    """Example 1: Using CodeFormatter."""
    print("\n" + "=" * 60)
    print("Example 1: Code Formatting")
    print("=" * 60)

    from code_agent.tools.custom import CodeFormatter

    formatter = CodeFormatter()

    # Unformatted code
    code = """
def hello(  ):
    x=1+2
    y=3+4
    return   x+y
"""

    print("\nOriginal code:")
    print(code)

    # Format code
    result = formatter.format(code)

    print("\nFormatted code:")
    print(result.code)

    print(f"\nCode changed: {result.changed}")
    print(f"Backend used: {result.backend_used}")

    if result.diff:
        print("\nDiff:")
        print(result.diff)


def example_2_code_linter() -> None:
    """Example 2: Using CodeLinter."""
    print("\n" + "=" * 60)
    print("Example 2: Code Linting")
    print("=" * 60)

    from code_agent.tools.custom import CodeLinter

    linter = CodeLinter()

    # Code with issues
    code = """
import os
import sys

def test():
    x = 1
    y = 2
"""

    print("\nCode to lint:")
    print(code)

    # Lint code
    result = linter.lint(code)

    print("\nLinting results:")
    print(f"  Errors: {result.error_count}")
    print(f"  Warnings: {result.warning_count}")
    print(f"  Info: {result.info_count}")

    if result.issues:
        print("\nIssues found:")
        for issue in result.issues[:5]:  # Show first 5
            print(f"  Line {issue.line}: [{issue.severity}] {issue.message}")


def example_3_dependency_analyzer() -> None:
    """Example 3: Using DependencyAnalyzer."""
    print("\n" + "=" * 60)
    print("Example 3: Dependency Analysis")
    print("=" * 60)

    from code_agent.tools.custom import DependencyAnalyzer

    analyzer = DependencyAnalyzer()

    # Code with dependencies
    code = """
import os
import sys
from pathlib import Path
from typing import Any, List

def test():
    path = Path(".")
    return list(path.iterdir())
"""

    print("\nCode to analyze:")
    print(code)

    # Analyze dependencies
    result = analyzer.analyze(code)

    print("\nDependency analysis:")
    print(f"  Total imports: {result.total_imports}")
    print(f"  Stdlib imports: {len(result.stdlib_imports)}")
    print(f"  Third-party imports: {len(result.third_party_imports)}")
    print(f"  Local imports: {len(result.local_imports)}")

    print("\nStdlib imports:")
    for imp in result.stdlib_imports:
        print(f"  - {imp}")

    if result.issues:
        print("\nIssues:")
        for issue in result.issues:
            print(f"  {issue.type}: {issue.message}")


def example_4_documentation_analyzer() -> None:
    """Example 4: Using DocumentationAnalyzer."""
    print("\n" + "=" * 60)
    print("Example 4: Documentation Analysis")
    print("=" * 60)

    from code_agent.tools.custom import DocumentationAnalyzer

    analyzer = DocumentationAnalyzer()

    # Code with documentation
    code = """
'''Module docstring.'''

class Calculator:
    '''A simple calculator class.'''

    def add(self, a: int, b: int) -> int:
        '''
        Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            int: Sum of a and b
        '''
        return a + b

    def subtract(self, a, b):
        return a - b
"""

    print("\nCode to analyze:")
    print(code)

    # Analyze documentation
    result = analyzer.analyze(code)

    print("\nDocumentation analysis:")
    print(f"  Coverage: {result.coverage:.1%}")
    print(f"  Documented: {result.documented_items}/{result.total_items}")

    if result.missing:
        print("\nMissing documentation:")
        for missing in result.missing:
            print(f"  - {missing}")

    if result.issues:
        print("\nIssues:")
        for issue in result.issues[:5]:
            print(f"  - {issue}")


def example_5_quality_workflow() -> None:
    """Example 5: Using QualityWorkflow."""
    print("\n" + "=" * 60)
    print("Example 5: Quality Workflow")
    print("=" * 60)

    from code_agent.workflows import QualityWorkflow

    workflow = QualityWorkflow()

    # Code to analyze
    code = """
def calculate(x,y):
    result=x+y
    return result
"""

    print("\nCode to analyze:")
    print(code)

    # Run quality workflow
    print("\nRunning quality workflow...")
    report = workflow.run(code)  # type: ignore[arg-type]  # Legacy interface returns QualityReport

    print("\nWorkflow results:")
    print(f"  Success: {report.success}")
    print(f"  Total steps: {report.summary['total_steps']}")  # type: ignore[attr-defined]
    print(f"  Successful steps: {report.summary['successful_steps']}")  # type: ignore[attr-defined]
    print(f"  Total errors: {report.total_errors}")  # type: ignore[attr-defined]
    print(f"  Total warnings: {report.total_warnings}")  # type: ignore[attr-defined]
    print(f"  Total duration: {report.total_duration:.2f}s")  # type: ignore[attr-defined]

    print("\nSteps executed:")
    for step in report.steps:  # type: ignore[attr-defined]
        status = "✓" if step.success else "✗"
        print(f"  {status} {step.name} ({step.duration:.2f}s)")
        if step.errors:
            for error in step.errors[:2]:
                print(f"      Error: {error}")
        if step.warnings:
            for warning in step.warnings[:2]:
                print(f"      Warning: {warning}")

    if report.recommendations:  # type: ignore[attr-defined]
        print("\nRecommendations:")
        for rec in report.recommendations[:5]:  # type: ignore[attr-defined]
            print(f"  - {rec}")


def example_6_custom_validators() -> None:
    """Example 6: Using custom validators."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Validators")
    print("=" * 60)

    from code_agent.tools.custom.validators import (
        validate_best_practices,
        validate_code_smells,
        validate_naming_conventions,
    )

    # Code with issues
    code = """
def MyFunction(param1, param2, param3, param4, param5, param6):
    if param1 == True:
        try:
            result = param2 + param3
        except:
            pass
    return result
"""

    print("\nCode to validate:")
    print(code)

    # Validate naming conventions
    print("\nNaming convention errors:")
    errors = validate_naming_conventions(code)
    for error in errors:
        print(f"  - {error}")

    # Validate code smells
    print("\nCode smells:")
    errors = validate_code_smells(code)
    for error in errors:
        print(f"  - {error}")

    # Validate best practices
    print("\nBest practice violations:")
    errors = validate_best_practices(code)
    for error in errors:
        print(f"  - {error}")


def example_7_complete_workflow() -> None:
    """Example 7: Complete workflow with all tools."""
    print("\n" + "=" * 60)
    print("Example 7: Complete Workflow")
    print("=" * 60)

    from code_agent.tools.custom import (
        CodeFormatter,
        CodeLinter,
        DependencyAnalyzer,
        DocumentationAnalyzer,
    )

    # Original code
    code = """
import os
def process(data):
    result=[]
    for item in data:
        result.append(item*2)
    return result
"""

    print("\nOriginal code:")
    print(code)

    # Step 1: Format
    print("\n1. Formatting...")
    formatter = CodeFormatter()
    format_result = formatter.format(code)
    code = format_result.code
    print(f"   Changed: {format_result.changed}")

    # Step 2: Lint
    print("\n2. Linting...")
    linter = CodeLinter()
    lint_result = linter.lint(code)
    print(f"   Errors: {lint_result.error_count}")
    print(f"   Warnings: {lint_result.warning_count}")

    # Step 3: Analyze dependencies
    print("\n3. Analyzing dependencies...")
    dep_analyzer = DependencyAnalyzer()
    dep_result = dep_analyzer.analyze(code)
    print(f"   Total imports: {dep_result.total_imports}")

    # Step 4: Analyze documentation
    print("\n4. Analyzing documentation...")
    doc_analyzer = DocumentationAnalyzer()
    doc_result = doc_analyzer.analyze(code)
    print(f"   Coverage: {doc_result.coverage:.1%}")

    print("\nFinal code:")
    print(code)


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Custom Workflows Examples")
    print("=" * 60)

    try:
        example_1_code_formatter()
        example_2_code_linter()
        example_3_dependency_analyzer()
        example_4_documentation_analyzer()
        example_5_quality_workflow()
        example_6_custom_validators()
        example_7_complete_workflow()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
