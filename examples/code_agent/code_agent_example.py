"""
Code Agent Usage Examples

Demonstrates the capabilities of the CodeAgent for autonomous code analysis,
refactoring, and generation tasks.

Examples:
1. Basic code analysis
2. Pattern detection and code smells
3. Refactoring suggestions
4. Code generation
5. Comprehensive code review workflow

Run with: python examples/code_agent/code_agent_example.py

Author: The Augster
Python Version: 3.12+
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from code_agent import CodeAgent, create_code_agent

# ============================================================================
# Example 1: Basic Code Analysis
# ============================================================================


def example_basic_analysis():
    """Example: Analyze code structure and metrics."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Code Analysis")
    print("=" * 60)

    # Create agent
    agent = create_code_agent()

    # Analyze a file
    result = agent.run_sync("Analyze the code structure in tools/task_planning_toolkit.py")

    print("\nAnalysis Result:")
    print(result.output)


# ============================================================================
# Example 2: Pattern Detection
# ============================================================================


def example_pattern_detection():
    """Example: Detect code smells and anti-patterns."""
    print("\n" + "=" * 60)
    print("Example 2: Pattern Detection")
    print("=" * 60)

    agent = create_code_agent()

    # Detect patterns in a file
    result = agent.run_sync("Detect code smells in tools/code_agent_toolkit.py with medium severity threshold")

    print("\nPattern Detection Result:")
    print(result.output)


# ============================================================================
# Example 3: Refactoring Suggestions
# ============================================================================


def example_refactoring_suggestions():
    """Example: Get refactoring suggestions."""
    print("\n" + "=" * 60)
    print("Example 3: Refactoring Suggestions")
    print("=" * 60)

    agent = create_code_agent()

    # Get refactoring suggestions
    result = agent.run_sync("Suggest refactoring opportunities for tools/filesystem_tools.py")

    print("\nRefactoring Suggestions:")
    print(result.output)


# ============================================================================
# Example 4: Code Generation
# ============================================================================


def example_code_generation():
    """Example: Generate code from description."""
    print("\n" + "=" * 60)
    print("Example 4: Code Generation")
    print("=" * 60)

    agent = create_code_agent()

    # Generate a function
    result = agent.run_sync(
        "Generate a function that validates email addresses using regex, with type hints and comprehensive docstring"
    )

    print("\nGenerated Code:")
    print(result.output)


# ============================================================================
# Example 5: Comprehensive Code Review
# ============================================================================


def example_comprehensive_review():
    """Example: Perform comprehensive code review."""
    print("\n" + "=" * 60)
    print("Example 5: Comprehensive Code Review")
    print("=" * 60)

    agent = create_code_agent()

    # Comprehensive review
    result = agent.run_sync(
        "Perform a comprehensive code review of tools/bash_tool.py including:\n"
        "1. Code structure analysis\n"
        "2. Quality metrics\n"
        "3. Pattern detection\n"
        "4. Refactoring suggestions\n"
        "Provide a summary with prioritized recommendations."
    )

    print("\nCode Review:")
    print(result.output)


# ============================================================================
# Example 6: Dependency Analysis
# ============================================================================


def example_dependency_analysis():
    """Example: Analyze code dependencies."""
    print("\n" + "=" * 60)
    print("Example 6: Dependency Analysis")
    print("=" * 60)

    agent = create_code_agent()

    # Analyze dependencies
    result = agent.run_sync("Analyze the dependencies in code_agent.py and categorize them")

    print("\nDependency Analysis:")
    print(result.output)


# ============================================================================
# Example 7: Syntax Validation
# ============================================================================


def example_syntax_validation():
    """Example: Validate Python syntax."""
    print("\n" + "=" * 60)
    print("Example 7: Syntax Validation")
    print("=" * 60)

    agent = create_code_agent()

    # Validate syntax
    result = agent.run_sync("Validate the syntax of tools/code_agent_toolkit.py with strict mode")

    print("\nSyntax Validation:")
    print(result.output)


# ============================================================================
# Example 8: Multi-File Analysis
# ============================================================================


def example_multi_file_analysis():
    """Example: Analyze multiple files."""
    print("\n" + "=" * 60)
    print("Example 8: Multi-File Analysis")
    print("=" * 60)

    agent = create_code_agent()

    # Analyze multiple files
    result = agent.run_sync(
        "Compare the code quality metrics between:\n"
        "1. tools/task_planning_toolkit.py\n"
        "2. tools/filesystem_tools.py\n"
        "3. tools/file_editing_toolkit.py\n"
        "Identify which has the best code quality and why."
    )

    print("\nMulti-File Analysis:")
    print(result.output)


# ============================================================================
# Example 9: Code Improvement Workflow
# ============================================================================


async def example_improvement_workflow():
    """Example: Complete code improvement workflow."""
    print("\n" + "=" * 60)
    print("Example 9: Code Improvement Workflow")
    print("=" * 60)

    agent = create_code_agent()

    # Step 1: Analyze
    print("\nStep 1: Analyzing code...")
    result1 = await agent.run("Analyze tools/code_agent_toolkit.py and identify the top 3 areas for improvement")
    print(result1.output)

    # Step 2: Suggest refactoring
    print("\nStep 2: Getting refactoring suggestions...")
    result2 = await agent.run("Based on the analysis, suggest specific refactoring steps for the top issue")
    print(result2.output)

    # Step 3: Generate improved code
    print("\nStep 3: Generating improved code example...")
    result3 = await agent.run("Generate an example of how to refactor the most complex function")
    print(result3.output)


# ============================================================================
# Example 10: Direct Tool Usage
# ============================================================================


def example_direct_tool_usage():
    """Example: Use code analysis tools directly."""
    print("\n" + "=" * 60)
    print("Example 10: Direct Tool Usage")
    print("=" * 60)

    from tools.code_agent_toolkit import (
        AnalyzeCodeInput,
        CodeAgentState,
        analyze_code,
        calculate_metrics,
    )

    state = CodeAgentState()

    # Direct analysis
    print("\nDirect Analysis:")
    result = analyze_code(AnalyzeCodeInput(file_path="tools/task_planning_toolkit.py", analysis_type="full"), state)
    print(result)

    # Direct metrics
    print("\nDirect Metrics:")
    metrics = calculate_metrics("tools/filesystem_tools.py", state)
    print(metrics)


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CODE AGENT EXAMPLES")
    print("=" * 60)

    # Run synchronous examples
    try:
        example_basic_analysis()
    except Exception as e:
        print(f"Error in basic analysis: {e}")

    try:
        example_pattern_detection()
    except Exception as e:
        print(f"Error in pattern detection: {e}")

    try:
        example_refactoring_suggestions()
    except Exception as e:
        print(f"Error in refactoring suggestions: {e}")

    try:
        example_code_generation()
    except Exception as e:
        print(f"Error in code generation: {e}")

    try:
        example_dependency_analysis()
    except Exception as e:
        print(f"Error in dependency analysis: {e}")

    try:
        example_syntax_validation()
    except Exception as e:
        print(f"Error in syntax validation: {e}")

    try:
        example_direct_tool_usage()
    except Exception as e:
        print(f"Error in direct tool usage: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
