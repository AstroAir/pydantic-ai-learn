# Custom Tools for Code Agent

This module provides custom tools that extend the Code Agent's capabilities with advanced code quality, formatting, linting, dependency analysis, and documentation features.

## Overview

The custom tools module includes:

1. **CodeFormatter** - Automatic code formatting using black or ruff
2. **CodeLinter** - Code linting and issue detection using ruff
3. **DependencyAnalyzer** - Import and dependency analysis
4. **DocumentationAnalyzer** - Documentation coverage and quality analysis
5. **Custom Validators** - Additional validation rules for code quality

## Installation

The custom tools require additional dependencies:

```bash
# Install with custom tools support
pip install black ruff

# Or just ruff (which can also format)
pip install ruff
```

## Quick Start

### CodeFormatter

Format Python code automatically:

```python
from code_agent.tools.custom import CodeFormatter

formatter = CodeFormatter()

# Format code
code = """
def hello(  ):
    x=1+2
    return   x
"""

result = formatter.format(code)
print(result.code)  # Formatted code
print(result.changed)  # True if code was changed
print(result.diff)  # Unified diff showing changes

# Check if code is formatted
is_formatted = formatter.check(code)
print(f"Code is formatted: {is_formatted}")
```

### CodeLinter

Lint Python code and detect issues:

```python
from code_agent.tools.custom import CodeLinter

linter = CodeLinter()

# Lint code
code = """
import os
import sys

def test():
    x = 1
    y = 2
"""

result = linter.lint(code)
print(f"Found {result.error_count} errors")
print(f"Found {result.warning_count} warnings")

for issue in result.issues:
    print(f"{issue.severity}: {issue.message} (line {issue.line})")

# Check if code is clean
is_clean = linter.check(code)
print(f"Code is clean: {is_clean}")
```

### DependencyAnalyzer

Analyze imports and dependencies:

```python
from code_agent.tools.custom import DependencyAnalyzer

analyzer = DependencyAnalyzer()

# Analyze dependencies
code = """
import os
import sys
from pathlib import Path
from typing import Any, List
import requests

def test():
    pass
"""

result = analyzer.analyze(code)
print(f"Total imports: {result.total_imports}")
print(f"Stdlib imports: {result.stdlib_imports}")
print(f"Third-party imports: {result.third_party_imports}")
print(f"Local imports: {result.local_imports}")

for issue in result.issues:
    print(f"{issue.type}: {issue.message}")
```

### DocumentationAnalyzer

Analyze documentation coverage:

```python
from code_agent.tools.custom import DocumentationAnalyzer

analyzer = DocumentationAnalyzer()

# Analyze documentation
code = """
class MyClass:
    '''Class docstring.'''

    def my_method(self, param: str) -> int:
        '''
        Method docstring.

        Args:
            param: Parameter description

        Returns:
            int: Return value
        '''
        return 42
"""

result = analyzer.analyze(code)
print(f"Documentation coverage: {result.coverage:.1%}")
print(f"Documented items: {result.documented_items}/{result.total_items}")

for missing in result.missing:
    print(f"Missing docs: {missing}")
```

## Configuration

All tools support custom configuration:

```python
from code_agent.config.custom import (
    FormatterConfig,
    LinterConfig,
    DependencyConfig,
    DocumentationConfig,
)

# Configure formatter
formatter_config = FormatterConfig(
    line_length=100,
    skip_string_normalization=True,
)
formatter = CodeFormatter(formatter_config)

# Configure linter
linter_config = LinterConfig(
    max_complexity=15,
    max_line_length=100,
    ignore_rules=["E501"],  # Ignore line too long
)
linter = CodeLinter(linter_config)

# Configure dependency analyzer
dep_config = DependencyConfig(
    check_unused=True,
    check_circular=True,
)
analyzer = DependencyAnalyzer(dep_config)

# Configure documentation analyzer
doc_config = DocumentationConfig(
    min_coverage=0.8,
    require_param_docs=True,
    require_return_docs=True,
)
doc_analyzer = DocumentationAnalyzer(doc_config)
```

## Custom Validators

Use custom validators for additional code quality checks:

```python
from code_agent.tools.custom.validators import (
    validate_naming_conventions,
    validate_enhanced_complexity,
    validate_code_smells,
    validate_best_practices,
    create_quality_validator,
)

code = """
def MyFunction():  # Bad: should be snake_case
    pass
"""

# Validate naming conventions
errors = validate_naming_conventions(code)
for error in errors:
    print(f"Naming error: {error}")

# Validate complexity
errors = validate_enhanced_complexity(code)
for error in errors:
    print(f"Complexity error: {error}")

# Validate code smells
errors = validate_code_smells(code)
for error in errors:
    print(f"Code smell: {error}")

# Validate best practices
errors = validate_best_practices(code)
for error in errors:
    print(f"Best practice violation: {error}")

# Use composite validator
validator = create_quality_validator(min_doc_coverage=0.8)
errors = validator(code)
for error in errors:
    print(f"Quality issue: {error}")
```

## Workflows

Use the QualityWorkflow to orchestrate multiple tools:

```python
from code_agent.workflows import QualityWorkflow

workflow = QualityWorkflow()

code = """
def hello():
    print("Hello, World!")
"""

# Run complete quality check
report = workflow.run(code)

print(f"Success: {report.success}")
print(f"Total steps: {report.summary['total_steps']}")
print(f"Errors: {report.total_errors}")
print(f"Warnings: {report.total_warnings}")

for step in report.steps:
    print(f"{step.name}: {'✓' if step.success else '✗'}")

# Quick check
is_good = workflow.quick_check(code)
print(f"Code quality: {'Good' if is_good else 'Needs improvement'}")

# Format and fix
fixed_code = workflow.format_and_fix(code)
print(fixed_code)
```

## Integration with Code Agent

The custom tools are automatically available when imported:

```python
from code_agent import (
    CodeFormatter,
    CodeLinter,
    DependencyAnalyzer,
    DocumentationAnalyzer,
    QualityWorkflow,
)

# Use with existing tools
from code_agent import CodeAnalyzer, RefactoringEngine

analyzer = CodeAnalyzer()
formatter = CodeFormatter()
workflow = QualityWorkflow()

# Complete analysis
code = "def test(): pass"

# 1. Analyze
analysis = analyzer.analyze(code)

# 2. Format
formatted = formatter.format(code)

# 3. Run quality workflow
report = workflow.run(formatted.code)
```

## API Reference

See the individual tool modules for detailed API documentation:

- `code_agent.tools.custom.formatter` - CodeFormatter
- `code_agent.tools.custom.linter` - CodeLinter
- `code_agent.tools.custom.dependencies` - DependencyAnalyzer
- `code_agent.tools.custom.documentation` - DocumentationAnalyzer
- `code_agent.tools.custom.validators` - Custom validators
- `code_agent.workflows.quality` - QualityWorkflow

## Examples

See `code_agent/examples/custom_workflows.py` for complete examples.

## Testing

Run tests for custom tools:

```bash
# Test custom tools
pytest tests/code_agent/test_custom_tools.py -v

# Test workflows
pytest tests/code_agent/test_workflows.py -v

# Test all
pytest tests/code_agent/ -v
```

## License

Same as Code Agent package.
