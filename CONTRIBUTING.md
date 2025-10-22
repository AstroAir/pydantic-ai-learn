# Contributing to pydantic-ai-learn

Thank you for your interest in contributing to pydantic-ai-learn! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pydantic-ai-learn.git
   cd pydantic-ai-learn
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/pydantic-ai-learn.git
   ```

## Development Setup

### Quick Setup (Recommended)

Use our automated setup script:

**Windows (PowerShell):**
```powershell
.\scripts\setup.ps1
```

**Linux/macOS (Bash):**
```bash
./scripts/setup.sh
```

### Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Install pre-commit hooks (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues reported in the issue tracker
- **New features**: Add new tools, examples, or functionality
- **Documentation**: Improve README, docstrings, or add examples
- **Tests**: Add or improve test coverage
- **Code quality**: Refactoring, performance improvements

### Contribution Workflow

1. **Check existing issues**: Look for existing issues or create a new one
2. **Create a branch**: Create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes**: Implement your changes following our coding standards
4. **Test**: Run tests and ensure they pass
5. **Commit**: Write clear, descriptive commit messages
6. **Push**: Push your changes to your fork
7. **Pull Request**: Open a PR against the `main` branch

## Coding Standards

### Code Style

We use **Ruff** for linting and formatting:

```bash
# Check code style
ruff check .

# Auto-format code
ruff format .

# Or use the convenience script
./scripts/format.sh  # or format.ps1 on Windows
```

### Type Hints

- Use type hints for all function signatures
- We use **mypy** for type checking:
  ```bash
  mypy code_agent tools utils

  # Or use the convenience script
  ./scripts/lint.sh  # or lint.ps1 on Windows
  ```

### Code Organization

- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Follow existing project structure

### Documentation

- Update docstrings when changing function signatures
- Add examples for new features
- Update README.md if adding user-facing features
- Keep CHANGELOG.md updated

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_formatter.py

# Run with coverage
pytest --cov=code_agent --cov=tools --cov=utils

# Or use the convenience script
./scripts/run_tests.sh  # or run_tests.ps1 on Windows
```

### Writing Tests

- Write tests for all new functionality
- Aim for high test coverage (>80%)
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Group related tests in classes
- Use fixtures for common setup

Example:
```python
def test_bash_tool_executes_simple_command_successfully():
    """Test that bash tool can execute a simple echo command."""
    result = run_bash_command("echo 'hello'")
    assert result.exit_code == 0
    assert "hello" in result.output
```

### Test Organization

- Place tests in `tests/` directory
- Mirror the source code structure
- Use `conftest.py` for shared fixtures

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Run tests
   pytest

   # Run linting
   ruff check .

   # Run type checking
   mypy code_agent tools utils

   # Or run all checks at once
   ./scripts/run_all_checks.sh  # or run_all_checks.ps1 on Windows
   ```

3. **Ensure tests pass**: All tests must pass before submitting

4. **Update documentation**: Update relevant documentation

### PR Guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Fill out the PR template completely
- **Link issues**: Reference related issues using `Fixes #123` or `Relates to #456`
- **Small PRs**: Keep PRs focused and reasonably sized
- **Commits**: Use clear commit messages following conventional commits format

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat(tools): add file editing toolkit

Add comprehensive file editing toolkit with support for:
- Single file edits
- Multi-file atomic edits
- Jupyter notebook editing

Fixes #123
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, a maintainer will merge your PR

## Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml) to report bugs.

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, package version
- Error logs or stack traces

## Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml) to suggest features.

Include:
- Problem statement
- Proposed solution
- Use case
- Code examples (if applicable)

## Questions?

- Open a [Discussion](https://github.com/ORIGINAL_OWNER/pydantic-ai-learn/discussions)
- Check existing [Issues](https://github.com/ORIGINAL_OWNER/pydantic-ai-learn/issues)
- Read the [README](README.md)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

---

Thank you for contributing to pydantic-ai-learn! 🎉
