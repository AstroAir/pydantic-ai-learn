# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Essential development commands

Setup

```bash path=null start=null
# Using uv (recommended)
uv sync                 # install deps
uv sync --extra dev     # include dev tools (mypy, ruff, pytest)

# Using pip (alternative)
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

Lint and type-check

```bash path=null start=null
# Ruff (lint)
ruff check .

# MyPy (strict typing per pyproject)
mypy code_agent --strict --show-error-codes

# Run the bundled quality script
python scripts/run_all_checks.py
```

Tests

```bash path=null start=null
# Run all tests
pytest

# Run a specific test file (verbose)
pytest tests/test_filesystem_tools.py -v

# Run a single test function
pytest tests/test_filesystem_tools.py::test_glob_basic

# With coverage for key packages
pytest --cov=code_agent --cov=tools --cov=utils
```

Utilities

```bash path=null start=null
# Check for circular dependencies
python scripts/check_circular_deps.py

# Console scripts (available if installed as a package)
run-all-checks
check-circular-deps
```

Notes

- Prefer uv for environment management; otherwise editable install + requirements-dev is sufficient.
- Example scripts live under examples/ (including basic/, tools/, code_agent/, graph/, mcp/, multi-agent/). Run them with python path/to/script.py.

## High-level architecture overview

The project centers on the code_agent package, a modular code analysis and automation framework built on PydanticAI.

- Core orchestration (code_agent/core): CodeAgent is the primary façade coordinating analysis, refactoring suggestions, code generation, and optional streaming. AgentConfig/AgentState capture configuration and runtime state; types define Execution* models for code execution.
- Tools layer (code_agent/tools): CodeAnalyzer performs AST/code metrics and pattern detection; RefactoringEngine proposes improvements; CodeGenerator produces stubs/templates; executor/validators provide safe code execution with caching and validation.
- Utils (code_agent/utils): Structured logging, performance metrics, and robust error handling (retry strategies, circuit breakers, error categorization/diagnosis) underpin resilience and telemetry.
- Adapters (code_agent/adapters): ContextManager handles conversation/history pruning; WorkflowOrchestrator coordinates multi-step fix/validation flows; Graph* enables graph-structured execution/persistence.
- Config (code_agent/config): Centralizes logging, execution safety controls, and MCP configuration; helpers create safe/restricted/full execution profiles.
- UI (code_agent/ui): Terminal UI components and launch helpers for interactive runs.
- Public surface (code_agent/__init__.py): Re-exports the core types, convenience functions (quick_analyze/quick_refactor), logging/error APIs, workflow/context/graph adapters, and terminal UI. This file is the canonical import point for consumers.

Outside code_agent, first-party PydanticAI tools (tools/) provide general-purpose capabilities (bash execution, filesystem ops, file editing, task planning) with tests in tests/ (including tests/code_agent/ for code agent specific tests).

Data flow at a glance: a request enters CodeAgent → configuration/logging load → analysis/refactoring/generation tools run (with retries/fault tolerance) → optional workflow/graph orchestration → results streamed or returned.

## Workflow patterns specific to this repo

- Use strict typing and enforce quality locally before committing: run python scripts/run_all_checks.py.
- Keep ruff and mypy aligned with pyproject.toml; don’t relax strictness unless justified.
- All tests are in tests/ directory (including tests/code_agent/ for code agent tests); prefer pytest -q or -v with file/function targets for focused runs.
- Examples serve as smoke tests for public APIs; validate changes by running representative examples in examples/ (basic/, tools/, code_agent/, graph/, mcp/, multi-agent/).

## Conventions from project config and Cursor rules

- Ruff: target Python 3.12, line length 120; rulesets enabled: E, F, W, I, N, UP, B, A, C4, SIM, RET; per-file ignores for imports in examples and specific modules.
- MyPy: strict mode for code_agent; selected modules have relaxed overrides; prefer explicit typing for public APIs.
- Style (Cursor rules): sorted imports (stdlib, third-party, local), early returns over deep nesting, avoid broad excepts, meaningful names, comments for rationale/edge cases.
- Quality commands to remember (from rules):

```bash path=null start=null
python scripts/run_all_checks.py
python scripts/check_circular_deps.py
mypy code_agent --strict
ruff check .
pytest -q
```
