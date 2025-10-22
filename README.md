# pydantic-ai-learn

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-614%20passed-success.svg)](tests/)

A comprehensive learning project for PydanticAI featuring tools, utilities, and extensive examples.

> **📢 Project Reorganization (2025-10-21):** The project structure has been reorganized to follow Python best practices. See [PROJECT_REORGANIZATION_SUMMARY.md](PROJECT_REORGANIZATION_SUMMARY.md) for details.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Tool](#cli-tool)
- [Features](#features)
- [Examples](#examples)
- [Development](#development)
  - [Automation Scripts](#automation-scripts)
  - [Contributing](#contributing)
  - [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Changelog](#changelog)

## 🎯 Overview

This project is a comprehensive learning resource for PydanticAI, featuring:

- **Production-ready tools** for bash execution, file system operations, file editing, task planning, and code analysis
- **Extensive examples** demonstrating various PydanticAI features and patterns
- **Utility functions** for formatting, terminal streaming, and more
- **Code Agent** - An advanced autonomous code analysis and manipulation agent
- **Best practices** for structuring Python applications with PydanticAI

## 📁 Project Structure

```
pydantic-ai-learn/
├── code_agent/              # Advanced code analysis agent package
│   ├── __init__.py          # Package initialization and exports
│   ├── legacy/              # Backward compatibility layer
│   │   ├── __init__.py
│   │   ├── agent.py         # Main CodeAgent class
│   │   ├── toolkit.py       # Code analysis tools
│   │   ├── error_handling.py # Error handling and retry logic
│   │   ├── workflow.py      # Workflow orchestration
│   │   ├── context_management.py # Context management
│   │   ├── logging_config.py # Structured logging
│   │   └── mcp_config.py    # MCP configuration
│   ├── ui/                  # Terminal UI components
│   │   ├── __init__.py
│   │   ├── terminal.py      # Terminal UI implementation
│   │   └── runner.py        # Terminal runner
│   ├── config/              # Configuration modules
│   │   ├── __init__.py
│   │   ├── mcp.py           # MCP configuration
│   │   ├── logging.py       # Logging configuration
│   │   └── mcp_legacy.py    # Legacy MCP config
│   ├── examples/            # Code agent specific examples
│   │   ├── __init__.py
│   │   ├── main.py          # Main examples
│   │   ├── auto_debug.py    # Auto-debug examples
│   │   ├── context_management.py # Context examples
│   │   ├── graph.py         # Graph examples
│   │   └── terminal_ui.py   # Terminal UI examples
│   └── [documentation files] # Architecture, migration guides, etc.
│   ├── core/                # Core modules (refactored)
│   ├── tools/               # Tool modules
│   ├── utils/               # Utility modules
│   └── adapters/            # Adapter modules
│
├── tools/                   # PydanticAI tools and utilities
│   ├── __init__.py
│   ├── bash_tool.py        # Bash command execution
│   ├── filesystem_tools.py # File system operations (glob, grep, ls)
│   ├── file_editing_toolkit.py # File editing operations
│   ├── task_planning_toolkit.py # Task planning and management
│   ├── code_agent_toolkit.py # Code analysis toolkit
│   └── *.md                # Tool documentation
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── formatter.py        # Conversation formatting
│   ├── terminal_stream.py  # Terminal streaming
│   └── check_url.py        # URL validation
│
├── examples/                # Example scripts and demos
│   ├── basic/              # Basic PydanticAI examples
│   ├── messages/           # Message handling examples
│   ├── output/             # Output formatting examples
│   ├── toolsets/           # Toolset usage examples
│   ├── tools/              # Tool-specific examples
│   ├── code_agent/         # Code agent examples
│   ├── graph/              # Graph-based workflow examples
│   ├── mcp/                # MCP (Model Context Protocol) examples
│   ├── multi-agent/        # Multi-agent pattern examples
│   └── simple_demo.py      # Simple demo script
│
├── tests/                   # Test files
│   ├── code_agent/         # Code agent tests
│   │   ├── test_comprehensive.py
│   │   ├── test_pydanticai_*.py
│   │   └── [other code agent tests]
│   ├── test_bash_tool.py
│   ├── test_filesystem_tools.py
│   ├── test_file_editing_toolkit.py
│   ├── test_task_planning_toolkit.py
│   └── test_formatter.py
│
├── docs/                    # Development documentation
│   ├── STRUCTURE_BEFORE_REORGANIZATION.md
│   ├── CHANGES_SUMMARY.md
│   └── [other development docs]
│
├── scripts/                 # Utility scripts
│   ├── check_circular_deps.py # Check for circular dependencies
│   ├── run_all_checks.py   # Run all quality checks
│   └── test_import.py      # Test imports
│
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
├── LICENSE                 # License file
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/pydantic-ai-learn.git
cd pydantic-ai-learn

# Install dependencies
uv sync

# Or install with development dependencies
uv sync --extra dev
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/pydantic-ai-learn.git
cd pydantic-ai-learn

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## ⚡ Quick Start

### Using the Code Agent

```python
from code_agent import CodeAgent

# Create an agent
agent = CodeAgent()

# Analyze code
result = agent.run_sync("Analyze the code in tools/task_planning_toolkit.py")
print(result.output)

# Detect code smells
result = agent.run_sync("Detect code smells in tools/bash_tool.py")
print(result.output)
```

### Using Tools Directly

```python
from tools.filesystem_tools import glob_files, GlobInput

# Find all Python files
result = glob_files(GlobInput(pattern="**/*.py", max_results=10))
print(result.files)
```

### Running Examples

```bash
# Basic examples
python examples/basic/hello_world.py
python examples/basic/streaming_iter.py

# Tool examples
python examples/tools/filesystem_tools_example.py
python examples/tools/bash_tool_example.py

# Code agent examples
python examples/code_agent/code_agent_example.py

# Graph-based workflow examples
python examples/graph/ai_q_and_a_graph.py
python examples/graph/count_down.py

# MCP examples
python examples/mcp/client_example.py
python examples/mcp/mcp_server.py

# Multi-agent examples
python examples/multi-agent/agent_delegation_simple.py
python examples/multi-agent/programmatic_handoff.py
```

## 🖥️ CLI Tool

The Code Agent Terminal UI can be launched as a command-line tool with enhanced features including keyboard shortcuts, streaming display, and session management.

### Quick Launch

```bash
# Method 1: Direct script (no installation required)
python launch_terminal.py

# Method 2: Python module (no installation required)
python -m code_agent.cli

# Method 3: Installed CLI tool (after pip install -e .)
code-agent
```

### CLI Options

```bash
# Launch with streaming mode (recommended)
python -m code_agent.cli --streaming

# Launch with custom model
python -m code_agent.cli --model openai:gpt-4-turbo

# Launch with debug logging
python -m code_agent.cli --log-level DEBUG

# Check dependencies
python -m code_agent.cli --check-deps

# Show help
python -m code_agent.cli --help
```

### Keyboard Shortcuts

When `prompt_toolkit` is installed:

| Shortcut | Action |
|----------|--------|
| `Ctrl+X H` | Show help |
| `Ctrl+X M` | Show metrics |
| `Ctrl+X E` | Show errors |
| `Ctrl+X C` | Clear screen |
| `Ctrl+L` | Clear screen |
| `↑` / `↓` | Navigate history |
| `Tab` | Auto-complete |

### Terminal Commands

| Command | Description |
|---------|-------------|
| `help` | Show help message |
| `clear` | Clear the screen |
| `metrics` | Show performance metrics |
| `errors` | Show error history |
| `save` | Save current session |
| `export [file]` | Export session to file |
| `exit` | Exit terminal |

### Documentation

- **Installation Guide**: [CLI_INSTALLATION_GUIDE.md](CLI_INSTALLATION_GUIDE.md)
- **CLI README**: [CLI_README.md](CLI_README.md)
- **Quick Reference**: [CLI_QUICK_REFERENCE.md](CLI_QUICK_REFERENCE.md)
- **Terminal Quick Start**: [code_agent/TERMINAL_QUICK_START.md](code_agent/TERMINAL_QUICK_START.md)
- **Keyboard Shortcuts**: [code_agent/TERMINAL_KEYBOARD_SHORTCUTS.md](code_agent/TERMINAL_KEYBOARD_SHORTCUTS.md)

## ✨ Features

### Code Agent

- **Autonomous code analysis** with AST-based parsing
- **Pattern detection** for code smells and anti-patterns
- **Refactoring suggestions** with examples
- **Code generation** with type hints and docstrings
- **Dependency analysis** and metrics calculation
- **Auto-debugging** with circuit breakers and retry logic
- **Streaming support** for real-time feedback
- **Context management** for multi-turn conversations

### Tools

- **BashTool**: Execute bash/PowerShell commands with timeout and error handling
- **FileSystemTools**: Glob, grep, and ls operations with Pydantic validation
- **FileEditingToolkit**: Edit, write, and manipulate files
- **TaskPlanningToolkit**: Manage task lists with state tracking
- **CodeAgentToolkit**: Comprehensive code analysis and manipulation

### Utilities

- **ConversationFormatter**: Format PydanticAI conversations for display
- **TerminalStream**: Stream output to terminal with formatting
- **URL Checker**: Validate and check URLs

## 📚 Examples

The `examples/` directory contains extensive examples organized by category:

- **basic/**: Basic PydanticAI usage (agents, streaming, tools, etc.)
- **messages/**: Message handling and conversation management
- **output/**: Output formatting and structured responses
- **toolsets/**: Toolset composition and filtering
- **tools/**: Tool-specific examples and tests
- **code_agent/**: Code agent usage examples
- **graph/**: Graph-based workflow examples using pydantic_graph
- **mcp/**: Model Context Protocol (MCP) integration examples
- **multi-agent/**: Multi-agent patterns and delegation examples

## 🛠️ Development

### Automation Scripts

We provide convenient scripts for common development tasks. All scripts work on both Windows (PowerShell) and Unix-like systems (Bash).

#### Setup

**Windows:**
```powershell
.\scripts\setup.ps1
```

**Linux/macOS:**
```bash
./scripts/setup.sh
```

This script will:
- Check Python version (3.12+ required)
- Create virtual environment
- Install all dependencies
- Verify installation

#### Running Tests

**Windows:**
```powershell
# Run all tests
.\scripts\run_tests.ps1

# Run with coverage
.\scripts\run_tests.ps1 -Coverage

# Run specific path
.\scripts\run_tests.ps1 -Path tests/test_formatter.py

# Run in parallel (faster)
.\scripts\run_tests.ps1 -Fast

# Stop on first failure
.\scripts\run_tests.ps1 -FailFast
```

**Linux/macOS:**
```bash
# Run all tests
./scripts/run_tests.sh

# Run with coverage
./scripts/run_tests.sh --coverage

# Run specific path
./scripts/run_tests.sh tests/test_formatter.py

# Run in parallel
./scripts/run_tests.sh --fast
```

#### Code Quality

**Windows:**
```powershell
# Run all quality checks
.\scripts\lint.ps1

# Run specific tool
.\scripts\lint.ps1 -Tool ruff
.\scripts\lint.ps1 -Tool mypy

# Auto-fix issues
.\scripts\lint.ps1 -Fix

# Format code
.\scripts\format.ps1

# Check formatting without changes
.\scripts\format.ps1 -Check
```

**Linux/macOS:**
```bash
# Run all quality checks
./scripts/lint.sh

# Run specific tool
./scripts/lint.sh --tool ruff
./scripts/lint.sh --tool mypy

# Auto-fix issues
./scripts/lint.sh --fix

# Format code
./scripts/format.sh

# Check formatting
./scripts/format.sh --check
```

#### Running Examples

**Windows:**
```powershell
# List all examples
.\scripts\run_examples.ps1 -List

# Run simple examples
.\scripts\run_examples.ps1

# Run specific category
.\scripts\run_examples.ps1 -Category basic

# Run specific file
.\scripts\run_examples.ps1 -File examples/simple_demo.py
```

**Linux/macOS:**
```bash
# List all examples
./scripts/run_examples.sh --list

# Run simple examples
./scripts/run_examples.sh

# Run specific category
./scripts/run_examples.sh --category basic

# Run specific file
./scripts/run_examples.sh --file examples/simple_demo.py
```

#### Cleanup

**Windows:**
```powershell
# Clean build artifacts and cache
.\scripts\clean.ps1

# Also remove virtual environment
.\scripts\clean.ps1 -All

# Dry run (see what would be deleted)
.\scripts\clean.ps1 -DryRun
```

**Linux/macOS:**
```bash
# Clean build artifacts and cache
./scripts/clean.sh

# Also remove virtual environment
./scripts/clean.sh --all

# Dry run
./scripts/clean.sh --dry-run
```

### Manual Commands

If you prefer to run commands manually:

```bash
# Run all checks (mypy, ruff, tests)
python scripts/run_all_checks.py

# Check for circular dependencies
python scripts/check_circular_deps.py

# Run type checking
mypy code_agent tools utils --strict

# Run linting
ruff check .

# Run tests
pytest

# Run specific test file
pytest tests/test_filesystem_tools.py -v

# Run with coverage
pytest --cov=code_agent --cov=tools --cov=utils
```

### Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes and releases.

## 🙏 Acknowledgments

- Built with [PydanticAI](https://github.com/pydantic/pydantic-ai)
- Inspired by the PydanticAI community and examples

---

**Note**: This is a learning project. For production use, review and adapt the code to your specific needs.
