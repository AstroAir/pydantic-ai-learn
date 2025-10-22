# Code Agent v2.1 - Enhanced Autonomous Code Analysis with Auto-Debugging

A sophisticated, autonomous code agent built on PydanticAI with **advanced auto-debugging capabilities** for comprehensive code analysis, refactoring, and generation.

## 🚀 Features

### Core Capabilities

- **Code Analysis**: Parse and analyze Python code structure using AST
- **Pattern Detection**: Identify code smells, anti-patterns, and potential issues
- **Quality Metrics**: Calculate complexity, LOC, and maintainability metrics
- **Dependency Analysis**: Analyze imports and categorize dependencies
- **Syntax Validation**: Check Python syntax with optional strict mode
- **Refactoring Suggestions**: Propose concrete improvement opportunities
- **Code Generation**: Generate well-structured, documented Python code

### 🆕 Auto-Debugging Features (v2.1) - **NEW!**

- **Automatic Error Recovery**: Circuit breakers prevent cascading failures
- **Intelligent Retry Strategy**: Exponential backoff with jitter for transient errors
- **Error Categorization**: Automatic classification (transient/permanent, recoverable/fatal)
- **Error Diagnosis Engine**: Provides recovery suggestions based on error type
- **Workflow Automation**: Multi-step debugging with state checkpointing
- **Fix Strategy Engine**: Multiple strategies (syntax, imports, logic, refactor)
- **Validation & Rollback**: Validate fixes before applying, rollback on failure
- **Structured Logging**: JSON & human-readable formats with performance metrics
- **Log Sanitization**: Automatic removal of sensitive data (API keys, tokens)

### 🖥️ Interactive Terminal UI (v2.1) - **NEW!**

- **Rich Terminal Interface**: Claude Code-inspired interactive terminal
- **Real-time Streaming**: See responses as they're generated
- **Syntax Highlighting**: Automatic highlighting for code blocks
- **Markdown Rendering**: Beautiful rendering of formatted text
- **Command History**: Navigate through previous commands
- **Session Export**: Save conversations to markdown files
- **Metrics Dashboard**: Real-time performance and error tracking
- **Workflow Monitoring**: Visual workflow status and debugging progress

### 🧠 Intelligent Routing System (v2.2) - **NEW!**

- **Prompt Enhancement**: Automatically improve prompt clarity and specificity
- **Request Classification**: Analyze difficulty (SIMPLE/MODERATE/COMPLEX) and mode (CHAT/AGENT)
- **Multi-Model Support**: Configure multiple AI models with different capabilities
- **Smart Routing**: Route requests to optimal models based on:
  - Difficulty level and task complexity
  - Request type (chat vs agent operations)
  - Model capabilities and cost optimization
  - Custom routing policies
- **Cost Optimization**: Automatically use cheaper models for simple tasks
- **Dry Run Mode**: Test routing decisions without applying changes
- **Metrics & Logging**: Track routing decisions and model usage
- **100% Backward Compatible**: All features are opt-in

### Advanced Features (v2.0)

- **Streaming Support**: Real-time analysis feedback with progress updates
- **Async Iteration**: Access to execution nodes for fine-grained control
- **Usage Limits**: Token tracking and limits for cost control
- **Retry Logic**: Automatic retry on failures for robust execution
- **Conversation History**: Multi-turn conversations with context preservation
- **Package Structure**: Proper Python package organization

### Integration

- **Task Planning**: Integrates with existing task planning toolkit
- **File System**: Works with filesystem tools for code discovery
- **File Editing**: Connects to file editing toolkit for modifications
- **Autonomous Operation**: Can break down complex tasks and execute them

### Technical Features

- Modern Python 3.12+ with latest type hints
- Pydantic v2 validation for robust input handling
- AST-based analysis (no code execution)
- Comprehensive error handling with retry support
- Security-focused design
- Extensive documentation and examples
- Full backward compatibility with v1.0

## 📦 Installation

The code agent is part of the pydantic-ai-learn project and requires:

```bash
# Python 3.12+
# pydantic-ai >= 1.0.15
# Pydantic v2
```

All dependencies are already configured in `pyproject.toml`.

## 🎯 Quick Start

### Basic Usage

```python
from code_agent import CodeAgent

# Create agent
agent = CodeAgent()

# Analyze code
result = agent.run_sync("Analyze the code in tools/task_planning_toolkit.py")
print(result.output)

# Detect patterns
result = agent.run_sync("Find code smells in my_module.py")
print(result.output)

# Get refactoring suggestions
result = agent.run_sync("Suggest refactoring for complex functions in main.py")
print(result.output)
```

### 🆕 Streaming Usage (v2.0)

```python
from code_agent import CodeAgent

agent = CodeAgent(enable_streaming=True)

# Stream analysis in real-time
async for text in agent.run_stream("Analyze large_module.py"):
    print(text, end="", flush=True)
```

### 🆕 Usage Limits (v2.0)

```python
from code_agent import CodeAgent, UsageLimits

# Create agent with token limits
agent = CodeAgent(
    usage_limits=UsageLimits(response_tokens_limit=1000)
)

result = agent.run_sync("Analyze my_module.py")
print(agent.get_usage_summary())
```

### 🆕 Intelligent Routing (v2.2)

```python
from code_agent.core import create_code_agent, create_default_routing_config

# Enable intelligent routing
routing_config = create_default_routing_config()
routing_config.enabled = True

agent = create_code_agent(
    model="openai:gpt-4o-mini",
    routing_config=routing_config,
)

# Routing happens automatically
result = agent.run_sync("Analyze my code")  # Routes to appropriate model

# Check routing metrics
if agent.model_router:
    print(agent.model_router.get_metrics())
```

See [Routing System Documentation](docs/ROUTING_SYSTEM.md) for details.

### 🆕 Multi-Turn Conversations (v2.0)

```python
from code_agent import CodeAgent

agent = CodeAgent()

# First turn
result1 = agent.run_sync("Analyze main.py")

# Second turn with history
result2 = agent.run_sync(
    "What are the top 3 issues?",
    message_history=result1.new_messages()
)
```

### 🆕 Auto-Debugging with Error Recovery (v2.1) - **NEW!**

```python
from code_agent import CodeAgent, LogLevel, LogFormat

# Create agent with auto-debugging features
agent = CodeAgent(
    model="openai:gpt-4",
    log_level=LogLevel.DEBUG,
    log_format=LogFormat.JSON,
    enable_workflow=True
)

# Analyze with automatic error recovery
result = agent.run_sync("Analyze code_agent/agent.py")

# Check error history and recovery suggestions
error_summary = agent.state.get_error_summary()
print(f"Total Errors: {error_summary['total_errors']}")

# View performance metrics
metrics = agent.state.logger.get_metrics_summary()
print(f"Success Rate: {metrics['successful']}/{metrics['total_operations']}")
```

### 🆕 Circuit Breaker Pattern (v2.1) - **NEW!**

```python
from code_agent import CodeAgent

agent = CodeAgent()

# Circuit breaker automatically prevents cascading failures
cb = agent.state.get_or_create_circuit_breaker(
    "analyze_code",
    failure_threshold=5,
    recovery_timeout=60.0
)

# Circuit opens after threshold failures
# Automatically recovers after timeout
result = agent.run_sync("Analyze module.py")
print(f"Circuit State: {cb.state.value}")
```

### 🆕 Workflow Automation (v2.1) - **NEW!**

```python
from code_agent import CodeAgent, WorkflowState

agent = CodeAgent(enable_workflow=True)

if agent.state.workflow_orchestrator:
    workflow = agent.state.workflow_orchestrator

    # Create checkpoint before risky operation
    checkpoint = workflow.create_checkpoint(
        input_data={"operation": "refactor"},
        output_data=None
    )

    try:
        result = agent.run_sync("Refactor legacy_code.py")
        workflow.transition_to(WorkflowState.COMPLETED)
    except Exception:
        # Rollback on failure
        workflow.rollback_to_checkpoint(checkpoint.checkpoint_id)
```

### 🖥️ Interactive Terminal UI (v2.1) - **NEW!**

```python
from code_agent import launch_terminal

# Launch interactive terminal with streaming
launch_terminal(use_streaming=True)
```

**Command Line:**
```bash
# Launch terminal
python -m code_agent.run_terminal

# With streaming
python -m code_agent.run_terminal --streaming

# Custom configuration
python -m code_agent.run_terminal --model openai:gpt-4 --log-level DEBUG
```

**Available Commands:**
- `help` - Show all commands
- `metrics` - Show performance metrics
- `errors` - Show error history
- `workflow` - Show workflow status
- `export [file]` - Export session
- `exit` - Exit terminal

See [TERMINAL_UI_README.md](TERMINAL_UI_README.md) for complete documentation.

### Async Usage

```python
import asyncio
from code_agent import CodeAgent

async def main():
    agent = CodeAgent()
    result = await agent.run("Analyze code structure in src/")
    print(result.output)

asyncio.run(main())
```

## 🛠️ Available Tools

### 1. Analyze Python Code

Comprehensive code analysis including structure, metrics, patterns, and dependencies.

```python
result = agent.run_sync(
    "Analyze tools/code_agent_toolkit.py with full analysis"
)
```

**Parameters:**
- `file_path`: Path to Python file
- `analysis_type`: structure, metrics, patterns, dependencies, or full
- `include_metrics`: Include quality metrics (default: True)
- `include_patterns`: Detect code smells (default: True)

### 2. Validate Python Syntax

Check Python file syntax with optional strict validation.

```python
result = agent.run_sync(
    "Validate syntax of my_script.py with strict mode"
)
```

**Parameters:**
- `file_path`: Path to Python file
- `strict`: Use strict validation mode (default: False)

### 3. Detect Code Patterns

Identify code smells and anti-patterns.

```python
result = agent.run_sync(
    "Detect code patterns in module.py with high severity threshold"
)
```

**Parameters:**
- `file_path`: Path to Python file
- `severity_threshold`: low, medium, or high (default: low)

**Detected Patterns:**
- Long functions (>50 lines)
- Long parameter lists (>5 parameters)
- High complexity (>20)
- Deep nesting (>4 levels)
- Magic numbers
- God classes (>20 methods)

### 4. Get Code Metrics

Calculate comprehensive code quality metrics.

```python
result = agent.run_sync("Calculate metrics for src/main.py")
```

**Metrics Calculated:**
- Lines of code (total, code, comments, blank)
- Function and class counts
- Cyclomatic complexity (average, max)
- Complexity distribution

### 5. Analyze Dependencies

Find and categorize code dependencies.

```python
result = agent.run_sync("Analyze dependencies in my_module.py")
```

**Categories:**
- Standard library imports
- Third-party packages
- Local/relative imports

### 6. Get Refactoring Suggestions

Suggest concrete refactoring opportunities.

```python
result = agent.run_sync(
    "Suggest refactoring for tools/bash_tool.py with examples"
)
```

**Parameters:**
- `file_path`: Path to Python file
- `include_examples`: Include code examples (default: True)

**Suggestions Include:**
- Complexity reduction strategies
- Function extraction opportunities
- Code organization improvements
- Design pattern applications

### 7. Generate Python Code

Generate well-structured Python code from descriptions.

```python
result = agent.run_sync(
    "Generate a function to validate email addresses with type hints and docstrings"
)
```

**Parameters:**
- `description`: Detailed description of code to generate
- `code_type`: function, class, module, or snippet
- `include_docstrings`: Include docstrings (default: True)
- `include_type_hints`: Include type hints (default: True)

## 📚 Examples

### Example 1: Comprehensive Code Review

```python
agent = CodeAgent()

result = agent.run_sync("""
Perform a comprehensive code review of tools/task_planning_toolkit.py including:
1. Code structure analysis
2. Quality metrics
3. Pattern detection
4. Refactoring suggestions
Provide a summary with prioritized recommendations.
""")

print(result.output)
```

### Example 2: Multi-File Comparison

```python
result = agent.run_sync("""
Compare code quality metrics between:
1. tools/task_planning_toolkit.py
2. tools/filesystem_tools.py
3. tools/file_editing_toolkit.py
Identify which has the best code quality and why.
""")
```

### Example 3: Code Improvement Workflow

```python
async def improve_code():
    agent = CodeAgent()

    # Step 1: Analyze
    analysis = await agent.run(
        "Analyze my_module.py and identify top 3 areas for improvement"
    )

    # Step 2: Get suggestions
    suggestions = await agent.run(
        "Suggest specific refactoring steps for the top issue"
    )

    # Step 3: Generate improved code
    improved = await agent.run(
        "Generate an example of the refactored code"
    )
```

### Example 4: Direct Tool Usage

```python
from tools.code_agent_toolkit import (
    CodeAgentState,
    AnalyzeCodeInput,
    analyze_code,
)

state = CodeAgentState()

result = analyze_code(
    AnalyzeCodeInput(
        file_path="my_script.py",
        analysis_type="full"
    ),
    state
)

print(result)
```

## 🏗️ Architecture

### Components

1. **Code Agent Toolkit** (`tools/code_agent_toolkit.py`)
   - Core analysis functions
   - Pydantic input models
   - State management
   - Helper functions

2. **Code Agent** (`code_agent.py`)
   - Main agent class
   - Tool registration
   - PydanticAI integration
   - System prompts

3. **Examples** (`code_agent_example.py`)
   - Usage demonstrations
   - Best practices
   - Common workflows

### State Management

The `CodeAgentState` class manages:
- Analysis result cache
- Task planning state (if available)
- File editing state (if available)
- Current code context

### Integration Points

- **Task Planning Toolkit**: For breaking down complex tasks
- **Filesystem Tools**: For code discovery and search
- **File Editing Toolkit**: For code modifications
- **Bash Tool**: For running tests and validation

## 🔒 Security

- **No Code Execution**: Analysis uses AST parsing only
- **Path Validation**: All file paths are validated
- **File Size Limits**: Prevents resource exhaustion (1MB max)
- **Error Handling**: Prevents information leakage
- **Input Validation**: Pydantic models validate all inputs

## 📖 API Reference

### CodeAgent Class

```python
class CodeAgent:
    def __init__(self, model: str = "openai:gpt-4")
    def run_sync(self, prompt: str) -> Any
    async def run(self, prompt: str) -> Any
```

### Core Functions

```python
def analyze_code(input_params: AnalyzeCodeInput, state: CodeAgentState) -> str
def validate_syntax(input_params: ValidateSyntaxInput, state: CodeAgentState) -> str
def detect_patterns(input_params: DetectPatternsInput, state: CodeAgentState) -> str
def calculate_metrics(file_path: str, state: CodeAgentState) -> str
def find_dependencies(file_path: str, state: CodeAgentState) -> str
def suggest_refactoring(input_params: SuggestRefactoringInput, state: CodeAgentState) -> str
def generate_code(input_params: GenerateCodeInput, state: CodeAgentState) -> str
```

## 🧪 Testing

Run the examples to test the code agent:

```bash
python code_agent_example.py
```

## 📝 Best Practices

1. **Start with Analysis**: Always analyze before suggesting changes
2. **Use Appropriate Thresholds**: Adjust severity thresholds based on needs
3. **Cache Results**: The agent caches analysis results for efficiency
4. **Provide Context**: Give detailed prompts for better results
5. **Validate Changes**: Always validate syntax after modifications

## 🤝 Contributing

The code agent follows the existing codebase patterns:
- Modern Python 3.12+ syntax
- Comprehensive docstrings
- Type hints throughout
- Pydantic v2 validation
- Security-first design

## 📄 License

Part of the pydantic-ai-learn project.

## 🙏 Acknowledgments

Built on:
- PydanticAI framework
- Existing toolkit infrastructure
- Python AST module
- Pydantic v2
