# Code Agent Architecture

## Overview

The Code Agent has been refactored into a modular, maintainable architecture following SOLID principles and modern Python best practices.

## Directory Structure

```
code_agent/
├── core/                    # Core agent functionality
│   ├── __init__.py
│   ├── agent.py            # Main CodeAgent class
│   ├── config.py           # AgentConfig dataclass
│   └── types.py            # Type definitions
├── config/                 # Configuration management
│   ├── __init__.py
│   ├── manager.py          # ConfigManager class
│   └── mcp.py              # MCP configuration
├── tools/                  # Code analysis and generation tools
│   ├── __init__.py
│   ├── analyzer.py         # CodeAnalyzer class
│   ├── refactoring.py      # RefactoringEngine class
│   └── generator.py        # CodeGenerator class
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── logging.py          # StructuredLogger class
│   └── errors.py           # Error handling utilities
├── adapters/               # External system adapters
│   ├── __init__.py
│   ├── context.py          # ContextAdapter class
│   ├── workflow.py         # WorkflowAdapter class
│   └── graph.py            # GraphAdapter class
├── tests/                  # Comprehensive test suite
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_config.py
│   ├── test_tools.py
│   ├── test_utils.py
│   └── test_adapters.py
├── examples/               # Usage examples
│   ├── __init__.py
│   ├── basic_usage.py
│   └── advanced_features.py
└── __init__.py            # Package initialization
```

## Module Descriptions

### Core Module (`core/`)

**Responsibility**: Main agent functionality and orchestration

**Key Classes**:
- `CodeAgent`: Main agent class for code analysis and manipulation
- `AgentConfig`: Configuration dataclass with all agent settings
- `AgentState`: State tracking for agent execution

**Key Types**:
- `AnalysisResult`: Result of code analysis
- `RefactoringResult`: Result of refactoring suggestions
- `CodeGenerationResult`: Result of code generation

### Config Module (`config/`)

**Responsibility**: Configuration management and loading

**Key Classes**:
- `ConfigManager`: Load and manage configuration from multiple sources
- `MCPConfigLoader`: Load MCP (Model Context Protocol) configuration
- `MCPConfig`: MCP configuration dataclass

**Features**:
- YAML/JSON file loading
- Environment variable substitution
- Nested configuration access
- Configuration validation

### Tools Module (`tools/`)

**Responsibility**: Code analysis and manipulation tools

**Key Classes**:
- `CodeAnalyzer`: Analyze Python code for metrics, patterns, and dependencies
- `RefactoringEngine`: Detect code smells and suggest refactoring
- `CodeGenerator`: Generate code templates and stubs

**Features**:
- AST-based code analysis
- Complexity detection
- Pattern recognition
- Code generation templates

### Utils Module (`utils/`)

**Responsibility**: Utility functions and error handling

**Key Classes**:
- `StructuredLogger`: JSON and human-readable logging
- `ErrorContext`: Error information and context
- `CircuitBreaker`: Circuit breaker pattern implementation
- `RetryStrategy`: Exponential backoff retry logic
- `ErrorDiagnosisEngine`: Automatic error diagnosis

**Features**:
- Structured logging with sanitization
- Error categorization (transient/permanent/recoverable/fatal)
- Circuit breaker pattern
- Exponential backoff with jitter
- Performance metrics tracking

### Adapters Module (`adapters/`)

**Responsibility**: Integration with external systems

**Key Classes**:
- `ContextAdapter`: Context window management with pruning
- `WorkflowAdapter`: Workflow orchestration and execution
- `GraphAdapter`: Graph-based workflow orchestration

**Features**:
- Context segment management
- Workflow step execution
- Graph node and edge management
- Execution metrics tracking

## Design Patterns

### 1. Separation of Concerns
Each module has a single, well-defined responsibility:
- Core: Agent orchestration
- Config: Configuration management
- Tools: Code analysis
- Utils: Error handling and logging
- Adapters: External integrations

### 2. Dependency Injection
Components receive their dependencies through constructors:
```python
agent = CodeAgent(config=custom_config)
logger = StructuredLogger(name="app", level=LogLevel.DEBUG)
```

### 3. Factory Pattern
Factory functions create configured instances:
```python
agent = create_code_agent(model="openai:gpt-4", enable_streaming=True)
```

### 4. Strategy Pattern
Pluggable strategies for different behaviors:
```python
adapter.prune(strategy="recency")  # or "importance", "sliding_window"
```

### 5. Circuit Breaker Pattern
Fault tolerance through circuit breakers:
```python
breaker = CircuitBreaker(name="api", failure_threshold=5)
result = breaker.call(api_function)
```

## Data Flow

```
User Input
    ↓
CodeAgent (core/agent.py)
    ├→ ConfigManager (config/manager.py) - Load configuration
    ├→ StructuredLogger (utils/logging.py) - Log operations
    ├→ CodeAnalyzer (tools/analyzer.py) - Analyze code
    ├→ RefactoringEngine (tools/refactoring.py) - Suggest refactoring
    ├→ CodeGenerator (tools/generator.py) - Generate code
    ├→ ErrorDiagnosisEngine (utils/errors.py) - Handle errors
    ├→ CircuitBreaker (utils/errors.py) - Fault tolerance
    ├→ ContextAdapter (adapters/context.py) - Manage context
    ├→ WorkflowAdapter (adapters/workflow.py) - Orchestrate workflow
    └→ GraphAdapter (adapters/graph.py) - Graph orchestration
    ↓
Output/Result
```

## SOLID Principles

### Single Responsibility
Each class has one reason to change:
- `CodeAnalyzer`: Changes only when analysis logic changes
- `StructuredLogger`: Changes only when logging format changes
- `ConfigManager`: Changes only when configuration loading changes

### Open/Closed
Classes are open for extension, closed for modification:
- Strategy pattern for pruning strategies
- Pluggable error handlers
- Extensible configuration loaders

### Liskov Substitution
Subtypes can be used interchangeably:
- Different adapters implement same interface
- Different loggers implement same interface

### Interface Segregation
Clients depend on specific interfaces:
- `CodeAnalyzer` doesn't depend on logging
- `StructuredLogger` doesn't depend on analysis

### Dependency Inversion
Depend on abstractions, not concretions:
- Inject configuration objects
- Inject logger instances
- Inject adapter instances

## Testing Strategy

Comprehensive test suite organized by module:
- `test_core.py`: Agent and configuration tests
- `test_config.py`: Configuration loading tests
- `test_tools.py`: Analysis and generation tests
- `test_utils.py`: Logging and error handling tests
- `test_adapters.py`: Adapter integration tests

Run tests:
```bash
pytest code_agent/tests/ -v
```

## Backward Compatibility

The refactored architecture maintains backward compatibility:
- Original `CodeAgent` class still available
- All original exports maintained in `__init__.py`
- New modules available as `core.CodeAgent`, `tools.CodeAnalyzer`, etc.

## Migration Guide

### From Old to New Architecture

**Old**:
```python
from code_agent import CodeAgent
agent = CodeAgent()
```

**New** (recommended):
```python
from code_agent.core import create_code_agent
agent = create_code_agent(model="openai:gpt-4")
```

**New** (with configuration):
```python
from code_agent.core import CodeAgent, AgentConfig
config = AgentConfig(model="openai:gpt-4", enable_streaming=True)
agent = CodeAgent(config=config)
```

## Performance Considerations

1. **Lazy Loading**: Modules loaded on demand
2. **Caching**: Analysis results cached to avoid recomputation
3. **Streaming**: Support for streaming responses
4. **Context Pruning**: Automatic context window management
5. **Circuit Breaker**: Prevents cascading failures

## Future Enhancements

1. Plugin system for custom tools
2. Distributed execution support
3. Advanced caching strategies
4. Performance profiling tools
5. Integration with more LLM providers
