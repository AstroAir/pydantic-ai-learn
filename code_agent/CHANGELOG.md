# Changelog - Code Agent

All notable changes to the Code Agent project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-10-15

### 🎉 Major Release - Enhanced with PydanticAI Advanced Features

This release represents a significant enhancement of the Code Agent with advanced PydanticAI patterns while maintaining full backward compatibility.

### ✨ Added

#### Streaming Support
- **Real-time Analysis Feedback**: Stream analysis results as they're generated
- **Event Handlers**: Custom event handlers for streaming events
- **Progress Updates**: Live progress updates during long-running analysis
- **`run_stream()` Method**: New async streaming method
- **`analyze_code_streaming()` Function**: Streaming-enabled analysis function

#### Async Iteration
- **Execution Node Access**: Iterate over agent execution nodes
- **`iter_nodes()` Method**: Access intermediate execution steps
- **Fine-grained Control**: Better control over execution flow

#### Usage Limits and Tracking
- **Token Limits**: Set response token limits to control costs
- **Usage Tracking**: Track input/output tokens and request counts
- **`UsageLimits` Integration**: Full PydanticAI usage limits support
- **`get_usage_summary()` Method**: Get formatted usage statistics
- **Automatic Usage Updates**: Usage tracked automatically after each request

#### Retry Logic
- **Automatic Retries**: Tools automatically retry on failure (up to 2 times)
- **`ModelRetry` Integration**: Intelligent retry with feedback to the model
- **Enhanced Tool Wrappers**: All analysis tools have retry-enabled versions
  - `analyze_code_with_retry()`
  - `validate_syntax_with_retry()`
  - `detect_patterns_with_retry()`

#### Conversation History
- **Multi-turn Conversations**: Maintain context across multiple interactions
- **Message History Management**: Store and retrieve conversation history
- **`message_history` Parameter**: Pass history to `run_sync()` and `run()`
- **`clear_history()` Method**: Clear conversation history
- **`get_message_history()` Method**: Retrieve message history

#### Package Structure
- **Proper Python Package**: Organized as `code_agent/` package
- **Clean Exports**: Well-defined `__init__.py` with all exports
- **Modular Design**: Separated into logical modules
  - `agent.py`: Main CodeAgent class
  - `toolkit.py`: Enhanced analysis toolkit
  - `examples.py`: Comprehensive examples
- **Package Metadata**: Version, author, description

#### Convenience Functions
- **`quick_analyze()`**: Quick code analysis without creating agent
- **`quick_refactor()`**: Quick refactoring suggestions
- **`create_code_agent()`**: Factory function for agent creation

#### Documentation
- **Enhanced README**: Updated with v2.0 features
- **Migration Guide**: Complete migration guide from v1.0
- **Package Index**: Comprehensive package index
- **Changelog**: This file
- **Import Tests**: Automated import verification

### 🔄 Changed

#### Enhanced State Management
- **`CodeAgentState`**: Extended with message history and usage tracking
- **New State Methods**:
  - `add_message()`: Add message to history
  - `clear_history()`: Clear conversation history
  - `update_usage()`: Update usage statistics
  - `get_usage_summary()`: Get formatted usage summary

#### Enhanced CodeAgent Class
- **New Constructor Parameters**:
  - `usage_limits`: Optional usage limits
  - `enable_streaming`: Enable streaming by default
- **Enhanced Methods**:
  - `run_sync()`: Now supports message history and usage limits
  - `run()`: Now supports message history and usage limits
- **New Methods**:
  - `run_stream()`: Stream analysis results
  - `iter_nodes()`: Iterate over execution nodes
  - `get_usage_summary()`: Get usage statistics
  - `clear_history()`: Clear conversation history
  - `get_message_history()`: Get message history

#### Tool Registration
- **Retry Support**: All tools registered with `retries=2`
- **ModelRetry Integration**: Tools raise `ModelRetry` for intelligent retries
- **Enhanced Error Messages**: Better error messages for debugging

### 📁 File Structure

#### New Files
- `code_agent/__init__.py`: Package initialization
- `code_agent/agent.py`: Enhanced CodeAgent class
- `code_agent/toolkit.py`: Enhanced toolkit
- `code_agent/examples.py`: Enhanced examples
- `code_agent/MIGRATION.md`: Migration guide
- `code_agent/INDEX.md`: Package index
- `code_agent/CHANGELOG.md`: This file
- `code_agent/test_imports.py`: Import verification tests

#### Moved Files
- `README_CODE_AGENT.md` → `code_agent/README.md`
- `CODE_AGENT_QUICKSTART.md` → `code_agent/QUICKSTART.md`
- `tools/README_CODE_AGENT_TOOLKIT.md` → `code_agent/TOOLKIT.md`

#### Deprecated Files (Still Available)
- `code_agent.py` (root): Use `code_agent/agent.py` instead
- `code_agent_example.py` (root): Use `code_agent/examples.py` instead

### ✅ Backward Compatibility

**100% Backward Compatible**: All v1.0 code works without modifications.

```python
# v1.0 code (still works!)
from code_agent import CodeAgent

agent = CodeAgent()
result = agent.run_sync("Analyze my_module.py")
print(result.output)
```

### 🔒 Security

- No breaking changes to security model
- All existing security features maintained
- Enhanced error handling for better security

### 📊 Performance

- Streaming reduces perceived latency
- Usage tracking helps optimize costs
- Retry logic improves reliability
- Conversation history enables context-aware analysis

### 🐛 Bug Fixes

- None (new features only)

### 🗑️ Deprecated

- None (all v1.0 features maintained)

### ❌ Removed

- None (full backward compatibility)

---

## [1.0.0] - 2025-10-14

### Initial Release

#### Added
- **Core Code Analysis**: Parse and analyze Python code using AST
- **Pattern Detection**: Identify code smells and anti-patterns
- **Quality Metrics**: Calculate complexity and maintainability metrics
- **Dependency Analysis**: Analyze imports and dependencies
- **Syntax Validation**: Check Python syntax
- **Refactoring Suggestions**: Propose code improvements
- **Code Generation**: Generate Python code from descriptions
- **PydanticAI Integration**: Built on PydanticAI framework
- **Tool Integration**: Integrates with existing toolkits
- **Comprehensive Documentation**: README, Quick Start, Toolkit docs
- **Examples**: 10 comprehensive examples

#### Features
- Modern Python 3.12+ with type hints
- Pydantic v2 validation
- AST-based analysis (no code execution)
- Security-focused design
- Comprehensive error handling
- Extensive documentation

---

## Version Comparison

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Basic Analysis | ✅ | ✅ |
| Pattern Detection | ✅ | ✅ |
| Code Generation | ✅ | ✅ |
| Refactoring Suggestions | ✅ | ✅ |
| Streaming Support | ❌ | ✅ |
| Async Iteration | ❌ | ✅ |
| Usage Limits | ❌ | ✅ |
| Retry Logic | ❌ | ✅ |
| Conversation History | ❌ | ✅ |
| Package Structure | ❌ | ✅ |
| Backward Compatible | N/A | ✅ |

---

**For migration instructions, see [MIGRATION.md](MIGRATION.md)**
