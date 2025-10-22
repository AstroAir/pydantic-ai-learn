# File System Tools for PydanticAI

A production-grade implementation of three essential file system tools designed for seamless integration with PydanticAI agents. These tools provide robust file pattern matching, high-performance search, and directory listing capabilities.

## 🛠️ Tools Overview

### 1. Glob Tool (`glob_tool`)
Fast file pattern matching with modification time sorting.

**Features:**
- Supports glob patterns like `**/*.py`, `src/**/*.ts`, `*.{js,ts}`
- Results sorted by modification time (newest first)
- Recursive patterns with `**` for deep directory traversal
- Configurable result limits

**Parameters:**
- `pattern` (required): Glob pattern to match files
- `path` (optional): Directory to search in (defaults to current directory)
- `max_results` (optional): Maximum number of results (default: 1000)

### 2. Grep Tool (`grep_tool`)
High-performance regex search using ripgrep.

**Features:**
- Multiple output modes: `content`, `files_with_matches`, `count`
- Context lines before/after matches
- File filtering with glob patterns and file types
- Case-insensitive search option
- Multiline matching support

**Parameters:**
- `pattern` (required): Regular expression pattern to search for
- `path` (optional): File or directory to search in
- `glob` (optional): Glob pattern to filter files
- `output_mode` (optional): Output mode (default: `files_with_matches`)
- `context_before`, `context_after`, `context` (optional): Context lines
- `line_number` (optional): Show line numbers
- `ignore_case` (optional): Case-insensitive search
- `file_type` (optional): File type filter
- `head_limit` (optional): Limit output to first N entries
- `multiline` (optional): Enable multiline matching
- `timeout` (optional): Timeout in seconds (default: 30)

### 3. LS Tool (`ls_tool`)
Directory listing with metadata and ignore patterns.

**Features:**
- Lists files and directories with metadata
- File sizes and modification times
- Ignore patterns to exclude unwanted files
- Structured output with type information

**Parameters:**
- `path` (required): Absolute path to directory
- `ignore` (optional): Patterns to exclude from results
- `max_results` (optional): Maximum number of results (default: 1000)

## 🚀 Usage Examples

### Standalone Usage

```python
from tools.filesystem_tools import glob_files, GlobInput

# Find all Python files
result = glob_files(GlobInput(
    pattern="**/*.py",
    path="./src",
    max_results=10
))

print(f"Found {result.total_count} Python files:")
for file_path in result.files:
    print(f"  - {file_path}")
```

### PydanticAI Agent Integration

```python
from pydantic_ai import Agent
from tools.filesystem_tools import glob_tool, grep_tool, ls_tool

# Create an agent with filesystem tools
agent = Agent(
    "openai:gpt-4",
    system_prompt="You are a code explorer with access to filesystem tools."
)

# Register tools
agent.include_tools(glob_tool)
agent.include_tools(grep_tool)
agent.include_tools(ls_tool)

# Use the agent
async def main():
    result = await agent.run("Find all Python files and search for TODO comments")
    print(result.data)
```

### Advanced Examples

#### Search for TODO comments with context
```python
from tools.filesystem_tools import grep_search, GrepInput

result = grep_search(GrepInput(
    pattern="TODO|FIXME|XXX",
    path="./src",
    glob="*.py",
    output_mode="content",
    context=2,
    line_number=True,
    ignore_case=True
))

for match in result.matches:
    print(f"{match.file_path}:{match.line_number}")
    print(f"  {match.line_content}")
```

#### List directory while excluding build artifacts
```python
from tools.filesystem_tools import ls_directory, LSInput

result = ls_directory(LSInput(
    path="/path/to/project",
    ignore=["*.pyc", "__pycache__", ".git", "node_modules", ".venv"]
))

print(f"Contents of {result.directory_path}:")
for entry in result.entries:
    if entry.is_directory:
        print(f"  📁 {entry.name}/")
    else:
        size = f"{entry.size:,} bytes" if entry.size else "unknown size"
        print(f"  📄 {entry.name} ({size})")
```

## 📦 Installation

### Dependencies
- Python 3.12+
- pydantic-ai>=1.0.15
- ripgrep (rg) for the grep tool

### Install ripgrep
```bash
# macOS
brew install ripgrep

# Ubuntu/Debian
apt install ripgrep

# Windows
choco install ripgrep

# Or with cargo
cargo install ripgrep
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_filesystem_tools.py -v
```

The test suite covers:
- Input validation for all tools
- Happy path scenarios
- Edge cases and error handling
- Different output modes
- Cross-platform compatibility

## 📋 API Reference

### Core Functions

#### `glob_files(input_params: GlobInput) -> GlobResult`
Find files matching a glob pattern.

#### `grep_search(input_params: GrepInput) -> GrepResult`
Search for patterns in files using ripgrep.

#### `ls_directory(input_params: LSInput) -> LSResult`
List directory contents.

### Input Models

#### `GlobInput`
- `pattern`: str - Glob pattern
- `path`: Optional[str] - Search directory
- `max_results`: Optional[int] - Result limit

#### `GrepInput`
- `pattern`: str - Regex pattern
- `path`: Optional[str] - Search path
- `glob`: Optional[str] - File filter
- `output_mode`: Literal["content", "files_with_matches", "count"]
- Context and formatting options...

#### `LSInput`
- `path`: str - Absolute directory path
- `ignore`: Optional[List[str]] - Exclude patterns
- `max_results`: Optional[int] - Result limit

### Result Models

#### `GlobResult`
- `files`: List[str] - Matching file paths
- `total_count`: int - Total files found
- `truncated`: bool - If results were limited
- `pattern`: str - Pattern used
- `search_path`: str - Directory searched

#### `GrepResult`
- `output_mode`: str - Mode used
- `pattern`: str - Pattern searched
- Varying result fields based on output mode...

#### `LSResult`
- `entries`: List[LSEntry] - Directory entries
- `total_count`: int - Total entries found
- `truncated`: bool - If results were limited
- `directory_path`: str - Directory listed

## 🔧 Error Handling

The tools include comprehensive error handling:

- **GlobError**: File pattern matching errors
- **GrepError**: Search operation errors
- **LSError**: Directory listing errors
- **RipgrepNotFoundError**: ripgrep not installed
- **ValidationError**: Input validation failures

All tool functions gracefully handle errors and return informative error messages when used with PydanticAI agents.

## 🛡️ Security Considerations

- **Path validation**: All paths are normalized and validated to prevent traversal attacks
- **Input validation**: Comprehensive Pydantic v2 validation prevents malicious inputs
- **Safe subprocess execution**: Uses list arguments (not shell=True) to prevent injection
- **Resource limits**: Built-in limits prevent resource exhaustion attacks

## 🔄 Compatibility

- **Python**: 3.12+
- **PydanticAI**: 1.0.15+
- **Platforms**: Windows, macOS, Linux
- **Dependencies**: Minimal external dependencies

## 📝 Examples

See `tools/filesystem_tools_example.py` for comprehensive usage examples including:

- Basic usage patterns
- Advanced search techniques
- Error handling scenarios
- PydanticAI agent integration
- Performance optimization tips

## 🤝 Contributing

The implementation follows modern Python best practices:

- Type hints throughout
- Comprehensive docstrings
- Full test coverage
- Cross-platform compatibility
- Security-focused design

## 📄 License

This implementation follows the same license as the parent project.
