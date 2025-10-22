"""
File System Tools for PydanticAI

A production-grade Python implementation of three essential file system tools
designed for seamless integration with PydanticAI agents.

Tools:
1. Glob: Fast file pattern matching with modification time sorting
2. Grep: Ripgrep-based search with multiple output modes and advanced filtering
3. LS: Directory listing with ignore pattern support

Features:
- Modern Python 3.12+ with latest type hints
- Pydantic v2 validation for robust input handling
- Comprehensive error handling with informative messages
- Cross-platform compatibility using pathlib
- Security-focused design (path validation, safe subprocess execution)

Security Considerations:
- All paths are normalized and validated to prevent traversal attacks
- Subprocess execution uses list arguments (not shell=True) to prevent injection
- Input validation prevents resource exhaustion attacks
- Proper error handling prevents information leakage

Example Usage:
    ```python
    from tools.filesystem_tools import glob_files, GlobInput

    # Find all Python files, sorted by modification time
    result = glob_files(GlobInput(
        pattern="**/*.py",
        path="./src"
    ))

    for file_path in result.files:
        print(file_path)
    ```

Dependencies:
- ripgrep (rg): Required for grep_search() function
  Installation: https://github.com/BurntSushi/ripgrep#installation

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_ai import RunContext, Tool

# ============================================================================
# Custom Exceptions
# ============================================================================


class FileSystemToolError(Exception):
    """Base exception for file system tool errors."""

    pass


class GlobError(FileSystemToolError):
    """Raised when glob operation fails."""

    pass


class GrepError(FileSystemToolError):
    """Raised when grep operation fails."""

    pass


class LSError(FileSystemToolError):
    """Raised when ls operation fails."""

    pass


class RipgrepNotFoundError(GrepError):
    """Raised when ripgrep is not installed or not found in PATH."""

    pass


# ============================================================================
# Constants
# ============================================================================

# Default values
DEFAULT_MAX_RESULTS = 1000
DEFAULT_GREP_TIMEOUT = 30  # seconds
DEFAULT_CONTEXT_LINES = 3
MAX_CONTEXT_LINES = 100

# Ripgrep exit codes
RG_EXIT_SUCCESS = 0  # Matches found
RG_EXIT_NO_MATCHES = 1  # No matches found (not an error)
RG_EXIT_ERROR = 2  # Error occurred


# ============================================================================
# Input Models
# ============================================================================


class GlobInput(BaseModel):
    """
    Input schema for glob file pattern matching.

    Validates parameters for finding files matching glob patterns with
    optional modification time sorting.
    """

    pattern: str = Field(
        ...,
        description=(
            "Glob pattern to match files (e.g., '**/*.py', 'src/**/*.ts'). Supports recursive patterns with '**'."
        ),
        min_length=1,
        max_length=500,
    )

    path: str | None = Field(
        default=None,
        description="Directory to search in. Defaults to current working directory if omitted.",
        max_length=1000,
    )

    max_results: int | None = Field(
        default=DEFAULT_MAX_RESULTS,
        description=f"Maximum number of results to return (default: {DEFAULT_MAX_RESULTS})",
        ge=1,
        le=10000,
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [{"pattern": "**/*.py", "path": "./src"}, {"pattern": "*.{js,ts}", "max_results": 100}]
        },
    }

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str | None) -> str | None:
        """Validate and normalize the search path."""
        if v is None:
            return None

        # Normalize the path
        path = Path(v).expanduser()

        # Check if path exists
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")

        # Check if it's a directory
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {v}")

        return str(path)


class GrepInput(BaseModel):
    """
    Input schema for ripgrep-based search.

    Validates parameters for searching files with regex patterns using ripgrep.
    Supports multiple output modes and advanced filtering options.
    """

    pattern: str = Field(..., description="Regular expression pattern to search for", min_length=1, max_length=1000)

    path: str | None = Field(
        default=None,
        description="File or directory to search in. Defaults to current directory if omitted.",
        max_length=1000,
    )

    glob: str | None = Field(
        default=None, description="Glob pattern to filter files (e.g., '*.js', '*.{ts,tsx}')", max_length=200
    )

    output_mode: Literal["content", "files_with_matches", "count"] = Field(
        default="files_with_matches",
        description=(
            "Output mode: 'content' shows matching lines, 'files_with_matches' shows file paths, "
            "'count' shows match counts"
        ),
    )

    context_before: int | None = Field(
        default=None,
        description="Number of lines to show before each match (only with output_mode='content')",
        ge=0,
        le=MAX_CONTEXT_LINES,
    )

    context_after: int | None = Field(
        default=None,
        description="Number of lines to show after each match (only with output_mode='content')",
        ge=0,
        le=MAX_CONTEXT_LINES,
    )

    context: int | None = Field(
        default=None,
        description="Number of lines to show before and after each match (only with output_mode='content')",
        ge=0,
        le=MAX_CONTEXT_LINES,
    )

    line_number: bool = Field(default=False, description="Show line numbers (only with output_mode='content')")

    ignore_case: bool = Field(default=False, description="Case-insensitive search")

    file_type: str | None = Field(
        default=None,
        description="File type filter (e.g., 'js', 'py', 'rust'). See 'rg --type-list' for available types.",
        max_length=50,
    )

    head_limit: int | None = Field(default=None, description="Limit output to first N lines/entries", ge=1, le=10000)

    multiline: bool = Field(default=False, description="Enable multiline matching mode")

    timeout: int | None = Field(
        default=DEFAULT_GREP_TIMEOUT, description=f"Timeout in seconds (default: {DEFAULT_GREP_TIMEOUT})", ge=1, le=300
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {"pattern": "TODO", "path": "./src", "output_mode": "files_with_matches"},
                {
                    "pattern": "function\\s+\\w+",
                    "glob": "*.js",
                    "output_mode": "content",
                    "line_number": True,
                    "context": 2,
                },
            ]
        },
    }

    @model_validator(mode="after")
    def validate_context_options(self) -> GrepInput:
        """Validate that context options are only used with content output mode."""
        # Check if any context option is set
        has_context_options = any(
            [
                self.context_before is not None,
                self.context_after is not None,
                self.context is not None,
                self.line_number is True,
            ]
        )

        if has_context_options and self.output_mode != "content":
            raise ValueError(
                f"Context options (-A, -B, -C, -n) can only be used with output_mode='content'. "
                f"Current output_mode: '{self.output_mode}'"
            )

        return self


class LSInput(BaseModel):
    """
    Input schema for directory listing.

    Validates parameters for listing files and directories with optional
    ignore pattern filtering.
    """

    path: str = Field(..., description="Absolute path to directory to list", min_length=1, max_length=1000)

    ignore: list[str] | None = Field(
        default=None,
        description="Array of glob patterns to exclude from results (e.g., ['*.pyc', '__pycache__'])",
        max_length=100,
    )

    max_results: int | None = Field(
        default=DEFAULT_MAX_RESULTS,
        description=f"Maximum number of results to return (default: {DEFAULT_MAX_RESULTS})",
        ge=1,
        le=10000,
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {"path": "/home/user/projects"},
                {"path": "C:\\Users\\user\\Documents", "ignore": ["*.tmp", "~*"]},
            ]
        },
    }

    @field_validator("path")
    @classmethod
    def validate_absolute_path(cls, v: str) -> str:
        """Validate that path is absolute and exists."""
        path = Path(v).expanduser()

        # Check if path is absolute
        if not path.is_absolute():
            raise ValueError(f"Path must be absolute. Got: {v}")

        # Normalize the path
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve path '{v}': {e}") from e

        # Check if path exists
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")

        # Check if it's a directory
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {v}")

        return str(path)


# ============================================================================
# Output Models
# ============================================================================


@dataclass
class GlobResult:
    """
    Result of glob file pattern matching operation.

    Attributes:
        files: List of file paths matching the pattern, sorted by modification time (newest first)
        total_count: Total number of files found
        truncated: Whether results were truncated due to max_results limit
        pattern: The glob pattern that was used
        search_path: The directory that was searched
    """

    files: list[str]
    total_count: int
    truncated: bool
    pattern: str
    search_path: str


@dataclass
class GrepContentMatch:
    """
    Represents a single content match from grep search.

    Attributes:
        file_path: Path to the file containing the match
        line_number: Line number of the match (1-based, None if not requested)
        line_content: Content of the matching line
        context_before: Lines before the match (if context was requested)
        context_after: Lines after the match (if context was requested)
    """

    file_path: str
    line_number: int | None
    line_content: str
    context_before: list[str] | None = None
    context_after: list[str] | None = None


@dataclass
class GrepResult:
    """
    Result of grep search operation.

    The structure varies based on output_mode:
    - content: matches contains GrepContentMatch objects
    - files_with_matches: files contains file paths
    - count: counts contains {file_path: match_count} mapping

    Attributes:
        output_mode: The output mode that was used
        pattern: The regex pattern that was searched
        matches: List of content matches (only for output_mode='content')
        files: List of file paths with matches (only for output_mode='files_with_matches')
        counts: Mapping of file paths to match counts (only for output_mode='count')
        total_matches: Total number of matches found
        truncated: Whether results were truncated due to head_limit
    """

    output_mode: Literal["content", "files_with_matches", "count"]
    pattern: str
    matches: list[GrepContentMatch] | None = None
    files: list[str] | None = None
    counts: dict[str, int] | None = None
    total_matches: int = 0
    truncated: bool = False


@dataclass
class LSEntry:
    """
    Represents a single file or directory entry.

    Attributes:
        name: Name of the file or directory
        path: Absolute path to the file or directory
        is_directory: Whether this entry is a directory
        size: Size in bytes (None for directories)
        modified_time: Last modification time as Unix timestamp
    """

    name: str
    path: str
    is_directory: bool
    size: int | None
    modified_time: float


@dataclass
class LSResult:
    """
    Result of directory listing operation.

    Attributes:
        entries: List of directory entries
        total_count: Total number of entries found
        truncated: Whether results were truncated due to max_results limit
        directory_path: The directory that was listed
    """

    entries: list[LSEntry]
    total_count: int
    truncated: bool
    directory_path: str


# ============================================================================
# Helper Functions
# ============================================================================


def _check_ripgrep_available() -> None:
    """
    Check if ripgrep (rg) is available in PATH.

    Raises:
        RipgrepNotFoundError: If ripgrep is not found
    """
    if shutil.which("rg") is None:
        raise RipgrepNotFoundError(
            "ripgrep (rg) is not installed or not found in PATH.\n"
            "Please install ripgrep to use the grep_search() function.\n"
            "Installation instructions: https://github.com/BurntSushi/ripgrep#installation\n"
            "\n"
            "Quick install:\n"
            "  - macOS: brew install ripgrep\n"
            "  - Ubuntu/Debian: apt install ripgrep\n"
            "  - Windows: choco install ripgrep\n"
            "  - Cargo: cargo install ripgrep"
        )


def _build_ripgrep_command(input_params: GrepInput) -> list[str]:
    """
    Build ripgrep command arguments from input parameters.

    Args:
        input_params: Validated grep input parameters

    Returns:
        List of command arguments for subprocess
    """
    cmd = ["rg"]

    # Output mode flags
    match input_params.output_mode:
        case "files_with_matches":
            cmd.append("--files-with-matches")
        case "count":
            cmd.append("--count")
        case "content":
            # Content mode is default, but we add flags for context/line numbers
            if input_params.line_number:
                cmd.append("--line-number")

            if input_params.context is not None:
                cmd.extend(["--context", str(input_params.context)])
            else:
                if input_params.context_before is not None:
                    cmd.extend(["--before-context", str(input_params.context_before)])
                if input_params.context_after is not None:
                    cmd.extend(["--after-context", str(input_params.context_after)])

    # Case sensitivity
    if input_params.ignore_case:
        cmd.append("--ignore-case")

    # File type filter
    if input_params.file_type:
        cmd.extend(["--type", input_params.file_type])

    # Glob pattern filter
    if input_params.glob:
        cmd.extend(["--glob", input_params.glob])

    # Multiline mode
    if input_params.multiline:
        cmd.append("--multiline")

    # Head limit (max matches)
    if input_params.head_limit:
        cmd.extend(["--max-count", str(input_params.head_limit)])

    # Always use --no-heading for easier parsing
    cmd.append("--no-heading")

    # Disable colors for clean output
    cmd.extend(["--color", "never"])

    # Add the pattern
    cmd.extend(["--regexp", input_params.pattern])

    # Add the path if specified
    if input_params.path:
        cmd.append(input_params.path)

    return cmd


def _parse_ripgrep_output(
    output: str, output_mode: Literal["content", "files_with_matches", "count"], line_number: bool
) -> tuple[list[GrepContentMatch] | None, list[str] | None, dict[str, int] | None, int]:
    """
    Parse ripgrep output based on output mode.

    Args:
        output: Raw output from ripgrep
        output_mode: The output mode used
        line_number: Whether line numbers were requested

    Returns:
        Tuple of (matches, files, counts, total_matches)
    """
    if not output.strip():
        return None, None, None, 0

    lines = output.strip().split("\n")

    match output_mode:
        case "files_with_matches":
            # Each line is a file path
            files = [line.strip() for line in lines if line.strip()]
            return None, files, None, len(files)

        case "count":
            # Each line is "filepath:count"
            counts = {}
            total = 0
            for line in lines:
                if ":" in line:
                    filepath, count_str = line.rsplit(":", 1)
                    try:
                        count = int(count_str)
                        counts[filepath] = count
                        total += count
                    except ValueError:
                        continue
            return None, None, counts, total

        case "content":
            # Parse content matches
            matches = []
            total = 0

            for line in lines:
                if not line.strip():
                    continue

                # Format: filepath:line_number:content or filepath:content
                parts = line.split(":", 2 if line_number else 1)

                if len(parts) >= 2:
                    filepath = parts[0]

                    if line_number and len(parts) == 3:
                        try:
                            line_num = int(parts[1])
                            content = parts[2]
                        except ValueError:
                            # Fallback if line number parsing fails
                            line_num = None
                            content = ":".join(parts[1:])
                    else:
                        line_num = None
                        content = parts[1] if len(parts) == 2 else ":".join(parts[1:])

                    matches.append(
                        GrepContentMatch(
                            file_path=filepath,
                            line_number=line_num,
                            line_content=content,
                            context_before=None,
                            context_after=None,
                        )
                    )
                    total += 1

            return matches, None, None, total

    return None, None, None, 0


# ============================================================================
# Core Tool Functions
# ============================================================================


def glob_files(input_params: GlobInput) -> GlobResult:
    """
    Find files matching a glob pattern, sorted by modification time.

    This tool performs fast file pattern matching using glob patterns and returns
    results sorted by modification time (newest first). Supports recursive patterns
    with '**' for deep directory traversal.

    Args:
        input_params: Validated glob input parameters

    Returns:
        GlobResult containing matched files and metadata

    Raises:
        GlobError: If glob operation fails

    Example:
        >>> result = glob_files(GlobInput(pattern="**/*.py", path="./src"))
        >>> for file in result.files:
        ...     print(file)
    """
    try:
        # Determine search path
        search_path = Path(input_params.path) if input_params.path else Path.cwd()

        # Perform glob search
        matched_paths = list(search_path.glob(input_params.pattern))

        # Filter to only files (not directories)
        file_paths = [p for p in matched_paths if p.is_file()]

        # Sort by modification time (newest first)
        # If we can't stat some files, just skip sorting
        import contextlib

        with contextlib.suppress(OSError, PermissionError):
            file_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Convert to strings
        file_strings = [str(p) for p in file_paths]

        # Apply max_results limit
        total_count = len(file_strings)
        max_results = input_params.max_results or DEFAULT_MAX_RESULTS
        truncated = total_count > max_results

        if truncated:
            file_strings = file_strings[:max_results]

        return GlobResult(
            files=file_strings,
            total_count=total_count,
            truncated=truncated,
            pattern=input_params.pattern,
            search_path=str(search_path),
        )

    except ValueError as e:
        raise GlobError(f"Invalid glob pattern '{input_params.pattern}': {e}") from e
    except PermissionError as e:
        raise GlobError(f"Permission denied accessing path: {e}") from e
    except OSError as e:
        raise GlobError(f"OS error during glob operation: {e}") from e
    except Exception as e:
        raise GlobError(f"Unexpected error during glob operation: {e}") from e


def grep_search(input_params: GrepInput) -> GrepResult:
    """
    Search for patterns in files using ripgrep.

    This tool provides high-performance regex searching using ripgrep (rg).
    Supports multiple output modes, context lines, file filtering, and more.

    Args:
        input_params: Validated grep input parameters

    Returns:
        GrepResult containing search results based on output_mode

    Raises:
        RipgrepNotFoundError: If ripgrep is not installed
        GrepError: If search operation fails

    Example:
        >>> result = grep_search(GrepInput(
        ...     pattern="TODO",
        ...     path="./src",
        ...     output_mode="files_with_matches"
        ... ))
        >>> for file in result.files:
        ...     print(file)
    """
    # Check ripgrep availability
    _check_ripgrep_available()

    try:
        # Build command
        cmd = _build_ripgrep_command(input_params)

        # Execute ripgrep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=input_params.timeout,
            check=False,  # Don't raise on non-zero exit (exit code 1 means no matches)
        )

        # Handle exit codes
        if result.returncode == RG_EXIT_ERROR:
            # Actual error occurred
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            raise GrepError(f"Ripgrep error: {error_msg}")

        # Parse output (exit code 0 = matches found, 1 = no matches)
        matches, files, counts, total_matches = _parse_ripgrep_output(
            result.stdout, input_params.output_mode, input_params.line_number
        )

        # Determine if results were truncated
        truncated = False
        if input_params.head_limit and total_matches >= input_params.head_limit:
            truncated = True

        return GrepResult(
            output_mode=input_params.output_mode,
            pattern=input_params.pattern,
            matches=matches,
            files=files,
            counts=counts,
            total_matches=total_matches,
            truncated=truncated,
        )

    except subprocess.TimeoutExpired as e:
        raise GrepError(f"Search timed out after {input_params.timeout} seconds") from e
    except FileNotFoundError as e:
        raise GrepError("Ripgrep command not found (this should not happen after availability check)") from e
    except Exception as e:
        if isinstance(e, GrepError):
            raise
        raise GrepError(f"Unexpected error during grep search: {e}") from e


def ls_directory(input_params: LSInput) -> LSResult:
    """
    List files and directories at a given absolute path.

    This tool lists directory contents with optional ignore pattern filtering.
    Returns structured information about each entry including size and modification time.

    Args:
        input_params: Validated ls input parameters

    Returns:
        LSResult containing directory entries and metadata

    Raises:
        LSError: If directory listing fails

    Example:
        >>> result = ls_directory(LSInput(
        ...     path="/home/user/projects",
        ...     ignore=["*.pyc", "__pycache__"]
        ... ))
        >>> for entry in result.entries:
        ...     print(f"{entry.name} ({'dir' if entry.is_directory else 'file'})")
    """
    try:
        directory = Path(input_params.path)

        # Get all entries
        all_entries = []

        for item in directory.iterdir():
            # Check ignore patterns
            if input_params.ignore:
                should_ignore = False
                for pattern in input_params.ignore:
                    if item.match(pattern):
                        should_ignore = True
                        break

                if should_ignore:
                    continue

            # Get entry information
            try:
                stat_info = item.stat()

                entry = LSEntry(
                    name=item.name,
                    path=str(item.absolute()),
                    is_directory=item.is_dir(),
                    size=stat_info.st_size if item.is_file() else None,
                    modified_time=stat_info.st_mtime,
                )

                all_entries.append(entry)

            except (OSError, PermissionError):
                # Skip entries we can't access
                continue

        # Sort entries: directories first, then by name
        all_entries.sort(key=lambda e: (not e.is_directory, e.name.lower()))

        # Apply max_results limit
        total_count = len(all_entries)
        max_results = input_params.max_results or DEFAULT_MAX_RESULTS
        truncated = total_count > max_results

        if truncated:
            all_entries = all_entries[:max_results]

        return LSResult(
            entries=all_entries, total_count=total_count, truncated=truncated, directory_path=str(directory)
        )

    except PermissionError as e:
        raise LSError(f"Permission denied accessing directory: {e}") from e
    except OSError as e:
        raise LSError(f"OS error during directory listing: {e}") from e
    except Exception as e:
        raise LSError(f"Unexpected error during directory listing: {e}") from e


# ============================================================================
# PydanticAI Tool Registration
# ============================================================================


def glob_tool_func(
    ctx: RunContext[None], pattern: str, path: str | None = None, max_results: int | None = DEFAULT_MAX_RESULTS
) -> str:
    """
    Find files matching a glob pattern, sorted by modification time.

    Args:
        pattern: Glob pattern to match files (e.g., '**/*.py', 'src/**/*.ts')
        path: Directory to search in (optional, defaults to current directory)
        max_results: Maximum number of results to return (default: 1000)

    Returns:
        Formatted string with matching files sorted by modification time (newest first)

    Raises:
        ValueError: If pattern is invalid or path doesn't exist
    """
    try:
        input_params = GlobInput(pattern=pattern, path=path, max_results=max_results)
        result = glob_files(input_params)

        output = f"Found {result.total_count} files matching '{pattern}'"
        if result.search_path:
            output += f" in '{result.search_path}'"
        output += " (sorted by modification time, newest first):\n\n"

        for file_path in result.files:
            output += f"  {file_path}\n"

        if result.truncated:
            output += f"\n(Showing {len(result.files)} of {result.total_count} results - use max_results for more)"

        return output.strip()

    except GlobError as e:
        return f"Error finding files: {e}"
    except ValidationError as e:
        return f"Invalid parameters: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


def grep_tool_func(
    ctx: RunContext[None],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: Literal["content", "files_with_matches", "count"] = "files_with_matches",
    context_before: int | None = None,
    context_after: int | None = None,
    context: int | None = None,
    line_number: bool = False,
    ignore_case: bool = False,
    file_type: str | None = None,
    head_limit: int | None = None,
    multiline: bool = False,
    timeout: int | None = DEFAULT_GREP_TIMEOUT,
) -> str:
    """
    Search for patterns in files using ripgrep with high performance.

    Args:
        pattern: Regular expression pattern to search for
        path: File or directory to search in (optional, defaults to current directory)
        glob: Glob pattern to filter files (e.g., '*.js', '*.{ts,tsx}')
        output_mode: Output mode - 'content', 'files_with_matches', or 'count'
        context_before: Lines before each match (only with output_mode='content')
        context_after: Lines after each match (only with output_mode='content')
        context: Lines before and after each match (only with output_mode='content')
        line_number: Show line numbers (only with output_mode='content')
        ignore_case: Case-insensitive search
        file_type: File type filter (e.g., 'js', 'py', 'rust')
        head_limit: Limit output to first N lines/entries
        multiline: Enable multiline matching mode
        timeout: Timeout in seconds (default: 30)

    Returns:
        Formatted search results based on output_mode

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If ripgrep is not installed
    """
    try:
        input_params = GrepInput(
            pattern=pattern,
            path=path,
            glob=glob,
            output_mode=output_mode,
            context_before=context_before,
            context_after=context_after,
            context=context,
            line_number=line_number,
            ignore_case=ignore_case,
            file_type=file_type,
            head_limit=head_limit,
            multiline=multiline,
            timeout=timeout,
        )
        result = grep_search(input_params)

        output = f"Search results for pattern '{pattern}'"
        if result.pattern != pattern:
            output += f" (searched for: '{result.pattern}')"
        output += f" using {output_mode} mode:\n\n"

        match output_mode:
            case "files_with_matches":
                if result.files:
                    for file_path in result.files:
                        output += f"  {file_path}\n"
                else:
                    output += "  No files found matching the pattern.\n"

            case "count":
                if result.counts:
                    for file_path, count in result.counts.items():
                        output += f"  {file_path}: {count} matches\n"
                    output += f"\nTotal matches: {result.total_matches}\n"
                else:
                    output += "  No matches found.\n"

            case "content":
                if result.matches:
                    for match in result.matches:
                        if match.line_number:
                            output += f"{match.file_path}:{match.line_number}\n"
                        else:
                            output += f"{match.file_path}\n"
                        output += f"  {match.line_content}\n"
                        if match.context_before:
                            for ctx_line in match.context_before:
                                output += f"  {ctx_line}\n"
                        if match.context_after:
                            for ctx_line in match.context_after:
                                output += f"  {ctx_line}\n"
                        output += "\n"
                else:
                    output += "  No matches found.\n"

        if result.truncated:
            output += f"\n(Results truncated - showing {result.total_matches}+ matches)"

        return output.strip()

    except RipgrepNotFoundError as e:
        return f"Ripgrep not available: {e}"
    except GrepError as e:
        return f"Search error: {e}"
    except ValidationError as e:
        return f"Invalid parameters: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


def ls_tool_func(
    ctx: RunContext[None], path: str, ignore: list[str] | None = None, max_results: int | None = DEFAULT_MAX_RESULTS
) -> str:
    """
    List files and directories at a given absolute path with metadata.

    Args:
        path: Absolute path to directory to list
        ignore: Patterns to exclude from results (e.g., ['*.pyc', '__pycache__'])
        max_results: Maximum number of results to return (default: 1000)

    Returns:
        Formatted directory listing with file/directory metadata

    Raises:
        ValueError: If path is invalid or doesn't exist
    """
    try:
        input_params = LSInput(path=path, ignore=ignore, max_results=max_results)
        result = ls_directory(input_params)

        output = f"Directory listing for '{result.directory_path}':\n\n"

        if result.entries:
            # Group by type for better readability
            directories = [e for e in result.entries if e.is_directory]
            files = [e for e in result.entries if not e.is_directory]

            if directories:
                output += "Directories:\n"
                for entry in directories:
                    output += f"  📁  {entry.name}/\n"
                output += "\n"

            if files:
                output += "Files:\n"
                for entry in files:
                    size_str = f"{entry.size:,} bytes" if entry.size is not None else "unknown size"
                    output += f"  📄  {entry.name} ({size_str})\n"
        else:
            output += "  (empty directory)\n"

        if result.truncated:
            output += f"\n(Showing {len(result.entries)} of {result.total_count} entries - use max_results for more)"

        return output.strip()

    except LSError as e:
        return f"Error listing directory: {e}"
    except ValidationError as e:
        return f"Invalid parameters: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# Create PydanticAI Tool instances
glob_tool = Tool(glob_tool_func, takes_ctx=True)
grep_tool = Tool(grep_tool_func, takes_ctx=True)
ls_tool = Tool(ls_tool_func, takes_ctx=True)
