"""
Comprehensive test suite for filesystem_tools module.

Tests cover:
- Input validation for all three tools
- Happy path scenarios
- Edge cases (empty results, errors, permissions)
- Different output modes for Grep
- Cross-field validation
- Error handling

Author: The Augster
Python Version: 3.12+
"""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from tools.filesystem_tools import (
    DEFAULT_MAX_RESULTS,
    GlobError,
    GlobInput,
    GlobResult,
    GrepError,
    GrepInput,
    GrepResult,
    LSInput,
    LSResult,
    RipgrepNotFoundError,
    glob_files,
    grep_search,
    ls_directory,
)

# ============================================================================
# GlobInput Validation Tests
# ============================================================================


class TestGlobInput:
    """Test GlobInput validation."""

    def test_valid_input_minimal(self):
        """Test valid minimal input."""
        input_data = GlobInput(pattern="*.py")
        assert input_data.pattern == "*.py"
        assert input_data.path is None
        assert input_data.max_results == DEFAULT_MAX_RESULTS

    def test_valid_input_full(self, tmp_path):
        """Test valid input with all parameters."""
        input_data = GlobInput(pattern="**/*.js", path=str(tmp_path), max_results=100)
        assert input_data.pattern == "**/*.js"
        assert input_data.max_results == 100

    def test_invalid_empty_pattern(self):
        """Test that empty pattern is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GlobInput(pattern="")

        assert "pattern" in str(exc_info.value).lower()

    def test_invalid_pattern_too_long(self):
        """Test that overly long pattern is rejected."""
        with pytest.raises(ValidationError):
            GlobInput(pattern="a" * 501)

    def test_invalid_path_not_exists(self):
        """Test that non-existent path is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GlobInput(pattern="*.py", path="/nonexistent/path/12345")

        assert "does not exist" in str(exc_info.value).lower()

    def test_invalid_path_not_directory(self, tmp_path):
        """Test that file path (not directory) is rejected."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValidationError) as exc_info:
            GlobInput(pattern="*.py", path=str(test_file))

        assert "not a directory" in str(exc_info.value).lower()

    def test_invalid_max_results_negative(self):
        """Test that negative max_results is rejected."""
        with pytest.raises(ValidationError):
            GlobInput(pattern="*.py", max_results=-1)

    def test_invalid_max_results_too_large(self):
        """Test that overly large max_results is rejected."""
        with pytest.raises(ValidationError):
            GlobInput(pattern="*.py", max_results=10001)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            GlobInput(pattern="*.py", extra_field="value")


# ============================================================================
# GrepInput Validation Tests
# ============================================================================


class TestGrepInput:
    """Test GrepInput validation."""

    def test_valid_input_minimal(self):
        """Test valid minimal input."""
        input_data = GrepInput(pattern="TODO")
        assert input_data.pattern == "TODO"
        assert input_data.output_mode == "files_with_matches"
        assert input_data.ignore_case is False

    def test_valid_input_full(self):
        """Test valid input with all parameters."""
        input_data = GrepInput(
            pattern="function\\s+\\w+",
            path="./src",
            glob="*.js",
            output_mode="content",
            context=3,
            line_number=True,
            ignore_case=True,
            file_type="js",
            head_limit=100,
            multiline=False,
            timeout=60,
        )
        assert input_data.pattern == "function\\s+\\w+"
        assert input_data.output_mode == "content"
        assert input_data.context == 3
        assert input_data.line_number is True

    def test_invalid_empty_pattern(self):
        """Test that empty pattern is rejected."""
        with pytest.raises(ValidationError):
            GrepInput(pattern="")

    def test_invalid_context_with_wrong_output_mode(self):
        """Test that context options are rejected with non-content output mode."""
        with pytest.raises(ValidationError) as exc_info:
            GrepInput(pattern="TODO", output_mode="files_with_matches", context=3)

        error_msg = str(exc_info.value).lower()
        assert "context" in error_msg
        assert "content" in error_msg

    def test_invalid_line_number_with_wrong_output_mode(self):
        """Test that line_number is rejected with non-content output mode."""
        with pytest.raises(ValidationError) as exc_info:
            GrepInput(pattern="TODO", output_mode="count", line_number=True)

        assert "context" in str(exc_info.value).lower()

    def test_invalid_context_before_with_wrong_output_mode(self):
        """Test that context_before is rejected with non-content output mode."""
        with pytest.raises(ValidationError):
            GrepInput(pattern="TODO", output_mode="files_with_matches", context_before=2)

    def test_invalid_context_after_with_wrong_output_mode(self):
        """Test that context_after is rejected with non-content output mode."""
        with pytest.raises(ValidationError):
            GrepInput(pattern="TODO", output_mode="count", context_after=2)

    def test_valid_context_with_content_mode(self):
        """Test that context options are accepted with content output mode."""
        input_data = GrepInput(pattern="TODO", output_mode="content", context=3, line_number=True)
        assert input_data.context == 3
        assert input_data.line_number is True

    def test_invalid_context_too_large(self):
        """Test that overly large context is rejected."""
        with pytest.raises(ValidationError):
            GrepInput(pattern="TODO", output_mode="content", context=101)

    def test_invalid_timeout_too_large(self):
        """Test that overly large timeout is rejected."""
        with pytest.raises(ValidationError):
            GrepInput(pattern="TODO", timeout=301)


# ============================================================================
# LSInput Validation Tests
# ============================================================================


class TestLSInput:
    """Test LSInput validation."""

    def test_valid_input_minimal(self, tmp_path):
        """Test valid minimal input."""
        input_data = LSInput(path=str(tmp_path))
        assert Path(input_data.path).is_absolute()
        assert input_data.ignore is None

    def test_valid_input_with_ignore(self, tmp_path):
        """Test valid input with ignore patterns."""
        input_data = LSInput(path=str(tmp_path), ignore=["*.pyc", "__pycache__"])
        assert input_data.ignore == ["*.pyc", "__pycache__"]

    def test_invalid_relative_path(self):
        """Test that relative path is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LSInput(path="./relative/path")

        assert "absolute" in str(exc_info.value).lower()

    def test_invalid_path_not_exists(self):
        """Test that non-existent path is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LSInput(path="/nonexistent/absolute/path/12345")

        assert "does not exist" in str(exc_info.value).lower()

    def test_invalid_path_not_directory(self, tmp_path):
        """Test that file path (not directory) is rejected."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValidationError) as exc_info:
            LSInput(path=str(test_file.absolute()))

        assert "not a directory" in str(exc_info.value).lower()


# ============================================================================
# glob_files() Function Tests
# ============================================================================


class TestGlobFiles:
    """Test glob_files() function."""

    def test_glob_finds_files(self, tmp_path):
        """Test that glob finds matching files."""
        # Create test files
        (tmp_path / "test1.py").write_text("print('test1')")
        (tmp_path / "test2.py").write_text("print('test2')")
        (tmp_path / "test.txt").write_text("not python")

        # Search for Python files
        result = glob_files(GlobInput(pattern="*.py", path=str(tmp_path)))

        assert isinstance(result, GlobResult)
        assert len(result.files) == 2
        assert result.total_count == 2
        assert result.truncated is False
        assert all(f.endswith(".py") for f in result.files)

    def test_glob_recursive_pattern(self, tmp_path):
        """Test recursive glob pattern."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.js").write_text("// root")
        (subdir / "nested.js").write_text("// nested")

        # Search recursively
        result = glob_files(GlobInput(pattern="**/*.js", path=str(tmp_path)))

        assert len(result.files) == 2
        assert result.total_count == 2

    def test_glob_no_matches(self, tmp_path):
        """Test glob with no matching files."""
        (tmp_path / "test.txt").write_text("test")

        result = glob_files(GlobInput(pattern="*.py", path=str(tmp_path)))

        assert len(result.files) == 0
        assert result.total_count == 0
        assert result.truncated is False

    def test_glob_max_results_truncation(self, tmp_path):
        """Test that max_results truncates results."""
        # Create many files
        for i in range(10):
            (tmp_path / f"test{i}.py").write_text(f"# test {i}")

        result = glob_files(GlobInput(pattern="*.py", path=str(tmp_path), max_results=5))

        assert len(result.files) == 5
        assert result.total_count == 10
        assert result.truncated is True

    def test_glob_sorted_by_mtime(self, tmp_path):
        """Test that results are sorted by modification time."""
        import time

        # Create files with different mtimes
        file1 = tmp_path / "old.py"
        file1.write_text("old")
        time.sleep(0.01)

        file2 = tmp_path / "new.py"
        file2.write_text("new")

        result = glob_files(GlobInput(pattern="*.py", path=str(tmp_path)))

        # Newest file should be first
        assert "new.py" in result.files[0]

    def test_glob_default_path_is_cwd(self):
        """Test that default path is current working directory."""
        result = glob_files(GlobInput(pattern="*.py"))

        # Should not raise error and should have search_path set
        assert result.search_path is not None

    def test_glob_invalid_pattern_raises_error(self, tmp_path):
        """Test that invalid glob pattern raises GlobError."""
        # Note: Most patterns are valid, but we can test error handling
        # by using a pattern that causes issues
        with patch("pathlib.Path.glob", side_effect=ValueError("Invalid pattern")):
            with pytest.raises(GlobError) as exc_info:
                glob_files(GlobInput(pattern="*.py", path=str(tmp_path)))

            assert "invalid glob pattern" in str(exc_info.value).lower()


# ============================================================================
# grep_search() Function Tests
# ============================================================================


class TestGrepSearch:
    """Test grep_search() function."""

    def test_ripgrep_not_available_raises_error(self):
        """Test that missing ripgrep raises RipgrepNotFoundError."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RipgrepNotFoundError) as exc_info:
                grep_search(GrepInput(pattern="TODO"))

            assert "ripgrep" in str(exc_info.value).lower()
            assert "install" in str(exc_info.value).lower()

    @patch("shutil.which", return_value="/usr/bin/rg")
    @patch("subprocess.run")
    def test_grep_files_with_matches_mode(self, mock_run, mock_which):
        """Test grep with files_with_matches output mode."""
        # Mock ripgrep output
        mock_run.return_value = Mock(returncode=0, stdout="file1.py\nfile2.py\nfile3.py\n", stderr="")

        result = grep_search(GrepInput(pattern="TODO", output_mode="files_with_matches"))

        assert isinstance(result, GrepResult)
        assert result.output_mode == "files_with_matches"
        assert result.files == ["file1.py", "file2.py", "file3.py"]
        assert result.total_matches == 3
        assert result.matches is None
        assert result.counts is None

    @patch("shutil.which", return_value="/usr/bin/rg")
    @patch("subprocess.run")
    def test_grep_count_mode(self, mock_run, mock_which):
        """Test grep with count output mode."""
        mock_run.return_value = Mock(returncode=0, stdout="file1.py:5\nfile2.py:3\n", stderr="")

        result = grep_search(GrepInput(pattern="TODO", output_mode="count"))

        assert result.output_mode == "count"
        assert result.counts == {"file1.py": 5, "file2.py": 3}
        assert result.total_matches == 8
        assert result.files is None
        assert result.matches is None

    @patch("shutil.which", return_value="/usr/bin/rg")
    @patch("subprocess.run")
    def test_grep_content_mode_with_line_numbers(self, mock_run, mock_which):
        """Test grep with content output mode and line numbers."""
        mock_run.return_value = Mock(
            returncode=0, stdout="file1.py:10:# TODO: fix this\nfile2.py:25:# TODO: refactor\n", stderr=""
        )

        result = grep_search(GrepInput(pattern="TODO", output_mode="content", line_number=True))

        assert result.output_mode == "content"
        assert len(result.matches) == 2
        assert result.matches[0].file_path == "file1.py"
        assert result.matches[0].line_number == 10
        assert result.matches[0].line_content == "# TODO: fix this"
        assert result.total_matches == 2

    @patch("shutil.which", return_value="/usr/bin/rg")
    @patch("subprocess.run")
    def test_grep_content_mode_without_line_numbers(self, mock_run, mock_which):
        """Test grep with content output mode without line numbers."""
        mock_run.return_value = Mock(
            returncode=0, stdout="file1.py:# TODO: fix this\nfile2.py:# TODO: refactor\n", stderr=""
        )

        result = grep_search(GrepInput(pattern="TODO", output_mode="content", line_number=False))

        assert len(result.matches) == 2
        assert result.matches[0].line_number is None

    @patch("shutil.which", return_value="/usr/bin/rg")
    @patch("subprocess.run")
    def test_grep_no_matches(self, mock_run, mock_which):
        """Test grep with no matches found."""
        mock_run.return_value = Mock(
            returncode=1,  # Exit code 1 means no matches
            stdout="",
            stderr="",
        )

        result = grep_search(GrepInput(pattern="NONEXISTENT", output_mode="files_with_matches"))

        assert result.total_matches == 0
        assert result.files is None or result.files == []

    @patch("shutil.which", return_value="/usr/bin/rg")
    @patch("subprocess.run")
    def test_grep_error_exit_code(self, mock_run, mock_which):
        """Test grep with error exit code."""
        mock_run.return_value = Mock(
            returncode=2,  # Exit code 2 means error
            stdout="",
            stderr="Error: invalid regex",
        )

        with pytest.raises(GrepError) as exc_info:
            grep_search(GrepInput(pattern="[invalid"))

        assert "error" in str(exc_info.value).lower()

    @patch("shutil.which", return_value="/usr/bin/rg")
    @patch("subprocess.run")
    def test_grep_timeout(self, mock_run, mock_which):
        """Test grep timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["rg"], timeout=30)

        with pytest.raises(GrepError) as exc_info:
            grep_search(GrepInput(pattern="TODO", timeout=30))

        assert "timeout" in str(exc_info.value).lower()

    @patch("shutil.which", return_value="/usr/bin/rg")
    @patch("subprocess.run")
    def test_grep_builds_correct_command(self, mock_run, mock_which):
        """Test that grep builds correct ripgrep command."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="")

        grep_search(
            GrepInput(
                pattern="TODO",
                path="./src",
                glob="*.py",
                output_mode="content",
                line_number=True,
                context=3,
                ignore_case=True,
                file_type="py",
                multiline=True,
            )
        )

        # Verify command was called
        assert mock_run.called
        cmd = mock_run.call_args[0][0]

        assert "rg" in cmd
        assert "--line-number" in cmd
        assert "--context" in cmd
        assert "--ignore-case" in cmd
        assert "--type" in cmd
        assert "--glob" in cmd
        assert "--multiline" in cmd
        assert "TODO" in cmd


# ============================================================================
# ls_directory() Function Tests
# ============================================================================


class TestLSDirectory:
    """Test ls_directory() function."""

    def test_ls_lists_files_and_directories(self, tmp_path):
        """Test that ls lists both files and directories."""
        # Create test structure
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.py").write_text("content2")
        (tmp_path / "subdir").mkdir()

        result = ls_directory(LSInput(path=str(tmp_path)))

        assert isinstance(result, LSResult)
        assert len(result.entries) == 3
        assert result.total_count == 3
        assert result.truncated is False

        # Check that we have both files and directories
        names = [e.name for e in result.entries]
        assert "file1.txt" in names
        assert "file2.py" in names
        assert "subdir" in names

        # Check directory flag
        subdir_entry = next(e for e in result.entries if e.name == "subdir")
        assert subdir_entry.is_directory is True

        file_entry = next(e for e in result.entries if e.name == "file1.txt")
        assert file_entry.is_directory is False
        assert file_entry.size is not None

    def test_ls_with_ignore_patterns(self, tmp_path):
        """Test ls with ignore patterns."""
        # Create test files
        (tmp_path / "keep.py").write_text("keep")
        (tmp_path / "ignore.pyc").write_text("ignore")
        (tmp_path / "__pycache__").mkdir()

        result = ls_directory(LSInput(path=str(tmp_path), ignore=["*.pyc", "__pycache__"]))

        names = [e.name for e in result.entries]
        assert "keep.py" in names
        assert "ignore.pyc" not in names
        assert "__pycache__" not in names

    def test_ls_empty_directory(self, tmp_path):
        """Test ls on empty directory."""
        result = ls_directory(LSInput(path=str(tmp_path)))

        assert len(result.entries) == 0
        assert result.total_count == 0
        assert result.truncated is False

    def test_ls_max_results_truncation(self, tmp_path):
        """Test that max_results truncates results."""
        # Create many files
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text(f"content {i}")

        result = ls_directory(LSInput(path=str(tmp_path), max_results=5))

        assert len(result.entries) == 5
        assert result.total_count == 10
        assert result.truncated is True

    def test_ls_entries_sorted(self, tmp_path):
        """Test that entries are sorted (directories first, then by name)."""
        # Create mixed structure
        (tmp_path / "zebra.txt").write_text("z")
        (tmp_path / "apple.txt").write_text("a")
        (tmp_path / "zoo_dir").mkdir()
        (tmp_path / "aaa_dir").mkdir()

        result = ls_directory(LSInput(path=str(tmp_path)))

        names = [e.name for e in result.entries]

        # Directories should come first
        assert result.entries[0].is_directory is True
        assert result.entries[1].is_directory is True

        # Then files
        assert result.entries[2].is_directory is False
        assert result.entries[3].is_directory is False

        # Within each group, sorted by name
        assert names[0] == "aaa_dir"
        assert names[1] == "zoo_dir"
        assert names[2] == "apple.txt"
        assert names[3] == "zebra.txt"

    def test_ls_entry_metadata(self, tmp_path):
        """Test that entry metadata is populated correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = ls_directory(LSInput(path=str(tmp_path)))

        entry = result.entries[0]
        assert entry.name == "test.txt"
        assert entry.path == str(test_file.absolute())
        assert entry.is_directory is False
        assert entry.size == 11  # "hello world" is 11 bytes
        assert entry.modified_time > 0  # Unix timestamp

    def test_ls_permission_error_skips_entry(self, tmp_path):
        """Test that permission errors are handled gracefully."""
        # This test is platform-dependent and may not work on all systems
        # We'll use mocking to simulate the behavior
        with patch("pathlib.Path.iterdir") as mock_iterdir:
            # Create a mock that raises PermissionError for one item
            mock_path = Mock()
            mock_path.stat.side_effect = PermissionError("Access denied")
            mock_iterdir.return_value = [mock_path]

            result = ls_directory(LSInput(path=str(tmp_path)))

            # Entry should be skipped
            assert len(result.entries) == 0
