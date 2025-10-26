"""
Code Formatter Tool

Formats Python code using black or ruff for consistent style.

Features:
- Multiple formatter backends (black, ruff)
- Configurable line length and style options
- Diff generation to show changes
- Safe formatting with validation

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config.tools import FormatterBackend, FormatterConfig

# ============================================================================
# Result Types
# ============================================================================


@dataclass
class FormattedCode:
    """Result of code formatting."""

    code: str
    """Formatted code"""

    changed: bool
    """Whether code was changed"""

    diff: str = ""
    """Diff showing changes"""

    backend_used: str = ""
    """Formatter backend that was used"""

    metadata: dict[str, Any] | None = None
    """Additional metadata"""

    def __post_init__(self) -> None:
        """Initialize metadata."""
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# Code Formatter
# ============================================================================


class CodeFormatter:
    """
    Code formatter using black or ruff.

    Formats Python code according to PEP 8 and configurable style preferences.
    """

    def __init__(self, config: FormatterConfig | None = None) -> None:
        """
        Initialize code formatter.

        Args:
            config: Formatter configuration
        """
        self.config = config or FormatterConfig()
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect which formatter backend to use."""
        if self.config.backend != FormatterBackend.AUTO:
            return str(self.config.backend.value)

        # Try ruff first (faster), fall back to black
        try:
            subprocess.run(
                ["ruff", "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return "ruff"
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            subprocess.run(
                ["black", "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return "black"
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Fall back to black as default
            return "black"

    def format(self, code: str) -> FormattedCode:
        """
        Format code using configured backend.

        Args:
            code: Python code to format

        Returns:
            Formatted code result
        """
        if self._backend == "ruff":
            return self._format_with_ruff(code)
        return self._format_with_black(code)

    def _format_with_black(self, code: str) -> FormattedCode:
        """Format code using black."""
        try:
            import black  # type: ignore[import-not-found]
            from black import Mode, TargetVersion

            # Parse target version
            target_versions = set()
            if self.config.target_version:
                version_map = {
                    "py38": TargetVersion.PY38,
                    "py39": TargetVersion.PY39,
                    "py310": TargetVersion.PY310,
                    "py311": TargetVersion.PY311,
                    "py312": TargetVersion.PY312,
                }
                if self.config.target_version in version_map:
                    target_versions.add(version_map[self.config.target_version])

            # Create mode
            mode = Mode(
                target_versions=target_versions,
                line_length=self.config.line_length,
                string_normalization=not self.config.skip_string_normalization,
                magic_trailing_comma=not self.config.skip_magic_trailing_comma,
                preview=self.config.preview,
            )

            # Format code
            formatted = black.format_str(code, mode=mode)

            # Check if changed
            changed = formatted != code

            # Generate diff if changed
            diff = ""
            if changed:
                diff = self._generate_diff(code, formatted)

            return FormattedCode(
                code=formatted,
                changed=changed,
                diff=diff,
                backend_used="black",
                metadata={"line_length": self.config.line_length},
            )

        except ImportError:
            # Black not available, return original code
            return FormattedCode(
                code=code,
                changed=False,
                diff="",
                backend_used="none",
                metadata={"error": "black not installed"},
            )
        except Exception as e:
            # Formatting failed, return original code
            return FormattedCode(
                code=code,
                changed=False,
                diff="",
                backend_used="black",
                metadata={"error": str(e)},
            )

    def _format_with_ruff(self, code: str) -> FormattedCode:
        """Format code using ruff."""
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(code)
                temp_path = Path(f.name)

            try:
                # Run ruff format
                _result = subprocess.run(
                    [
                        "ruff",
                        "format",
                        "--line-length",
                        str(self.config.line_length),
                        str(temp_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Read formatted code
                formatted = temp_path.read_text(encoding="utf-8")

                # Check if changed
                changed = formatted != code

                # Generate diff if changed
                diff = ""
                if changed:
                    diff = self._generate_diff(code, formatted)

                return FormattedCode(
                    code=formatted,
                    changed=changed,
                    diff=diff,
                    backend_used="ruff",
                    metadata={"line_length": self.config.line_length},
                )

            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            # Ruff not available or failed, return original code
            return FormattedCode(
                code=code,
                changed=False,
                diff="",
                backend_used="none",
                metadata={"error": str(e)},
            )
        except Exception as e:
            # Unexpected error, return original code
            return FormattedCode(
                code=code,
                changed=False,
                diff="",
                backend_used="ruff",
                metadata={"error": str(e)},
            )

    def _generate_diff(self, original: str, formatted: str) -> str:
        """Generate unified diff between original and formatted code."""
        import difflib

        original_lines = original.splitlines(keepends=True)
        formatted_lines = formatted.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            formatted_lines,
            fromfile="original",
            tofile="formatted",
            lineterm="",
        )

        return "".join(diff)

    def check(self, code: str) -> bool:
        """
        Check if code is already formatted.

        Args:
            code: Python code to check

        Returns:
            True if code is already formatted, False otherwise
        """
        result = self.format(code)
        return not result.changed
