"""
Code Linter Tool

Lints Python code using ruff for comprehensive quality checks.

Features:
- Fast linting with ruff
- Configurable rule selection
- Severity filtering
- Auto-fix support
- Detailed issue reporting

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...config.custom import LinterConfig, LintSeverity

# ============================================================================
# Result Types
# ============================================================================


@dataclass
class LintIssue:
    """A single lint issue."""

    code: str
    """Issue code (e.g., 'E501', 'F401')"""

    message: str
    """Issue message"""

    line: int
    """Line number"""

    column: int
    """Column number"""

    severity: LintSeverity
    """Issue severity"""

    fixable: bool = False
    """Whether issue can be auto-fixed"""

    fix: str | None = None
    """Suggested fix if available"""


@dataclass
class LintResult:
    """Result of code linting."""

    issues: list[LintIssue] = field(default_factory=list)
    """List of lint issues"""

    error_count: int = 0
    """Number of errors"""

    warning_count: int = 0
    """Number of warnings"""

    info_count: int = 0
    """Number of info messages"""

    fixed_code: str | None = None
    """Auto-fixed code if auto_fix was enabled"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return self.error_count > 0

    @property
    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return len(self.issues) > 0

    @property
    def is_clean(self) -> bool:
        """Check if code is clean (no issues)."""
        return not self.has_issues


# ============================================================================
# Code Linter
# ============================================================================


class CodeLinter:
    """
    Code linter using ruff.

    Provides comprehensive code quality checks including:
    - Style violations (PEP 8)
    - Code smells
    - Complexity issues
    - Import organization
    - Naming conventions
    """

    def __init__(self, config: LinterConfig | None = None) -> None:
        """
        Initialize code linter.

        Args:
            config: Linter configuration
        """
        self.config = config or LinterConfig()

    def lint(self, code: str) -> LintResult:
        """
        Lint code and return issues.

        Args:
            code: Python code to lint

        Returns:
            Lint result with issues
        """
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
                # Build ruff command
                cmd = [
                    "ruff",
                    "check",
                    "--output-format=json",
                    "--line-length",
                    str(self.config.line_length),
                ]

                # Add rule selection
                if self.config.select_rules:
                    cmd.extend(["--select", ",".join(self.config.select_rules)])

                # Add rule ignores
                if self.config.ignore_rules:
                    cmd.extend(["--ignore", ",".join(self.config.ignore_rules)])

                # Add auto-fix if enabled
                if self.config.auto_fix:
                    cmd.append("--fix")

                cmd.append(str(temp_path))

                # Run ruff
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Parse JSON output
                issues = []
                if result.stdout:
                    try:
                        ruff_output = json.loads(result.stdout)
                        issues = self._parse_ruff_output(ruff_output)
                    except json.JSONDecodeError:
                        pass

                # Read potentially fixed code
                fixed_code = None
                if self.config.auto_fix:
                    fixed_code = temp_path.read_text(encoding="utf-8")

                # Count issues by severity
                error_count = sum(1 for i in issues if i.severity == LintSeverity.ERROR)
                warning_count = sum(1 for i in issues if i.severity == LintSeverity.WARNING)
                info_count = sum(1 for i in issues if i.severity == LintSeverity.INFO)

                return LintResult(
                    issues=issues,
                    error_count=error_count,
                    warning_count=warning_count,
                    info_count=info_count,
                    fixed_code=fixed_code,
                    metadata={
                        "rules_checked": self.config.select_rules,
                        "auto_fix_enabled": self.config.auto_fix,
                    },
                )

            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            # Ruff not available or failed
            return LintResult(
                issues=[],
                metadata={"error": str(e), "linter": "ruff"},
            )
        except Exception as e:
            # Unexpected error
            return LintResult(
                issues=[],
                metadata={"error": str(e)},
            )

    def _parse_ruff_output(self, ruff_output: list[dict[str, Any]]) -> list[LintIssue]:
        """Parse ruff JSON output into LintIssue objects."""
        issues = []

        for item in ruff_output:
            # Map ruff severity to our severity
            severity = self._map_severity(item.get("code", ""))

            # Skip if below minimum severity
            if not self._should_include_severity(severity):
                continue

            issue = LintIssue(
                code=item.get("code", "UNKNOWN"),
                message=item.get("message", ""),
                line=item.get("location", {}).get("row", 0),
                column=item.get("location", {}).get("column", 0),
                severity=severity,
                fixable=item.get("fix") is not None,
                fix=item.get("fix", {}).get("message") if item.get("fix") else None,
            )
            issues.append(issue)

        return issues

    def _map_severity(self, code: str) -> LintSeverity:
        """Map ruff code to severity level."""
        # E, F codes are errors
        if code.startswith(("E", "F")):
            return LintSeverity.ERROR
        # W codes are warnings
        if code.startswith("W"):
            return LintSeverity.WARNING
        # Everything else is info
        return LintSeverity.INFO

    def _should_include_severity(self, severity: LintSeverity) -> bool:
        """Check if severity should be included based on config."""
        severity_order = {
            LintSeverity.ERROR: 3,
            LintSeverity.WARNING: 2,
            LintSeverity.INFO: 1,
            LintSeverity.HINT: 0,
        }

        return severity_order.get(severity, 0) >= severity_order.get(self.config.min_severity, 0)

    def check(self, code: str) -> bool:
        """
        Check if code passes linting.

        Args:
            code: Python code to check

        Returns:
            True if code is clean, False if there are issues
        """
        result = self.lint(code)
        return result.is_clean
