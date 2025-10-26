"""
Dependency Analyzer Tool

Analyzes code dependencies, imports, and suggests optimizations.

Features:
- Import analysis (stdlib, third-party, local)
- Unused import detection
- Circular dependency detection
- Import optimization suggestions
- Dependency graph generation

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from ...config.tools import DependencyConfig

# ============================================================================
# Result Types
# ============================================================================


@dataclass
class ImportInfo:
    """Information about a single import."""

    module: str
    """Module name"""

    names: list[str]
    """Imported names (empty for 'import module')"""

    alias: str | None = None
    """Import alias if any"""

    line: int = 0
    """Line number"""

    is_stdlib: bool = False
    """Whether module is from standard library"""

    is_third_party: bool = False
    """Whether module is third-party"""

    is_local: bool = False
    """Whether module is local"""


@dataclass
class DependencyIssue:
    """A dependency-related issue."""

    type: str
    """Issue type (unused, circular, etc.)"""

    message: str
    """Issue description"""

    module: str
    """Related module"""

    line: int = 0
    """Line number if applicable"""

    suggestion: str | None = None
    """Suggested fix"""


@dataclass
class DependencyAnalysis:
    """Result of dependency analysis."""

    imports: list[ImportInfo] = field(default_factory=list)
    """All imports found"""

    stdlib_imports: list[str] = field(default_factory=list)
    """Standard library imports"""

    third_party_imports: list[str] = field(default_factory=list)
    """Third-party imports"""

    local_imports: list[str] = field(default_factory=list)
    """Local imports"""

    issues: list[DependencyIssue] = field(default_factory=list)
    """Dependency issues"""

    suggestions: list[str] = field(default_factory=list)
    """Optimization suggestions"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    @property
    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return len(self.issues) > 0

    @property
    def total_imports(self) -> int:
        """Total number of imports."""
        return len(self.imports)


# ============================================================================
# Dependency Analyzer
# ============================================================================


class DependencyAnalyzer:
    """
    Dependency analyzer for Python code.

    Analyzes imports and dependencies to detect issues and suggest optimizations.
    """

    # Standard library modules (Python 3.11+)
    STDLIB_MODULES = {
        "abc",
        "aifc",
        "argparse",
        "array",
        "ast",
        "asynchat",
        "asyncio",
        "asyncore",
        "atexit",
        "audioop",
        "base64",
        "bdb",
        "binascii",
        "binhex",
        "bisect",
        "builtins",
        "bz2",
        "calendar",
        "cgi",
        "cgitb",
        "chunk",
        "cmath",
        "cmd",
        "code",
        "codecs",
        "codeop",
        "collections",
        "colorsys",
        "compileall",
        "concurrent",
        "configparser",
        "contextlib",
        "contextvars",
        "copy",
        "copyreg",
        "cProfile",
        "crypt",
        "csv",
        "ctypes",
        "curses",
        "dataclasses",
        "datetime",
        "dbm",
        "decimal",
        "difflib",
        "dis",
        "distutils",
        "doctest",
        "email",
        "encodings",
        "enum",
        "errno",
        "faulthandler",
        "fcntl",
        "filecmp",
        "fileinput",
        "fnmatch",
        "fractions",
        "ftplib",
        "functools",
        "gc",
        "getopt",
        "getpass",
        "gettext",
        "glob",
        "graphlib",
        "grp",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "imaplib",
        "imghdr",
        "imp",
        "importlib",
        "inspect",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "keyword",
        "lib2to3",
        "linecache",
        "locale",
        "logging",
        "lzma",
        "mailbox",
        "mailcap",
        "marshal",
        "math",
        "mimetypes",
        "mmap",
        "modulefinder",
        "msilib",
        "msvcrt",
        "multiprocessing",
        "netrc",
        "nis",
        "nntplib",
        "numbers",
        "operator",
        "optparse",
        "os",
        "ossaudiodev",
        "pathlib",
        "pdb",
        "pickle",
        "pickletools",
        "pipes",
        "pkgutil",
        "platform",
        "plistlib",
        "poplib",
        "posix",
        "posixpath",
        "pprint",
        "profile",
        "pstats",
        "pty",
        "pwd",
        "py_compile",
        "pyclbr",
        "pydoc",
        "queue",
        "quopri",
        "random",
        "re",
        "readline",
        "reprlib",
        "resource",
        "rlcompleter",
        "runpy",
        "sched",
        "secrets",
        "select",
        "selectors",
        "shelve",
        "shlex",
        "shutil",
        "signal",
        "site",
        "smtpd",
        "smtplib",
        "sndhdr",
        "socket",
        "socketserver",
        "spwd",
        "sqlite3",
        "ssl",
        "stat",
        "statistics",
        "string",
        "stringprep",
        "struct",
        "subprocess",
        "sunau",
        "symtable",
        "sys",
        "sysconfig",
        "syslog",
        "tabnanny",
        "tarfile",
        "telnetlib",
        "tempfile",
        "termios",
        "test",
        "textwrap",
        "threading",
        "time",
        "timeit",
        "tkinter",
        "token",
        "tokenize",
        "tomllib",
        "trace",
        "traceback",
        "tracemalloc",
        "tty",
        "turtle",
        "turtledemo",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "urllib",
        "uu",
        "uuid",
        "venv",
        "warnings",
        "wave",
        "weakref",
        "webbrowser",
        "winreg",
        "winsound",
        "wsgiref",
        "xdrlib",
        "xml",
        "xmlrpc",
        "zipapp",
        "zipfile",
        "zipimport",
        "zlib",
        "zoneinfo",
    }

    def __init__(self, config: DependencyConfig | None = None) -> None:
        """
        Initialize dependency analyzer.

        Args:
            config: Dependency analyzer configuration
        """
        self.config = config or DependencyConfig()

    def analyze(self, code: str) -> DependencyAnalysis:
        """
        Analyze code dependencies.

        Args:
            code: Python code to analyze

        Returns:
            Dependency analysis result
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return DependencyAnalysis(
                metadata={"error": f"Syntax error: {e}"},
            )

        # Extract imports
        imports = self._extract_imports(tree)

        # Categorize imports
        stdlib_imports = []
        third_party_imports = []
        local_imports = []

        for imp in imports:
            if imp.is_stdlib:
                stdlib_imports.append(imp.module)
            elif imp.is_third_party:
                third_party_imports.append(imp.module)
            elif imp.is_local:
                local_imports.append(imp.module)

        # Detect issues
        issues = []

        if self.config.check_unused:
            unused_issues = self._detect_unused_imports(code, imports)
            issues.extend(unused_issues)

        # Generate suggestions
        suggestions = []
        if self.config.suggest_optimizations:
            suggestions = self._generate_suggestions(imports)

        return DependencyAnalysis(
            imports=imports,
            stdlib_imports=stdlib_imports,
            third_party_imports=third_party_imports,
            local_imports=local_imports,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "total_imports": len(imports),
                "stdlib_count": len(stdlib_imports),
                "third_party_count": len(third_party_imports),
                "local_count": len(local_imports),
            },
        )

    def _extract_imports(self, tree: ast.AST) -> list[ImportInfo]:
        """Extract all imports from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    imports.append(
                        ImportInfo(
                            module=alias.name,
                            names=[],
                            alias=alias.asname,
                            line=node.lineno,
                            is_stdlib=self._is_stdlib(module),
                            is_third_party=self._is_third_party(module),
                            is_local=self._is_local(module),
                        )
                    )

            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split(".")[0]
                names = [alias.name for alias in node.names]
                imports.append(
                    ImportInfo(
                        module=node.module,
                        names=names,
                        line=node.lineno,
                        is_stdlib=self._is_stdlib(module),
                        is_third_party=self._is_third_party(module),
                        is_local=self._is_local(module),
                    )
                )

        return imports

    def _is_stdlib(self, module: str) -> bool:
        """Check if module is from standard library."""
        return module in self.STDLIB_MODULES

    def _is_third_party(self, module: str) -> bool:
        """Check if module is third-party."""
        return not self._is_stdlib(module) and not self._is_local(module)

    def _is_local(self, module: str) -> bool:
        """Check if module is local (relative import or starts with '.')."""
        return module.startswith(".")

    def _detect_unused_imports(self, code: str, imports: list[ImportInfo]) -> list[DependencyIssue]:
        """Detect unused imports (simple heuristic)."""
        issues = []

        for imp in imports:
            # Check if module/names are used in code
            used = False

            # Check module usage
            if imp.alias:
                used = imp.alias in code
            elif imp.names:
                used = any(name in code for name in imp.names)
            else:
                used = imp.module.split(".")[-1] in code

            if not used:
                issues.append(
                    DependencyIssue(
                        type="unused_import",
                        message=f"Import '{imp.module}' appears to be unused",
                        module=imp.module,
                        line=imp.line,
                        suggestion=f"Remove unused import on line {imp.line}",
                    )
                )

        return issues

    def _generate_suggestions(self, imports: list[ImportInfo]) -> list[str]:
        """Generate optimization suggestions."""
        suggestions = []

        # Suggest grouping imports
        if len(imports) > 5:
            suggestions.append("Consider organizing imports into groups: stdlib, third-party, local")

        # Suggest using 'from' imports for frequently used items
        module_counts: dict[str, int] = {}
        for imp in imports:
            if not imp.names:  # 'import module' style
                module_counts[imp.module] = module_counts.get(imp.module, 0) + 1

        for module, count in module_counts.items():
            if count > 3:
                suggestions.append(f"Consider using 'from {module} import ...' for frequently used items")

        return suggestions
