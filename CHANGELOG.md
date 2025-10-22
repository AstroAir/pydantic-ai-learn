# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub issue templates (bug report, feature request)
- GitHub pull request template
- GitHub Actions CI/CD workflows (tests, code quality)
- CONTRIBUTING.md with comprehensive contribution guidelines
- CODE_OF_CONDUCT.md following Contributor Covenant v1.4
- Automation scripts for common development tasks:
  - `setup.sh` / `setup.ps1` - Project setup
  - `run_tests.sh` / `run_tests.ps1` - Test execution
  - `run_examples.sh` / `run_examples.ps1` - Example verification
  - `lint.sh` / `lint.ps1` - Code quality checks
  - `format.sh` / `format.ps1` - Code formatting
  - `clean.sh` / `clean.ps1` - Cleanup build artifacts
- README badges for build status and code quality
- Comprehensive .gitignore for Python projects

### Changed
- Project structure reorganized following Python best practices
- All tests moved to `tests/` directory
- All examples moved to `examples/` directory
- Development documentation moved to `docs/` directory
- Updated README with new structure and automation scripts

### Fixed
- Test discovery issues with pytest configuration
- Import path issues in test files

## [0.1.0] - 2025-10-21

### Added
- Initial project structure
- Code Agent package for autonomous code analysis
- Production-ready tools:
  - Bash command execution tool
  - File system operations tool
  - File editing toolkit
  - Task planning toolkit
- Utility functions:
  - Message formatter
  - Terminal stream handler
  - URL checker
- Comprehensive examples demonstrating PydanticAI features
- Test suite with 614 tests
- Development scripts:
  - `check_circular_deps.py` - Circular dependency checker
  - `run_all_checks.py` - Run all quality checks
- Documentation:
  - README.md with project overview
  - WARP.md with development notes
  - LICENSE file

### Dependencies
- pydantic-ai >= 1.0.15
- mcp-run-python >= 0.0.21
- mcp[cli] >= 1.16.0
- tiktoken >= 0.12.0
- trio >= 0.31.0

### Development Dependencies
- mypy >= 1.0
- ruff >= 0.1.0
- pytest >= 7.0
- pytest-cov >= 4.0

---

## Release Types

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

## Version Format

This project uses [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

[Unreleased]: https://github.com/yourusername/pydantic-ai-learn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/pydantic-ai-learn/releases/tag/v0.1.0
