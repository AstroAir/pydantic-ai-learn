#!/usr/bin/env python3
"""
Run all code quality checks for the code_agent package.

Run with: python scripts/run_all_checks.py
"""

import os
import subprocess
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(name: str, command: list[str]) -> bool:
    """Run a command and report results."""
    print(f"\n{'=' * 70}")
    print(f"Running: {name}")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {name} PASSED")
            return True
        print(f"❌ {name} FAILED (exit code: {result.returncode})")
        return False

    except Exception as e:
        print(f"❌ {name} ERROR: {e}")
        return False


def main():
    """Run all quality checks."""
    print("=" * 70)
    print("CODE QUALITY ASSURANCE - COMPREHENSIVE CHECK")
    print("=" * 70)

    checks = [
        ("MyPy Strict Type Checking", ["mypy", "code_agent", "--strict", "--show-error-codes"]),
        ("Ruff Code Quality Linter", ["ruff", "check", "code_agent"]),
        ("Circular Dependency Check", [sys.executable, "scripts/check_circular_deps.py"]),
        ("Comprehensive Tests", [sys.executable, "code_agent/test_comprehensive.py"]),
    ]

    results = []
    for name, command in checks:
        passed = run_command(name, command)
        results.append((name, passed))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    print("\n" + "=" * 70)
    print(f"Results: {passed_count}/{total_count} checks passed")
    print("=" * 70)

    if passed_count == total_count:
        print("\n🎉 ALL CHECKS PASSED! Code is production-ready.")
        return 0
    print(f"\n⚠️  {total_count - passed_count} check(s) failed. Please review.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
