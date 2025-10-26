#!/usr/bin/env python3
"""
Check for circular dependencies in the code_agent package.

Run with: python scripts/check_circular_deps.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_circular_imports() -> bool:
    """Check for circular imports by attempting to import all modules."""
    print("Checking for circular dependencies in code_agent package...")
    print("=" * 70)

    modules_to_check = [
        "code_agent",
        "code_agent.agent",
        "code_agent.toolkit",
        "code_agent.logging_config",
        "code_agent.error_handling",
        "code_agent.workflow",
        "code_agent.context_management",
        "code_agent.terminal_ui",
    ]

    failed = []
    passed = []

    for module_name in modules_to_check:
        try:
            print(f"Importing {module_name}...", end=" ")
            __import__(module_name)
            print("[OK]")
            passed.append(module_name)
        except ImportError as e:
            print(f"[FAILED]: {e}")
            failed.append((module_name, str(e)))
        except Exception as e:
            print(f"[ERROR]: {e}")
            failed.append((module_name, str(e)))

    print("\n" + "=" * 70)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")

    if failed:
        print("\nFailed imports:")
        for module, error in failed:
            print(f"  - {module}: {error}")
        return False

    print("\n[SUCCESS] No circular dependencies detected!")
    return True


if __name__ == "__main__":
    success = check_circular_imports()
    sys.exit(0 if success else 1)
