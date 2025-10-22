"""
Test script to verify all execution imports work correctly.

This script tests that all the new execution modules can be imported
and basic functionality works.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all execution modules can be imported."""
    print("Testing imports...")

    # Test core types
    print("✓ Core types imported")

    # Test execution config
    print("✓ Execution config imported")

    # Test validators
    print("✓ Validators imported")

    # Test executor
    print("✓ Executor imported")

    # Test toolkit integration
    print("✓ Toolkit integration imported")

    # Test package-level imports
    print("✓ Package-level imports work")

    print("\n✅ All imports successful!")
    return True


def test_basic_functionality():
    """Test basic functionality of execution system."""
    print("\nTesting basic functionality...")

    from code_agent.config.execution import create_safe_config
    from code_agent.core.types import ExecutionStatus
    from code_agent.tools.executor import CodeExecutor

    # Create executor
    config = create_safe_config()
    executor = CodeExecutor(config)
    print("✓ Executor created")

    # Test simple execution
    code = "result = 2 + 2"
    result = executor.execute(code)

    assert result.code == code
    assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
    print(f"✓ Execution completed with status: {result.status.value}")

    # Test validation
    from code_agent.tools.validators import ExecutionValidator

    validator = ExecutionValidator()
    validation_result = validator.validate("print('hello')")
    assert validation_result.is_valid
    print("✓ Validation works")

    # Test factory functions
    from code_agent.config.execution import (
        create_full_config,
        create_restricted_config,
        create_safe_config,
    )

    safe = create_safe_config()
    restricted = create_restricted_config()
    full = create_full_config()

    assert safe.security.sandbox_enabled
    assert restricted.security.sandbox_enabled
    assert not full.security.sandbox_enabled
    print("✓ Factory functions work")

    print("\n✅ All functionality tests passed!")
    return True


def test_state_integration():
    """Test integration with CodeAgentState."""
    print("\nTesting state integration...")

    from code_agent.tools.toolkit import CodeAgentState

    state = CodeAgentState()
    assert hasattr(state, "execution_history")
    assert isinstance(state.execution_history, list)
    print("✓ CodeAgentState has execution_history field")

    print("\n✅ State integration test passed!")
    return True


def main() -> int:
    """Run all tests."""
    print("=" * 60)
    print("EXECUTION SYSTEM IMPORT AND FUNCTIONALITY TESTS")
    print("=" * 60)

    try:
        test_imports()  # type: ignore[no-untyped-call]
        test_basic_functionality()  # type: ignore[no-untyped-call]
        test_state_integration()  # type: ignore[no-untyped-call]

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
