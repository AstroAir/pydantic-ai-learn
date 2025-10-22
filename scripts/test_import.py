#!/usr/bin/env python3
"""Quick import test for task_planning_toolkit.

Run with: python scripts/test_import.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tools.task_planning_toolkit import (
        TaskListState,
        TodoItem,
        add_task,
        get_task_summary,
        mark_task_complete,
        mark_task_in_progress,
        task_launcher,
        todo_write,
    )

    print("✓ All imports successful!")

    # Quick functionality test
    state = TaskListState()
    print(f"✓ TaskListState created: {len(state.get_tasks())} tasks")

    # Test add_task
    result = add_task("Test task", state)
    print(f"✓ add_task: {result}")

    # Test get_task_summary
    summary = get_task_summary(state)
    print(f"✓ get_task_summary:\n{summary}")

    print("\n✓✓✓ All tests passed! ✓✓✓")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
