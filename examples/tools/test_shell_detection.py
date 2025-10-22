"""Test shell detection on this system

Run with: python examples/tools/test_shell_detection.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.bash_tool import _detect_shell

try:
    shell = _detect_shell()
    print(f"Detected shell: {shell}")
except Exception as e:
    print(f"Error detecting shell: {e}")
