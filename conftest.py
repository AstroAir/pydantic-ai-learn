"""
Root conftest.py for pytest configuration.

This file ensures that the project root is in sys.path so that
imports work correctly during testing without requiring package installation.

Author: The Augster
Python Version: 3.12+
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
