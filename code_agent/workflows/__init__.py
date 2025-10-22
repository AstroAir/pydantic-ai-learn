"""
Workflows Module

Multi-step workflow orchestration combining multiple tools.

This module provides pre-built workflows for common tasks:
- QualityWorkflow: Comprehensive code quality pipeline
- DocumentationWorkflow: Documentation generation and validation

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

from .quality import QualityReport, QualityWorkflow, WorkflowStep

__all__ = [
    "QualityWorkflow",
    "QualityReport",
    "WorkflowStep",
]
