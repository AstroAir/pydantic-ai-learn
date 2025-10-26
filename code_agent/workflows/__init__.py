"""
Workflows Module

Multi-step workflow orchestration combining multiple tools.

This module provides:
- Base workflow infrastructure (BaseWorkflow, WorkflowContext, WorkflowRegistry, etc.)
- Pre-built workflows for common tasks (QualityWorkflow, etc.)
- Workflow composition patterns (SequentialWorkflow, ParallelWorkflow, etc.)
- Workflow execution engine with cross-referencing and cycle detection

Features:
- Custom workflow support with registration and discovery
- Workflow cross-referencing and invocation
- Workflow composition and interleaving
- Cycle detection and error handling

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

# Base infrastructure
from .base import (
    BaseWorkflow,
    WorkflowContext,
    WorkflowMetadata,
    WorkflowResult,
    WorkflowStatus,
)

# Composition patterns
from .composition import (
    ConditionalWorkflow,
    ParallelWorkflow,
    SequentialWorkflow,
    WorkflowBranch,
)

# Execution engine
from .executor import (
    ExecutionState,
    WorkflowCycleError,
    WorkflowError,
    WorkflowExecutionError,
    WorkflowExecutor,
    WorkflowNotFoundError,
    WorkflowValidationError,
)

# Pre-built workflows
from .quality import QualityReport, QualityWorkflow, WorkflowStep

# Registry
from .registry import WorkflowRegistry

__all__ = [
    # Base infrastructure
    "BaseWorkflow",
    "WorkflowContext",
    "WorkflowMetadata",
    "WorkflowResult",
    "WorkflowStatus",
    # Registry
    "WorkflowRegistry",
    # Executor
    "WorkflowExecutor",
    "ExecutionState",
    "WorkflowError",
    "WorkflowNotFoundError",
    "WorkflowCycleError",
    "WorkflowValidationError",
    "WorkflowExecutionError",
    # Composition
    "SequentialWorkflow",
    "ParallelWorkflow",
    "ConditionalWorkflow",
    "WorkflowBranch",
    # Pre-built workflows
    "QualityWorkflow",
    "QualityReport",
    "WorkflowStep",
]
