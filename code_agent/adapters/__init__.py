"""
Adapters Module

Integration adapters for external systems and services.

Exports:
    - ContextManager: Context management system
    - WorkflowOrchestrator: Workflow orchestration
    - GraphConfig: Graph configuration
"""

from __future__ import annotations

from .context import (
    ContextConfig,
    ContextManager,
    ImportanceLevel,
    PruningStrategy,
    create_context_manager,
)
from .graph import (
    GraphConfig,
    GraphPersistenceAdapter,
    GraphState,
)
from .workflow import (
    FixStrategy,
    WorkflowOrchestrator,
    WorkflowState,
)

__all__ = [
    # Context
    "ContextManager",
    "ContextConfig",
    "PruningStrategy",
    "ImportanceLevel",
    "create_context_manager",
    # Workflow
    "WorkflowOrchestrator",
    "WorkflowState",
    "FixStrategy",
    # Graph
    "GraphConfig",
    "GraphState",
    "GraphPersistenceAdapter",
]
