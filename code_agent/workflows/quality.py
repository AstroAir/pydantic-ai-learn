"""
Quality Workflow

Multi-step workflow for comprehensive code quality analysis and improvement.

Workflow Steps:
1. Code Analysis - Analyze code structure and metrics
2. Linting - Check for style and quality issues
3. Formatting - Format code consistently
4. Refactoring - Suggest improvements
5. Documentation - Check documentation coverage
6. Dependency Analysis - Analyze imports and dependencies

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config.tools import QualityWorkflowConfig
from ..tools.analyzer import CodeAnalyzer
from ..tools.custom.dependencies import DependencyAnalyzer
from ..tools.custom.documentation import DocumentationAnalyzer
from ..tools.custom.formatter import CodeFormatter
from ..tools.custom.linter import CodeLinter
from ..tools.refactoring import RefactoringEngine
from .base import BaseWorkflow, WorkflowContext, WorkflowMetadata, WorkflowResult, WorkflowStatus

# ============================================================================
# Result Types
# ============================================================================


@dataclass
class WorkflowStep:
    """Result of a single workflow step."""

    name: str
    """Step name"""

    success: bool
    """Whether step succeeded"""

    result: Any
    """Step result"""

    errors: list[str] = field(default_factory=list)
    """Errors encountered"""

    warnings: list[str] = field(default_factory=list)
    """Warnings generated"""

    duration: float = 0.0
    """Step duration in seconds"""


@dataclass
class QualityReport:
    """Comprehensive quality analysis report."""

    success: bool
    """Whether workflow succeeded"""

    steps: list[WorkflowStep] = field(default_factory=list)
    """Workflow steps executed"""

    final_code: str | None = None
    """Final code after all transformations"""

    summary: dict[str, Any] = field(default_factory=dict)
    """Summary statistics"""

    recommendations: list[str] = field(default_factory=list)
    """Improvement recommendations"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    @property
    def total_errors(self) -> int:
        """Total errors across all steps."""
        return sum(len(step.errors) for step in self.steps)

    @property
    def total_warnings(self) -> int:
        """Total warnings across all steps."""
        return sum(len(step.warnings) for step in self.steps)

    @property
    def total_duration(self) -> float:
        """Total workflow duration."""
        return sum(step.duration for step in self.steps)


# ============================================================================
# Quality Workflow
# ============================================================================


class QualityWorkflow(BaseWorkflow):
    """
    Comprehensive code quality workflow.

    Orchestrates multiple tools to provide complete quality analysis and improvement.
    """

    def __init__(self, config: QualityWorkflowConfig | None = None, workflow_id: str | None = None) -> None:
        """
        Initialize quality workflow.

        Args:
            config: Workflow configuration
            workflow_id: Optional custom workflow ID
        """
        super().__init__(workflow_id)
        self.config = config or QualityWorkflowConfig()

        # Initialize tools
        self.analyzer = CodeAnalyzer()
        self.refactoring_engine = RefactoringEngine()
        self.formatter = CodeFormatter(self.config.formatter_config)
        self.linter = CodeLinter(self.config.linter_config)
        self.dependency_analyzer = DependencyAnalyzer(self.config.dependency_config)
        self.doc_analyzer = DocumentationAnalyzer(self.config.documentation_config)

    def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Run complete quality workflow.

        Supports both legacy (code string) and new (WorkflowContext) interfaces.

        Args:
            context: Workflow execution context OR Python code string (legacy)

        Returns:
            WorkflowResult (or QualityReport for legacy string input)
        """
        # Check if called with legacy interface (code string)
        if isinstance(context, str):
            # Legacy interface - call run_legacy
            return self.run_legacy(context)  # type: ignore[return-value]

        # New interface (WorkflowContext)
        return self._run_with_context(context)

    def _run_with_context(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute workflow with context (BaseWorkflow interface).

        Args:
            context: Workflow execution context

        Returns:
            Workflow result
        """
        # Extract code from context
        code = context.get_input("code", "")
        if not code:
            return WorkflowResult(
                workflow_id=self.workflow_id,
                workflow_name="QualityWorkflow",
                status=WorkflowStatus.FAILED,
                success=False,
                errors=["Missing required input: code"],
                context=context,
            )

        # Run the workflow
        report = self.run_legacy(code)

        # Convert to WorkflowResult
        return WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name="QualityWorkflow",
            status=WorkflowStatus.COMPLETED if report.success else WorkflowStatus.FAILED,
            success=report.success,
            outputs={
                "final_code": report.final_code,
                "summary": report.summary,
                "recommendations": report.recommendations,
                "steps": [{"name": s.name, "success": s.success} for s in report.steps],
            },
            errors=[e for step in report.steps for e in step.errors],
            warnings=[w for step in report.steps for w in step.warnings],
            metadata={
                "total_errors": report.total_errors,
                "total_warnings": report.total_warnings,
                "total_duration": report.total_duration,
                **report.metadata,
            },
            context=context,
        )

    def run_legacy(self, code: str) -> QualityReport:
        """
        Run complete quality workflow (legacy interface for backward compatibility).

        Args:
            code: Python code to analyze

        Returns:
            Quality report with all results
        """
        import time

        steps = []
        current_code = code
        recommendations = []

        # Step 1: Code Analysis
        if self.config.enable_analysis:
            start = time.time()
            analysis_result = self.analyzer.analyze(current_code)
            duration = time.time() - start

            errors = [] if analysis_result.get("valid") else [analysis_result.get("error", "Unknown error")]

            steps.append(
                WorkflowStep(
                    name="Code Analysis",
                    success=analysis_result.get("valid", False),
                    result=analysis_result,
                    errors=errors,
                    duration=duration,
                )
            )

            if not analysis_result.get("valid"):
                # Stop workflow if code is invalid
                return QualityReport(
                    success=False,
                    steps=steps,
                    final_code=current_code,
                    summary={"stopped_at": "analysis", "reason": "invalid_syntax"},
                )

        # Step 2: Dependency Analysis
        if self.config.enable_analysis:
            start = time.time()
            dep_result = self.dependency_analyzer.analyze(current_code)
            duration = time.time() - start

            warnings = [issue.message for issue in dep_result.issues]
            recommendations.extend(dep_result.suggestions)

            steps.append(
                WorkflowStep(
                    name="Dependency Analysis",
                    success=True,
                    result=dep_result,
                    warnings=warnings,
                    duration=duration,
                )
            )

        # Step 3: Linting
        if self.config.enable_linting:
            start = time.time()
            lint_result = self.linter.lint(current_code)
            duration = time.time() - start

            errors = [
                f"{issue.code}: {issue.message} (line {issue.line})"
                for issue in lint_result.issues
                if issue.severity.value == "error"
            ]
            warnings = [
                f"{issue.code}: {issue.message} (line {issue.line})"
                for issue in lint_result.issues
                if issue.severity.value == "warning"
            ]

            steps.append(
                WorkflowStep(
                    name="Linting",
                    success=not lint_result.has_errors,
                    result=lint_result,
                    errors=errors,
                    warnings=warnings,
                    duration=duration,
                )
            )

            # Use fixed code if auto-fix is enabled
            if self.config.auto_fix and lint_result.fixed_code:
                current_code = lint_result.fixed_code

            if self.config.fail_on_errors and lint_result.has_errors:
                return QualityReport(
                    success=False,
                    steps=steps,
                    final_code=current_code,
                    summary={"stopped_at": "linting", "reason": "lint_errors"},
                    recommendations=recommendations,
                )

        # Step 4: Formatting
        if self.config.enable_formatting:
            start = time.time()
            format_result = self.formatter.format(current_code)
            duration = time.time() - start

            warnings = []
            if format_result.changed:
                warnings.append("Code was reformatted")

            steps.append(
                WorkflowStep(
                    name="Formatting",
                    success=True,
                    result=format_result,
                    warnings=warnings,
                    duration=duration,
                )
            )

            # Use formatted code
            if self.config.auto_fix:
                current_code = format_result.code

        # Step 5: Refactoring Suggestions
        if self.config.enable_refactoring:
            start = time.time()
            refactor_result = self.refactoring_engine.suggest_refactoring(current_code)
            duration = time.time() - start

            warnings = [f"{s.title}: {s.description}" for s in refactor_result]
            recommendations.extend([s.suggested_fix for s in refactor_result if s.suggested_fix])

            steps.append(
                WorkflowStep(
                    name="Refactoring Analysis",
                    success=True,
                    result=refactor_result,
                    warnings=warnings,
                    duration=duration,
                )
            )

        # Step 6: Documentation Analysis
        if self.config.enable_documentation:
            start = time.time()
            doc_result = self.doc_analyzer.analyze(current_code)
            duration = time.time() - start

            warnings = doc_result.issues
            recommendations.extend(doc_result.suggestions)

            steps.append(
                WorkflowStep(
                    name="Documentation Analysis",
                    success=doc_result.meets_threshold,
                    result=doc_result,
                    warnings=warnings,
                    duration=duration,
                )
            )

        # Generate summary
        summary = {
            "total_steps": len(steps),
            "successful_steps": sum(1 for s in steps if s.success),
            "total_errors": sum(len(s.errors) for s in steps),
            "total_warnings": sum(len(s.warnings) for s in steps),
            "code_changed": current_code != code,
        }

        return QualityReport(
            success=all(s.success for s in steps),
            steps=steps,
            final_code=current_code,
            summary=summary,
            recommendations=list(set(recommendations)),  # Deduplicate
            metadata={
                "config": self.config.to_dict(),
            },
        )

    def quick_check(self, code: str) -> bool:
        """
        Quick quality check (analysis + linting only).

        Args:
            code: Python code to check

        Returns:
            True if code passes quick check, False otherwise
        """
        # Analysis
        analysis_result = self.analyzer.analyze(code)
        if not analysis_result.get("valid"):
            return False

        # Linting
        lint_result = self.linter.lint(code)
        return not lint_result.has_errors

    def format_and_fix(self, code: str) -> str:
        """
        Format code and auto-fix issues.

        Args:
            code: Python code to fix

        Returns:
            Fixed and formatted code
        """
        # Auto-fix linting issues
        lint_result = self.linter.lint(code)
        if lint_result.fixed_code:
            code = lint_result.fixed_code

        # Format code
        format_result = self.formatter.format(code)
        return format_result.code

    def get_metadata(self) -> WorkflowMetadata:
        """Get workflow metadata."""
        return WorkflowMetadata(
            name="QualityWorkflow",
            description="Comprehensive code quality analysis and improvement workflow",
            version="1.0.0",
            author="The Augster",
            tags=["quality", "analysis", "linting", "formatting"],
            capabilities=["code_analysis", "linting", "formatting", "refactoring", "documentation"],
            required_inputs=["code"],
            optional_inputs=[],
            output_schema={
                "final_code": "str",
                "summary": "dict",
                "recommendations": "list[str]",
                "steps": "list[dict]",
            },
        )
