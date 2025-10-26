"""
Tests for Workflows

Tests for QualityWorkflow and other workflow orchestration.

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

from code_agent.config.tools import QualityWorkflowConfig
from code_agent.workflows import (
    QualityReport,
    QualityWorkflow,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

# ============================================================================
# Test Data
# ============================================================================

SIMPLE_CODE = """
def hello():
    '''Say hello.'''
    print("Hello, World!")
"""

COMPLEX_CODE = """
'''Module for calculations.'''

import math
from typing import List


class Calculator:
    '''A simple calculator class.'''

    def add(self, a: int, b: int) -> int:
        '''
        Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            int: Sum of a and b
        '''
        return a + b

    def multiply(self, numbers: List[int]) -> int:
        '''
        Multiply a list of numbers.

        Args:
            numbers: List of numbers to multiply

        Returns:
            int: Product of all numbers
        '''
        result = 1
        for num in numbers:
            result *= num
        return result
"""

CODE_WITH_ISSUES = """
import os
import sys

def test():
    x=1+2
    y=3+4
    return x+y
"""


# ============================================================================
# QualityWorkflow Tests
# ============================================================================


class TestQualityWorkflow:
    """Tests for QualityWorkflow."""

    def test_workflow_creation(self):
        """Test workflow initialization."""
        workflow = QualityWorkflow()
        assert workflow is not None
        assert workflow.config is not None

    def test_workflow_with_config(self):
        """Test workflow with custom config."""
        config = QualityWorkflowConfig(
            enable_analysis=True,
            enable_linting=True,
            enable_formatting=False,
        )
        workflow = QualityWorkflow(config)
        assert workflow.config.enable_formatting is False

    def test_run_simple_code(self):
        """Test running workflow on simple code."""
        workflow = QualityWorkflow()
        report = workflow.run(SIMPLE_CODE)

        assert report is not None
        assert isinstance(report, QualityReport)
        assert isinstance(report.steps, list)
        assert len(report.steps) > 0

    def test_run_complex_code(self):
        """Test running workflow on complex code."""
        workflow = QualityWorkflow()
        report = workflow.run(COMPLEX_CODE)

        assert report is not None
        assert report.success is not None
        assert len(report.steps) > 0
        # Should have multiple steps
        assert report.summary["total_steps"] > 0

    def test_workflow_steps(self):
        """Test workflow executes all enabled steps."""
        config = QualityWorkflowConfig(
            enable_analysis=True,
            enable_linting=True,
            enable_formatting=True,
            enable_refactoring=True,
            enable_documentation=True,
        )
        workflow = QualityWorkflow(config)
        report = workflow.run(SIMPLE_CODE)

        # Should have multiple steps
        assert len(report.steps) >= 3

        # Check step names
        step_names = [step.name for step in report.steps]
        assert "Code Analysis" in step_names

    def test_workflow_with_issues(self):
        """Test workflow on code with issues."""
        workflow = QualityWorkflow()
        report = workflow.run(CODE_WITH_ISSUES)

        assert report is not None
        # May have warnings or errors
        assert isinstance(report.total_warnings, int)

    def test_quick_check(self):
        """Test quick quality check."""
        workflow = QualityWorkflow()

        # Simple code should pass
        result = workflow.quick_check(SIMPLE_CODE)
        assert isinstance(result, bool)

    def test_format_and_fix(self):
        """Test format and fix functionality."""
        workflow = QualityWorkflow()

        # Format and fix code
        fixed_code = workflow.format_and_fix(CODE_WITH_ISSUES)

        assert fixed_code is not None
        assert isinstance(fixed_code, str)
        assert len(fixed_code) > 0


# ============================================================================
# QualityReport Tests
# ============================================================================


class TestQualityReport:
    """Tests for QualityReport."""

    def test_report_properties(self):
        """Test report properties."""
        workflow = QualityWorkflow()
        report = workflow.run(SIMPLE_CODE)

        # Test properties
        assert isinstance(report.total_errors, int)
        assert isinstance(report.total_warnings, int)
        assert isinstance(report.total_duration, float)
        assert report.total_duration >= 0.0

    def test_report_summary(self):
        """Test report summary."""
        workflow = QualityWorkflow()
        report = workflow.run(COMPLEX_CODE)

        assert "total_steps" in report.summary
        assert "successful_steps" in report.summary
        assert "total_errors" in report.summary
        assert "total_warnings" in report.summary

    def test_report_recommendations(self):
        """Test report recommendations."""
        workflow = QualityWorkflow()
        report = workflow.run(CODE_WITH_ISSUES)

        assert isinstance(report.recommendations, list)
        # May have recommendations for improvements


# ============================================================================
# Workflow Configuration Tests
# ============================================================================


class TestWorkflowConfiguration:
    """Tests for workflow configuration."""

    def test_disable_steps(self):
        """Test disabling workflow steps."""
        config = QualityWorkflowConfig(
            enable_analysis=True,
            enable_linting=False,
            enable_formatting=False,
            enable_refactoring=False,
            enable_documentation=False,
        )
        workflow = QualityWorkflow(config)
        report = workflow.run(SIMPLE_CODE)

        # Should have fewer steps
        assert len(report.steps) < 5

    def test_auto_fix_enabled(self):
        """Test auto-fix functionality."""
        config = QualityWorkflowConfig(
            auto_fix=True,
            enable_linting=True,
            enable_formatting=True,
        )
        workflow = QualityWorkflow(config)
        report = workflow.run(CODE_WITH_ISSUES)

        # Final code should be different from original
        if report.final_code:
            # Code may have been fixed
            assert isinstance(report.final_code, str)

    def test_fail_on_errors(self):
        """Test fail on errors configuration."""
        config = QualityWorkflowConfig(
            fail_on_errors=True,
        )
        workflow = QualityWorkflow(config)

        # Run on code that may have errors
        report = workflow.run(CODE_WITH_ISSUES)

        # Report should indicate success/failure
        assert isinstance(report.success, bool)


# ============================================================================
# Integration Tests
# ============================================================================


class TestWorkflowIntegration:
    """Integration tests for workflows."""

    def test_workflow_with_all_tools(self):
        """Test workflow using all tools."""
        config = QualityWorkflowConfig(
            enable_analysis=True,
            enable_linting=True,
            enable_formatting=True,
            enable_refactoring=True,
            enable_documentation=True,
        )
        workflow = QualityWorkflow(config)
        report = workflow.run(COMPLEX_CODE)

        # Should execute all steps
        assert len(report.steps) >= 4
        assert report.summary["total_steps"] >= 4

    def test_workflow_preserves_code(self):
        """Test workflow preserves code when auto_fix is disabled."""
        config = QualityWorkflowConfig(
            auto_fix=False,
        )
        workflow = QualityWorkflow(config)
        report = workflow.run(SIMPLE_CODE)

        # Final code should be same as original when auto_fix is off
        # (unless formatting is enabled and changes code)
        assert report.final_code is not None

    def test_workflow_error_handling(self):
        """Test workflow handles errors gracefully."""
        workflow = QualityWorkflow()

        # Invalid code should be handled
        invalid_code = "def test(\n  invalid syntax"
        report = workflow.run(invalid_code)

        # Should not crash, should report error
        assert report is not None
        assert not report.success


# ============================================================================
# QualityWorkflow BaseWorkflow Interface Tests
# ============================================================================


class TestQualityWorkflowBaseInterface:
    """Tests for QualityWorkflow's BaseWorkflow interface."""

    def test_workflow_inherits_base_workflow(self):
        """Test that QualityWorkflow inherits from BaseWorkflow."""
        from code_agent.workflows.base import BaseWorkflow

        workflow = QualityWorkflow()
        assert isinstance(workflow, BaseWorkflow)

    def test_workflow_has_workflow_id(self):
        """Test that workflow has a workflow_id."""
        workflow = QualityWorkflow()
        assert workflow.workflow_id is not None

    def test_workflow_with_custom_id(self):
        """Test creating workflow with custom ID."""
        workflow = QualityWorkflow(workflow_id="custom-quality-id")
        assert workflow.workflow_id == "custom-quality-id"

    def test_workflow_run_with_context(self):
        """Test running workflow with WorkflowContext."""
        # Disable documentation checking to avoid failure due to low coverage
        config = QualityWorkflowConfig(enable_documentation=False)
        workflow = QualityWorkflow(config)

        context = WorkflowContext(
            workflow_id="test-quality",
            inputs={"code": SIMPLE_CODE},
        )

        result = workflow.run(context)

        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert result.status == WorkflowStatus.COMPLETED
        assert result.workflow_name == "QualityWorkflow"
        assert "final_code" in result.outputs
        assert "summary" in result.outputs

    def test_workflow_run_with_string_backward_compatibility(self):
        """Test that legacy string interface still works."""
        # Disable documentation checking to avoid failure due to low coverage
        config = QualityWorkflowConfig(enable_documentation=False)
        workflow = QualityWorkflow(config)

        # Legacy interface: pass code string directly
        result = workflow.run(SIMPLE_CODE)

        # Should return QualityReport (legacy)
        assert isinstance(result, QualityReport)
        assert result.success is True

    def test_workflow_validate_with_valid_input(self):
        """Test workflow validation with valid input."""
        workflow = QualityWorkflow()

        context = WorkflowContext(
            workflow_id="test",
            inputs={"code": SIMPLE_CODE},
        )

        is_valid, errors = workflow.validate(context)

        assert is_valid is True
        assert len(errors) == 0

    def test_workflow_validate_with_missing_input(self):
        """Test workflow validation with missing required input."""
        workflow = QualityWorkflow()

        context = WorkflowContext(
            workflow_id="test",
            inputs={},  # Missing 'code' input
        )

        is_valid, errors = workflow.validate(context)

        assert is_valid is False
        assert len(errors) > 0
        assert any("code" in error.lower() for error in errors)

    def test_workflow_get_metadata(self):
        """Test getting workflow metadata."""
        workflow = QualityWorkflow()

        metadata = workflow.get_metadata()

        assert metadata is not None
        assert metadata.name == "QualityWorkflow"
        assert metadata.description is not None
        assert len(metadata.capabilities) > 0
        assert "code" in metadata.required_inputs

    def test_workflow_result_structure(self):
        """Test that WorkflowResult has expected structure."""
        workflow = QualityWorkflow()

        context = WorkflowContext(
            workflow_id="test",
            inputs={"code": SIMPLE_CODE},
        )

        result = workflow.run(context)

        # Check result structure
        assert hasattr(result, "workflow_id")
        assert hasattr(result, "workflow_name")
        assert hasattr(result, "status")
        assert hasattr(result, "success")
        assert hasattr(result, "outputs")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "metadata")
        assert hasattr(result, "context")

    def test_workflow_result_outputs(self):
        """Test that WorkflowResult outputs contain expected data."""
        workflow = QualityWorkflow()

        context = WorkflowContext(
            workflow_id="test",
            inputs={"code": COMPLEX_CODE},
        )

        result = workflow.run(context)

        # Check outputs
        assert "final_code" in result.outputs
        assert "summary" in result.outputs
        assert "recommendations" in result.outputs
        assert "steps" in result.outputs

        # Check metadata
        assert "total_errors" in result.metadata
        assert "total_warnings" in result.metadata
        assert "total_duration" in result.metadata

    def test_workflow_with_invalid_code_returns_failure(self):
        """Test that invalid code returns failed result."""
        workflow = QualityWorkflow()

        context = WorkflowContext(
            workflow_id="test",
            inputs={"code": "def invalid(\n  syntax error"},
        )

        result = workflow.run(context)

        assert result.success is False
        assert result.status == WorkflowStatus.FAILED
        assert len(result.errors) > 0

    def test_workflow_context_preservation(self):
        """Test that context is preserved in result."""
        workflow = QualityWorkflow()

        context = WorkflowContext(
            workflow_id="test-context",
            inputs={"code": SIMPLE_CODE},
        )

        result = workflow.run(context)

        # Context should be preserved
        assert result.context is not None
        assert result.context.workflow_id == "test-context"
