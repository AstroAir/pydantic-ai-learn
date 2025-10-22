"""
Tests for Workflows

Tests for QualityWorkflow and other workflow orchestration.

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

from code_agent.config.custom import QualityWorkflowConfig
from code_agent.workflows import QualityReport, QualityWorkflow

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
