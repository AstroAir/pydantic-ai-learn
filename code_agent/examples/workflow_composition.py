"""
Workflow Composition Examples

Demonstrates workflow composition features:
1. Custom workflow creation
2. Workflow registration and discovery
3. Workflow cross-referencing
4. Workflow composition (sequential, parallel, conditional)
5. Cycle detection

Author: The Augster
Python Version: 3.11+
"""

from __future__ import annotations

from code_agent.workflows import (
    BaseWorkflow,
    ConditionalWorkflow,
    ParallelWorkflow,
    SequentialWorkflow,
    WorkflowBranch,
    WorkflowContext,
    WorkflowExecutor,
    WorkflowMetadata,
    WorkflowRegistry,
    WorkflowResult,
    WorkflowStatus,
)

# ============================================================================
# Example 1: Custom Workflow Creation
# ============================================================================


class SimpleAnalysisWorkflow(BaseWorkflow):
    """Simple custom workflow for code analysis."""

    def run(self, context: WorkflowContext) -> WorkflowResult:
        """Execute simple analysis."""
        code = context.get_input("code", "")

        # Simple analysis: count lines and functions
        lines = code.split("\n")
        functions = [line for line in lines if line.strip().startswith("def ")]

        return WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name="SimpleAnalysisWorkflow",
            status=WorkflowStatus.COMPLETED,
            success=True,
            outputs={
                "line_count": len(lines),
                "function_count": len(functions),
                "functions": [f.strip() for f in functions],
            },
            context=context,
        )

    def get_metadata(self) -> WorkflowMetadata:
        """Get workflow metadata."""
        return WorkflowMetadata(
            name="SimpleAnalysisWorkflow",
            description="Simple code analysis workflow",
            capabilities=["analysis", "metrics"],
            tags=["analysis", "simple"],
            required_inputs=["code"],
        )


class FormattingWorkflow(BaseWorkflow):
    """Simple custom workflow for code formatting."""

    def run(self, context: WorkflowContext) -> WorkflowResult:
        """Execute simple formatting."""
        code = context.get_input("code", "")

        # Simple formatting: ensure consistent indentation
        lines = code.split("\n")
        formatted_lines = [line.replace("\t", "    ") for line in lines]
        formatted_code = "\n".join(formatted_lines)

        return WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name="FormattingWorkflow",
            status=WorkflowStatus.COMPLETED,
            success=True,
            outputs={
                "formatted_code": formatted_code,
                "changed": formatted_code != code,
            },
            context=context,
        )

    def get_metadata(self) -> WorkflowMetadata:
        """Get workflow metadata."""
        return WorkflowMetadata(
            name="FormattingWorkflow",
            description="Simple code formatting workflow",
            capabilities=["formatting"],
            tags=["formatting", "simple"],
            required_inputs=["code"],
        )


def example_1_custom_workflow() -> None:
    """Example 1: Create and execute a custom workflow."""
    print("\n" + "=" * 60)
    print("Example 1: Custom Workflow Creation")
    print("=" * 60)

    # Create custom workflow
    workflow = SimpleAnalysisWorkflow()

    # Create context with input
    context = WorkflowContext(
        workflow_id="test-1",
        inputs={"code": "def foo():\n    pass\n\ndef bar():\n    return 42\n"},
    )

    # Execute workflow
    result = workflow.run(context)

    print(f"\nWorkflow: {result.workflow_name}")
    print(f"Status: {result.status.value}")
    print(f"Success: {result.success}")
    print(f"Outputs: {result.outputs}")


# ============================================================================
# Example 2: Workflow Registration and Discovery
# ============================================================================


def example_2_registry() -> None:
    """Example 2: Register and discover workflows."""
    print("\n" + "=" * 60)
    print("Example 2: Workflow Registration and Discovery")
    print("=" * 60)

    # Create registry
    registry = WorkflowRegistry()

    # Register workflows
    analysis_workflow = SimpleAnalysisWorkflow()
    formatting_workflow = FormattingWorkflow()

    registry.register(analysis_workflow)
    registry.register(formatting_workflow)

    print(f"\nTotal workflows: {len(registry.list_workflows())}")

    # Discover by capability
    analysis_workflows = registry.find_by_capability("analysis")
    print(f"Analysis workflows: {[w.get_metadata().name for w in analysis_workflows]}")

    formatting_workflows = registry.find_by_capability("formatting")
    print(f"Formatting workflows: {[w.get_metadata().name for w in formatting_workflows]}")

    # Discover by tag
    simple_workflows = registry.find_by_tag("simple")
    print(f"Simple workflows: {[w.get_metadata().name for w in simple_workflows]}")

    # Get workflow by name
    workflow = registry.get_workflow_by_name("SimpleAnalysisWorkflow")
    print(f"\nRetrieved workflow: {workflow.get_metadata().name if workflow else 'Not found'}")


# ============================================================================
# Example 3: Workflow Cross-Referencing
# ============================================================================


class CompositeAnalysisWorkflow(BaseWorkflow):
    """Workflow that references other workflows."""

    def __init__(self, executor: WorkflowExecutor, workflow_id: str | None = None) -> None:
        super().__init__(workflow_id)
        self.executor = executor

    def run(self, context: WorkflowContext) -> WorkflowResult:
        """Execute by calling other workflows."""
        # Call analysis workflow with inputs
        analysis_result = self.executor.execute_by_name("SimpleAnalysisWorkflow", inputs=context.inputs)

        # Call formatting workflow with inputs
        formatting_result = self.executor.execute_by_name("FormattingWorkflow", inputs=context.inputs)

        # Combine results
        return WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name="CompositeAnalysisWorkflow",
            status=WorkflowStatus.COMPLETED,
            success=analysis_result.success and formatting_result.success,
            outputs={
                "analysis": analysis_result.outputs,
                "formatting": formatting_result.outputs,
            },
            context=context,
        )

    def validate(self, context: WorkflowContext) -> tuple[bool, list[str]]:
        """Validate workflow inputs."""
        # CompositeAnalysisWorkflow requires 'code' input
        if "code" not in context.inputs:
            return False, ["Missing required input: code"]
        return True, []

    def get_metadata(self) -> WorkflowMetadata:
        """Get workflow metadata."""
        return WorkflowMetadata(
            name="CompositeAnalysisWorkflow",
            description="Composite workflow that calls other workflows",
            capabilities=["analysis", "formatting", "composite"],
            tags=["composite", "cross-reference"],
            required_inputs=["code"],
            dependencies=["SimpleAnalysisWorkflow", "FormattingWorkflow"],
        )


def example_3_cross_referencing() -> None:
    """Example 3: Workflow cross-referencing."""
    print("\n" + "=" * 60)
    print("Example 3: Workflow Cross-Referencing")
    print("=" * 60)

    # Create registry and executor
    registry = WorkflowRegistry()
    executor = WorkflowExecutor(registry)

    # Register base workflows
    registry.register(SimpleAnalysisWorkflow())
    registry.register(FormattingWorkflow())

    # Register composite workflow
    composite_workflow = CompositeAnalysisWorkflow(executor)
    registry.register(composite_workflow)

    # Execute composite workflow
    result = executor.execute(composite_workflow, inputs={"code": "def test():\n\tpass\n"})

    print(f"\nWorkflow: {result.workflow_name}")
    print(f"Success: {result.success}")
    print(f"Analysis outputs: {result.outputs.get('analysis')}")
    print(f"Formatting outputs: {result.outputs.get('formatting')}")


# ============================================================================
# Example 4: Sequential Workflow Composition
# ============================================================================


def example_4_sequential_composition() -> None:
    """Example 4: Sequential workflow composition."""
    print("\n" + "=" * 60)
    print("Example 4: Sequential Workflow Composition")
    print("=" * 60)

    # Create registry and executor
    registry = WorkflowRegistry()
    executor = WorkflowExecutor(registry)

    # Register workflows
    registry.register(SimpleAnalysisWorkflow())
    registry.register(FormattingWorkflow())

    # Create sequential workflow
    sequential = SequentialWorkflow(
        workflows=["SimpleAnalysisWorkflow", "FormattingWorkflow"],
        executor=executor,
    )

    # Execute
    result = executor.execute(sequential, inputs={"code": "def example():\n\treturn 1\n"})

    print(f"\nWorkflow: {result.workflow_name}")
    print(f"Success: {result.success}")
    print(f"Results: {len(result.outputs.get('results', []))} workflows executed")


# ============================================================================
# Example 5: Parallel Workflow Execution
# ============================================================================


def example_5_parallel_execution() -> None:
    """Example 5: Parallel workflow execution."""
    print("\n" + "=" * 60)
    print("Example 5: Parallel Workflow Execution")
    print("=" * 60)

    # Create registry and executor
    registry = WorkflowRegistry()
    executor = WorkflowExecutor(registry)

    # Register workflows
    registry.register(SimpleAnalysisWorkflow())
    registry.register(FormattingWorkflow())

    # Create parallel workflow
    parallel = ParallelWorkflow(
        workflows=["SimpleAnalysisWorkflow", "FormattingWorkflow"],
        executor=executor,
        max_workers=2,
    )

    # Execute
    result = executor.execute(parallel, inputs={"code": "def parallel_test():\n    return True\n"})

    print(f"\nWorkflow: {result.workflow_name}")
    print(f"Success: {result.success}")
    print(f"Results: {len(result.outputs.get('results', []))} workflows executed in parallel")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Enhanced Workflow System Examples")
    print("=" * 60)

    try:
        example_1_custom_workflow()
        example_2_registry()
        example_3_cross_referencing()
        example_4_sequential_composition()
        example_5_parallel_execution()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
