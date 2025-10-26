# Enhanced Workflow System

The enhanced workflow system provides a flexible, composable framework for building complex multi-step workflows with support for custom workflows, cross-referencing, and composition patterns.

## Features

### 1. Custom Workflow Support

Define and register your own custom workflows by inheriting from `BaseWorkflow`:

```python
from code_agent.workflows import BaseWorkflow, WorkflowContext, WorkflowResult, WorkflowMetadata, WorkflowStatus

class MyCustomWorkflow(BaseWorkflow):
    """Custom workflow implementation."""

    def run(self, context: WorkflowContext) -> WorkflowResult:
        """Execute the workflow."""
        # Get inputs from context
        input_data = context.get_input("data")

        # Perform workflow logic
        result_data = process(input_data)

        # Return result
        return WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name="MyCustomWorkflow",
            status=WorkflowStatus.COMPLETED,
            success=True,
            outputs={"result": result_data},
            context=context,
        )

    def get_metadata(self) -> WorkflowMetadata:
        """Provide workflow metadata."""
        return WorkflowMetadata(
            name="MyCustomWorkflow",
            description="My custom workflow",
            capabilities=["processing"],
            tags=["custom"],
            required_inputs=["data"],
        )
```

### 2. Workflow Registration and Discovery

Register workflows in a central registry for discovery and reuse:

```python
from code_agent.workflows import WorkflowRegistry

# Create registry
registry = WorkflowRegistry()

# Register workflow
workflow = MyCustomWorkflow()
registry.register(workflow)

# Discover workflows
analysis_workflows = registry.find_by_capability("analysis")
quality_workflows = registry.find_by_tag("quality")

# Get workflow by name
workflow = registry.get_workflow_by_name("MyCustomWorkflow")
```

### 3. Workflow Cross-Referencing

Workflows can invoke other workflows by name or ID:

```python
from code_agent.workflows import WorkflowExecutor

# Create executor with registry
executor = WorkflowExecutor(registry)

# Execute workflow by name
result = executor.execute_by_name("MyCustomWorkflow", inputs={"data": "..."})

# Execute workflow by ID
result = executor.execute_by_id(workflow_id, inputs={"data": "..."})
```

**Cycle Detection**: The executor automatically detects and prevents infinite loops:

```python
# This will raise WorkflowCycleError if A calls B and B calls A
executor = WorkflowExecutor(registry, enable_cycle_detection=True)
```

### 4. Workflow Composition

#### Sequential Composition

Chain workflows where each workflow's output becomes the next workflow's input:

```python
from code_agent.workflows import SequentialWorkflow

sequential = SequentialWorkflow(
    workflows=["AnalysisWorkflow", "FormattingWorkflow", "ValidationWorkflow"],
    executor=executor,
    continue_on_error=False,  # Stop on first failure
)

result = sequential.run(context)
```

#### Parallel Execution

Execute multiple workflows concurrently:

```python
from code_agent.workflows import ParallelWorkflow

parallel = ParallelWorkflow(
    workflows=["LintingWorkflow", "TypeCheckWorkflow", "SecurityScanWorkflow"],
    executor=executor,
    max_workers=3,  # Limit concurrent workers
)

result = parallel.run(context)
```

#### Conditional Branching

Execute different workflows based on runtime conditions:

```python
from code_agent.workflows import ConditionalWorkflow, WorkflowBranch

conditional = ConditionalWorkflow(
    branches=[
        WorkflowBranch(
            condition=lambda ctx: ctx.get_input("language") == "python",
            workflow="PythonAnalysisWorkflow",
            name="python_branch",
        ),
        WorkflowBranch(
            condition=lambda ctx: ctx.get_input("language") == "javascript",
            workflow="JavaScriptAnalysisWorkflow",
            name="javascript_branch",
        ),
    ],
    executor=executor,
    default_workflow="GenericAnalysisWorkflow",
)

result = conditional.run(context)
```

### 5. Context and State Management

Pass state between workflows using `WorkflowContext`:

```python
from code_agent.workflows import WorkflowContext

# Create context with inputs
context = WorkflowContext(
    workflow_id="my-workflow",
    inputs={"code": "def foo(): pass"},
)

# Access inputs
code = context.get_input("code")

# Set outputs
context.set_output("result", processed_code)

# Share state across workflows
context.set_shared("config", {"strict": True})
shared_config = context.get_shared("config")
```

## Complete Example

```python
from code_agent.workflows import (
    BaseWorkflow,
    WorkflowRegistry,
    WorkflowExecutor,
    SequentialWorkflow,
    WorkflowContext,
)

# 1. Define custom workflows
class AnalysisWorkflow(BaseWorkflow):
    def run(self, context):
        code = context.get_input("code")
        # Analyze code...
        return WorkflowResult(...)

    def get_metadata(self):
        return WorkflowMetadata(name="AnalysisWorkflow", ...)

class FormattingWorkflow(BaseWorkflow):
    def run(self, context):
        code = context.get_input("code")
        # Format code...
        return WorkflowResult(...)

    def get_metadata(self):
        return WorkflowMetadata(name="FormattingWorkflow", ...)

# 2. Register workflows
registry = WorkflowRegistry()
registry.register(AnalysisWorkflow())
registry.register(FormattingWorkflow())

# 3. Create executor
executor = WorkflowExecutor(registry)

# 4. Compose workflows
pipeline = SequentialWorkflow(
    workflows=["AnalysisWorkflow", "FormattingWorkflow"],
    executor=executor,
)

# 5. Execute
context = WorkflowContext(
    workflow_id="pipeline-1",
    inputs={"code": "def example(): pass"},
)

result = pipeline.run(context)
print(f"Success: {result.success}")
print(f"Outputs: {result.outputs}")
```

## Error Handling

The workflow system provides comprehensive error handling:

```python
from code_agent.workflows import (
    WorkflowError,
    WorkflowNotFoundError,
    WorkflowCycleError,
    WorkflowValidationError,
    WorkflowExecutionError,
)

try:
    result = executor.execute_by_name("NonExistentWorkflow")
except WorkflowNotFoundError as e:
    print(f"Workflow not found: {e}")

try:
    result = workflow.run(context)
except WorkflowValidationError as e:
    print(f"Validation failed: {e}")

try:
    result = executor.execute(circular_workflow)
except WorkflowCycleError as e:
    print(f"Cycle detected: {e}")
```

## Backward Compatibility

Existing workflows (like `QualityWorkflow`) maintain backward compatibility:

```python
from code_agent.workflows import QualityWorkflow

# Legacy interface (still works)
workflow = QualityWorkflow()
report = workflow.run("def foo(): pass")

# New interface (also works)
context = WorkflowContext(workflow_id="test", inputs={"code": "def foo(): pass"})
result = workflow.run(context)
```

## Best Practices

1. **Define Clear Metadata**: Provide comprehensive metadata for workflow discovery
2. **Validate Inputs**: Implement `validate()` method to check inputs before execution
3. **Handle Errors Gracefully**: Return meaningful error messages in `WorkflowResult`
4. **Use Context for State**: Pass state through `WorkflowContext` rather than global variables
5. **Avoid Deep Nesting**: Keep workflow composition depth reasonable (default max: 10)
6. **Enable Cycle Detection**: Always enable cycle detection in production
7. **Document Dependencies**: List workflow dependencies in metadata

## Examples

See `code_agent/examples/workflow_composition.py` for complete working examples demonstrating:
- Custom workflow creation
- Workflow registration and discovery
- Cross-referencing workflows
- Sequential composition
- Parallel execution
- Conditional branching

## API Reference

### Core Classes

- **`BaseWorkflow`**: Abstract base class for all workflows
- **`WorkflowContext`**: Context for passing state between workflows
- **`WorkflowResult`**: Standardized workflow execution result
- **`WorkflowMetadata`**: Workflow metadata for discovery
- **`WorkflowRegistry`**: Registry for workflow management
- **`WorkflowExecutor`**: Execution engine with cross-referencing

### Composition Classes

- **`SequentialWorkflow`**: Execute workflows in sequence
- **`ParallelWorkflow`**: Execute workflows in parallel
- **`ConditionalWorkflow`**: Execute workflows based on conditions
- **`WorkflowBranch`**: Conditional branch definition

### Exceptions

- **`WorkflowError`**: Base exception for workflow errors
- **`WorkflowNotFoundError`**: Workflow not found in registry
- **`WorkflowCycleError`**: Workflow cycle detected
- **`WorkflowValidationError`**: Workflow validation failed
- **`WorkflowExecutionError`**: Workflow execution failed
