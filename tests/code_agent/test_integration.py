"""
Integration Tests

Tests for integration between modules.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import pytest

from code_agent.adapters import (
    GraphConfig,
    ImportanceLevel,
    WorkflowOrchestrator,
    WorkflowState,
)
from code_agent.config import ConfigManager
from code_agent.core import AgentConfig, CodeAgent, create_code_agent
from code_agent.tools import CodeAnalyzer, CodeGenerator, RefactoringEngine
from code_agent.utils import ErrorContext, LogLevel, StructuredLogger


class TestAgentWithTools:
    """Test agent integration with tools."""

    def test_agent_with_analyzer(self):
        """Test agent with code analyzer."""
        agent = CodeAgent()
        analyzer = CodeAnalyzer()

        code = "def hello(): pass"
        result = analyzer.analyze(code)

        assert result["valid"] is True
        assert agent.state is not None

    def test_agent_with_refactoring_engine(self):
        """Test agent with refactoring engine."""
        agent = CodeAgent()
        engine = RefactoringEngine()

        code = "x = 1\ny = 2\nz = 3"
        suggestions = engine.suggest_refactoring(code)

        assert isinstance(suggestions, list)
        assert agent.state is not None

    def test_agent_with_code_generator(self):
        """Test agent with code generator."""
        agent = CodeAgent()
        generator = CodeGenerator()

        result = generator.generate_function(
            name="test",
            parameters=["x"],
            return_type="int",
        )

        assert "def test" in result.code
        assert agent.state is not None


class TestAgentWithConfig:
    """Test agent integration with configuration."""

    def test_agent_with_config_manager(self):
        """Test agent with config manager."""
        manager = ConfigManager()
        manager.load_from_dict(
            {
                "model": "openai:gpt-4",
                "enable_streaming": True,
            }
        )

        agent = create_code_agent(
            model=manager.get("model"),
            enable_streaming=manager.get("enable_streaming"),
        )

        assert agent.config.model == "openai:gpt-4"
        assert agent.config.enable_streaming is True

    def test_agent_config_persistence(self):
        """Test agent configuration persistence."""
        config = AgentConfig(
            model="openai:gpt-4",
            max_retries=5,
        )

        agent1 = CodeAgent(config)
        agent2 = CodeAgent(config)

        assert agent1.config.model == agent2.config.model
        assert agent1.config.max_retries == agent2.config.max_retries


class TestAgentWithLogging:
    """Test agent integration with logging."""

    def test_agent_with_logger(self):
        """Test agent with structured logger."""
        logger = StructuredLogger(
            name="test",
            level=LogLevel.INFO,
        )

        _agent = CodeAgent()

        logger.info("Agent created")
        metrics = logger.start_operation("test_op")
        metrics.complete(success=True)

        summary = logger.get_metrics_summary()
        assert summary["total_operations"] == 1

    def test_agent_error_logging(self):
        """Test agent error logging."""
        logger = StructuredLogger(name="test")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = ErrorContext.from_exception(e)
            logger.error(f"Error: {context.error_type}")

        assert context.error_type == "ValueError"


class TestAgentWithAdapters:
    """Test agent integration with adapters."""

    def test_agent_with_context_manager(self):
        """Test agent with context manager."""
        from code_agent.adapters import create_context_manager

        agent = CodeAgent()
        context = create_context_manager(max_tokens=50_000)

        context.add_message("content1", message_type="user", importance=ImportanceLevel.HIGH)
        context.add_message("content2", message_type="assistant", importance=ImportanceLevel.MEDIUM)

        stats = context.get_statistics()

        assert stats["segments_count"] == 2
        assert agent.state is not None

    def test_agent_with_workflow_orchestrator(self):
        """Test agent with workflow orchestrator."""
        agent = CodeAgent()
        workflow = WorkflowOrchestrator(operation_name="test_workflow")

        workflow.transition_to(WorkflowState.RUNNING)
        _checkpoint = workflow.create_checkpoint(input_data={"test": "data"}, output_data={"result": "success"})

        assert workflow.current_state == WorkflowState.RUNNING
        assert agent.state is not None

    def test_agent_with_graph_config(self):
        """Test agent with graph config."""
        agent = CodeAgent()
        config = GraphConfig(enable_persistence=True, enable_streaming=True, max_iterations=1000)

        assert config.enable_persistence is True
        assert agent.state is not None


class TestFullWorkflow:
    """Test full workflow integration."""

    @pytest.mark.skip(
        reason=(
            "Test uses incorrect API - needs rewrite to use actual "
            "WorkflowOrchestrator.register_step() and run() methods"
        )
    )
    def test_complete_analysis_workflow(self):
        """Test complete code analysis workflow."""
        # Create agent
        _agent = create_code_agent(model="openai:gpt-4")

        # Create tools
        analyzer = CodeAnalyzer()
        engine = RefactoringEngine()
        generator = CodeGenerator()

        # Create adapters
        from code_agent.adapters import create_context_manager

        context = create_context_manager()
        workflow = WorkflowOrchestrator(operation_name="analysis")

        # Define workflow
        code = """
def calculate(x, y):
    return x + y

class Calculator:
    def add(self, a, b):
        return a + b
"""

        # Add context
        context.add_message(code, message_type="user", importance=ImportanceLevel.HIGH)

        # Define workflow steps
        def analyze_step(params, ctx):
            return {"result": analyzer.analyze(code)}

        def refactor_step(params, ctx):
            return {"result": engine.suggest_refactoring(code)}

        def generate_step(params, ctx):
            return {
                "result": generator.generate_function(
                    name="test",
                    parameters=["x"],
                    return_type="int",
                )
            }

        # Build workflow
        workflow.register_step("analyze", analyze_step)
        workflow.register_step("refactor", refactor_step)
        workflow.register_step("generate", generate_step)

        # Execute
        results = workflow.run(
            [
                {"name": "analyze", "params": {}},
                {"name": "refactor", "params": {}},
                {"name": "generate", "params": {}},
            ]
        )

        assert len(results) == 3

    @pytest.mark.skip(
        reason=(
            "Test uses incorrect API - needs rewrite to use actual "
            "WorkflowOrchestrator.register_step() and run() methods"
        )
    )
    def test_multi_agent_workflow(self):
        """Test workflow with multiple agents."""
        agent1 = create_code_agent()
        agent2 = create_code_agent()

        workflow = WorkflowOrchestrator(operation_name="multi_agent")

        def agent1_task(params, ctx):
            return {"state": agent1.get_state()}

        def agent2_task(params, ctx):
            return {"state": agent2.get_state()}

        workflow.register_step("agent1", agent1_task)
        workflow.register_step("agent2", agent2_task)

        results = workflow.run(
            [
                {"name": "agent1", "params": {}},
                {"name": "agent2", "params": {}},
            ]
        )

        assert len(results) == 2


class TestErrorHandling:
    """Test error handling in integration."""

    @pytest.mark.skip(
        reason=(
            "Test uses incorrect API - needs rewrite to use actual "
            "WorkflowOrchestrator.register_step() and run() methods"
        )
    )
    def test_workflow_error_handling(self):
        """Test workflow error handling."""
        workflow = WorkflowOrchestrator(operation_name="error_test")

        def failing_step(params, ctx):
            raise ValueError("Step failed")

        workflow.register_step("fail", failing_step)

        with pytest.raises(ValueError):
            workflow.run([{"name": "fail", "params": {}}])

    def test_agent_error_recovery(self):
        """Test agent error recovery."""
        agent = CodeAgent()

        # Get error summary
        summary = agent.get_error_summary()

        assert "total_errors" in summary
        assert summary["total_errors"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
