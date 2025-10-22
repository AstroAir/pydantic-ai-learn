from dataclasses import dataclass, field
from typing import Annotated

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext

"""

model = OpenAIChatModel(
    'qwen/qwen3-235b-a22b:free',
    provider=OpenRouterProvider(api_key='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'),
)
model = OpenAIChatModel(
    'claude-sonnet-4.5',
    provider=OpenAIProvider(api_key='sk-EHHj15Bkug5e7J9PjJdLKbH0gAyGfPRrcVbMeyNNw6AuXTT5',
                            base_url="https://veloera.henryyang.net/v1")
)
"""

logfire.configure(token="pylf_v1_us_NJrHYjFmpZCLMfF5YrVmTDZmvw1Bd0q30BqBQnlg9sKc")
logfire.instrument_pydantic_ai()

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)

ask_agent = Agent("openai:gpt-4o", output_type=str, instrument=True)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPT-4o."""

    docstring_notes = True

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Annotated["Answer", Edge(label="Ask the question")]:
        result = await ask_agent.run(
            "Ask a simple question with a single correct answer.",
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.new_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> "Evaluate":
        answer = input(f"{self.question}: ")
        return Evaluate(answer)


class EvaluationResult(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    "openai:gpt-4o",
    output_type=EvaluationResult,
    system_prompt="Given a question and answer, evaluate if the answer is correct.",
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Annotated[End[str], Edge(label="success")] | "Reprimand":
        assert ctx.state.question is not None
        from pydantic_ai.messages import format_as_xml  # noqa: F821

        result = await evaluate_agent.run(
            format_as_xml({"question": ctx.state.question, "answer": self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.new_messages()
        if result.output.correct:
            return End(result.output.comment)
        return Reprimand(result.output.comment)


@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        print(f"Comment: {self.comment}")
        ctx.state.question = None
        return Ask()


question_graph = Graph(nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState)
