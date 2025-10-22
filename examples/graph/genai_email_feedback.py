from __future__ import annotations as _annotations

from dataclasses import dataclass, field

import logfire
from pydantic import BaseModel, EmailStr
from pydantic_ai import Agent, ModelMessage, format_as_xml
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

logfire.configure(token="pylf_v1_us_NJrHYjFmpZCLMfF5YrVmTDZmvw1Bd0q30BqBQnlg9sKc")
logfire.instrument_pydantic_ai()

model = AnthropicModel(
    "glm-4.6",
    provider=AnthropicProvider(
        base_url="https://open.bigmodel.cn/api/anthropic/", api_key="64cd973353b54234bf5a80f6b77c012a.peYVDTsHYbHAEnLF"
    ),
)


@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]


@dataclass
class Email:
    subject: str
    body: str


@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)


email_writer_agent = Agent(
    model,
    output_type=Email,
    system_prompt="Write a welcome email to our tech blog.",
)


@dataclass
class WriteEmail(BaseNode[State]):
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> Feedback:
        if self.email_feedback:
            prompt = (
                f"Rewrite the email for the user:\n{format_as_xml(ctx.state.user)}\nFeedback: {self.email_feedback}"
            )
        else:
            prompt = f"Write a welcome email for the user:\n{format_as_xml(ctx.state.user)}"

        result = await email_writer_agent.run(
            prompt,
            message_history=ctx.state.write_agent_messages,
        )
        ctx.state.write_agent_messages += result.new_messages()
        return Feedback(result.output)


class EmailRequiresWrite(BaseModel):
    feedback: str


class EmailOk(BaseModel):
    pass


feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    model,
    output_type=EmailRequiresWrite | EmailOk,  # type: ignore
    system_prompt=("Review the email and provide feedback, email must reference the users specific interests."),
)


@dataclass
class Feedback(BaseNode[State, None, Email]):
    email: Email

    async def run(
        self,
        ctx: GraphRunContext[State],
    ) -> WriteEmail | End[Email]:
        prompt = format_as_xml({"user": ctx.state.user, "email": self.email})
        result = await feedback_agent.run(prompt)
        if isinstance(result.output, EmailRequiresWrite):
            return WriteEmail(email_feedback=result.output.feedback)
        return End(self.email)


async def main() -> None:
    user = User(
        name="John Doe",
        email="john.joe@example.com",
        interests=["Haskel", "Lisp", "Fortran"],
    )
    state = State(user)
    feedback_graph = Graph(nodes=(WriteEmail, Feedback))
    result = await feedback_graph.run(WriteEmail(), state=state)
    print(result.output)
    """
    Email(
        subject='Welcome to our tech blog!',
        body='Hello John, Welcome to our tech blog! ...',
    )
    """


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
