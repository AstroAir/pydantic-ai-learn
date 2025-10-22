import sys
from pathlib import Path

from ai_q_and_a_graph import Answer, Ask, Evaluate, QuestionState, question_graph
from pydantic_ai import ModelMessage  # noqa: F401
from pydantic_graph import End
from pydantic_graph.persistence.file import FileStatePersistence


async def main() -> None:
    answer: str | None = sys.argv[1] if len(sys.argv) > 1 else None
    persistence = FileStatePersistence(Path("question_graph.json"))
    persistence.set_graph_types(question_graph)

    if snapshot := await persistence.load_next():
        state = snapshot.state
        assert answer is not None
        node = Evaluate(answer)
    else:
        state = QuestionState()
        node = Ask()

    async with question_graph.iter(node, state=state, persistence=persistence) as run:
        while True:
            node = await run.next()
            if isinstance(node, End):
                print("END:", node.data)
                history = await persistence.load_all()
                print([e.node for e in history])
                break
            if isinstance(node, Answer):
                print(node.question)
                # > What is the capital of France?
                break
            # otherwise just continue


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
