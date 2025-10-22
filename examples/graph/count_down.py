from __future__ import annotations as _annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class CountDownState:
    counter: int


@dataclass
class CountDown(BaseNode[CountDownState, None, int]):
    async def run(self, ctx: GraphRunContext[CountDownState]) -> CountDown | End[int]:
        if ctx.state.counter <= 0:
            return End(ctx.state.counter)
        ctx.state.counter -= 1
        return CountDown()


count_down_graph = Graph(nodes=[CountDown])


async def main() -> None:
    state = CountDownState(counter=3)
    async with count_down_graph.iter(CountDown(), state=state) as run:
        async for node in run:
            print("Node:", node)
            # > Node: CountDown()
            # > Node: CountDown()
            # > Node: CountDown()
            # > Node: CountDown()
            # > Node: End(data=0)
    if run.result is not None:
        print("Final output:", run.result.output)
        # > Final output: 0


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
