from count_down import CountDown, CountDownState, count_down_graph
from pydantic_graph import End, FullStatePersistence


async def main() -> None:
    state = CountDownState(counter=5)
    persistence = FullStatePersistence()
    async with count_down_graph.iter(CountDown(), state=state, persistence=persistence) as run:
        node = run.next_node
        while not isinstance(node, End):
            print("Node:", node)
            # > Node: CountDown()
            # > Node: CountDown()
            # > Node: CountDown()
            # > Node: CountDown()
            if state.counter == 2:
                break
            node = await run.next(node)

        print(run.result)
        # > None

        for step in persistence.history:
            print("History Step:", step.state, step.state)
            # > History Step: CountDownState(counter=5) CountDownState(counter=5)
            # > History Step: CountDownState(counter=4) CountDownState(counter=4)
            # > History Step: CountDownState(counter=3) CountDownState(counter=3)
            # > History Step: CountDownState(counter=2) CountDownState(counter=2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
