from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from rich.prompt import Prompt


@dataclass
class MachineState:
    user_balance: float = 0.0
    product: str | None = None


@dataclass
class InsertCoin(BaseNode[MachineState]):
    async def run(self, ctx: GraphRunContext[MachineState]) -> CoinsInserted:
        return CoinsInserted(float(Prompt.ask("Insert coins")))


@dataclass
class CoinsInserted(BaseNode[MachineState]):
    amount: float

    async def run(self, ctx: GraphRunContext[MachineState]) -> SelectProduct | Purchase:
        ctx.state.user_balance += self.amount
        if ctx.state.product is not None:
            return Purchase(ctx.state.product)
        return SelectProduct()


@dataclass
class SelectProduct(BaseNode[MachineState]):
    async def run(self, ctx: GraphRunContext[MachineState]) -> Purchase:
        return Purchase(Prompt.ask("Select product"))


PRODUCT_PRICES = {
    "water": 1.25,
    "soda": 1.50,
    "crisps": 1.75,
    "chocolate": 2.00,
}


@dataclass
class Purchase(BaseNode[MachineState, None, None]):
    product: str

    async def run(self, ctx: GraphRunContext[MachineState]) -> End | InsertCoin | SelectProduct:
        if price := PRODUCT_PRICES.get(self.product):
            ctx.state.product = self.product
            if ctx.state.user_balance >= price:
                ctx.state.user_balance -= price
                return End(None)
            diff = price - ctx.state.user_balance
            print(f"Not enough money for {self.product}, need {diff:0.2f} more")
            # > Not enough money for crisps, need 0.75 more
            return InsertCoin()
        print(f"No such product: {self.product}, try again")
        return SelectProduct()


vending_machine_graph = Graph(nodes=[InsertCoin, CoinsInserted, SelectProduct, Purchase])


async def main() -> None:
    state = MachineState()
    await vending_machine_graph.run(InsertCoin(), state=state)
    print(f"purchase successful item={state.product} change={state.user_balance:0.2f}")
    # > purchase successful item=crisps change=0.25


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
