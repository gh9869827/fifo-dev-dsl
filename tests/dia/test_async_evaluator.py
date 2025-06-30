import asyncio
from fifo_dev_common.introspection.tool_decorator import tool_handler
from fifo_dev_dsl.dia.dsl.parser.parser import parse_dsl
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationStatus
from fifo_dev_dsl.dia.runtime.async_evaluator import AsyncEvaluator
from fifo_dev_dsl.dia.dsl.elements.intent_evaluated_success import IntentEvaluatedSuccess


class Demo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    @tool_handler("add")
    async def add(self, a: int, b: int) -> int:
        """
        Add two numbers asynchronously.

        Args:
            a (int):
                first number to add

            b (int):
                second number to add

        Returns:
            int:
                the sum of ``a`` and ``b``
        """
        await asyncio.sleep(0)
        self.calls.append(("add", (a, b)))
        return a + b

    @tool_handler("multiply")
    def multiply(self, a: int, b: int) -> int:
        """
        Multiply two numbers.

        Args:
            a (int):
                first factor

            b (int):
                second factor

        Returns:
            int:
                the product of ``a`` and ``b``
        """
        self.calls.append(("multiply", (a, b)))
        return a * b


def test_async_evaluator_depth_first() -> None:
    async def run() -> None:
        demo = Demo()
        runtime = LLMRuntimeContext(tools=[demo.add, demo.multiply], query_sources=[])
        root = parse_dsl("multiply(a=3, b=add(a=1, b=2))")
        outcome = await AsyncEvaluator(runtime, root).evaluate()
        assert outcome.status is EvaluationStatus.SUCCESS
        assert demo.calls == [("add", (1, 2)), ("multiply", (3, 3))]
        assert isinstance(root.get_children()[0], IntentEvaluatedSuccess)

    asyncio.run(run())
