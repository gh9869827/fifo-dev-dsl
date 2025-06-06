from typing import Any
from unittest.mock import patch
import pytest
from fifo_dev_common.introspection.tool_decorator import tool_handler
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.resolver import Resolver
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationStatus
from fifo_dev_dsl.dia.runtime.evaluator import Evaluator

class Demo:

    def __init__(self) -> None:
        self.call_trace: list[tuple[str, Any]] = []

    @tool_handler("add")
    def add(self, a: int, b: int) -> int:
        """
        Add two numbers.

        Args:
            a (int):
                first number to add

            b (int):
                second number to add

        Returns:
            int:
                Return the sum of a and b
        """
        self.call_trace.append(("add", (a, b)))
        return a + b

    @tool_handler("add_list")
    def add_list(self, v: list[int]) -> int:
        """
        Add all the numbers in the list.

        Args:
            v (list[int]):
                numbers to add

        Returns:
            int:
                Return the sum of the number in the list
        """
        self.call_trace.append(("add_list", v))
        return sum(v)

    @tool_handler("multiply")
    def multiply(self, a: int, b: int) -> int:
        """
        Multiply two numbers.

        Args:
            a (int):
                first number to multiply

            b (int):
                second number to multiply

        Returns:
            int:
                Return the product of a and b
        """
        self.call_trace.append(("multiply", (a, b)))
        return a * b

@pytest.mark.parametrize(
    "prompt, mock_dsl_response, expected_call_trace",
    [
        (
            "add 2 and 3",
            "add(a=2, b=3)",
            [("add", (2, 3))]
        ),
        (
            "sum 2, 3 and 4",
            "add_list(v=[2,3,4])",
            [("add_list", [2, 3, 4])]
        ),
        (
            "add 2 and 3 and multiply the result by 4",
            "multiply(a=4, b=add(a=2, b=3))",
            [("add", (2, 3)), ("multiply", (4, 5))]
        )
    ]
)
def test_dsl_resolution(prompt: str,
                        mock_dsl_response: str,
                        expected_call_trace: list[tuple[str, int, int]]):
    demo = Demo()
    runtime_context = LLMRuntimeContext(
        tools=[demo.add, demo.add_list, demo.multiply],
        query_sources=[]
    )

    with patch("fifo_dev_dsl.dia.resolution.resolver.call_airlock_model_server",
               return_value=mock_dsl_response):
        resolver = Resolver(runtime_context=runtime_context, prompt=prompt)
        outcome_resolver = resolver(interaction_reply=None)

        assert outcome_resolver is not None
        assert outcome_resolver.result is ResolutionResult.UNCHANGED

        evaluator = Evaluator(runtime_context, resolver.dsl_elements)
        outcome_evalutor = evaluator.evaluate()

        assert outcome_evalutor.status is EvaluationStatus.SUCCESS
        assert demo.call_trace == expected_call_trace
