from typing import Any
from unittest.mock import patch
import pytest
from fifo_dev_common.introspection.tool_decorator import tool_handler
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.resolver import Resolver
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationStatus
from fifo_dev_dsl.dia.runtime.evaluator import Evaluator
from fifo_dev_dsl.dia.resolution.interaction import Interaction, InteractionAnswer

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

    @tool_handler("retrieve_screw")
    def retrieve_screw(self, count: int, length: int) -> str:
        """
        Retrieve screws of a given length.

        Args:
            count (int):
                number of screws to retrieve

            length (int):
                length of the screws in millimeters

        Returns:
            str:
                confirmation message
        """
        self.call_trace.append(("retrieve_screw", (count, length)))
        return f"retrieved {count} screws of {length}mm"

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
        ),
        (
            "add a couple and a few",
            'add(a=F("a couple"), b=F("a few"))',
            [("add", (2, 3))]
        ),
    ]
)
def test_dsl_resolution(prompt: str,
                        mock_dsl_response: str,
                        expected_call_trace: list[tuple[str, Any]]):
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


def test_query_fill() -> None:
    """
    Test partial DSL resolution using QUERY_FILL followed by evaluation.
    """
    prompt = "add 2 and the second prime number"
    mock_dsl_response = "add(a=2, b=QUERY_FILL('what is the second prime number?'))"
    mock_dsl_answer_query_fill = "reasoning: the second prime number is 3\nvalue: 3\nabort:"
    expected_call_trace = [("add", (2, 3))]

    demo = Demo()

    runtime_context = LLMRuntimeContext(
        tools=[demo.add, demo.add_list, demo.multiply],
        query_sources=[]
    )

    with patch("fifo_dev_dsl.dia.resolution.resolver.call_airlock_model_server",
               return_value=mock_dsl_response) ,\
         patch("fifo_dev_dsl.dia.dsl.elements.query_fill.call_airlock_model_server",
               return_value=mock_dsl_answer_query_fill):

        resolver = Resolver(runtime_context=runtime_context, prompt=prompt)
        outcome_resolver = resolver(interaction_reply=None)

        assert outcome_resolver is not None
        assert outcome_resolver.result is ResolutionResult.UNCHANGED

        evaluator = Evaluator(runtime_context, resolver.dsl_elements)
        outcome_evalutor = evaluator.evaluate()

        assert outcome_evalutor.status is EvaluationStatus.SUCCESS
        assert demo.call_trace == expected_call_trace


def test_ask() -> None:
    """
    Test interactive resolution using ASK followed by evaluation.
    """

    prompt = "retrieve 2 screws"
    mock_dsl_response = 'retrieve_screw(count=2, length=ASK("what length?"))'
    mock_user_answer = "12mm"
    mock_ask_llm_answer = "12"
    expected_call_trace = [("retrieve_screw", (2, 12))]

    demo = Demo()

    runtime_context = LLMRuntimeContext(
        tools=[demo.retrieve_screw],
        query_sources=[]
    )

    with patch("fifo_dev_dsl.dia.resolution.resolver.call_airlock_model_server",
               return_value=mock_dsl_response):
        resolver = Resolver(runtime_context=runtime_context, prompt=prompt)

    first_outcome = resolver(interaction_reply=None)

    assert first_outcome.result is ResolutionResult.INTERACTION_REQUESTED
    assert first_outcome.interaction is not None

    interaction = Interaction(
        request=first_outcome.interaction,
        answer=InteractionAnswer(mock_user_answer)
    )

    with patch("fifo_dev_dsl.dia.dsl.elements.helper.call_airlock_model_server",
               return_value=mock_ask_llm_answer):
        final_outcome = resolver(interaction)

    assert final_outcome.result is ResolutionResult.UNCHANGED

    evaluator = Evaluator(runtime_context, resolver.dsl_elements)
    outcome_evalutor = evaluator.evaluate()

    assert outcome_evalutor.status is EvaluationStatus.SUCCESS
    assert demo.call_trace == expected_call_trace


def test_query_user() -> None:
    """
    Test interactive resolution using QUERY_USER followed by evaluation.
    """

    prompt = "What is the longest screw length in the inventory?"
    mock_dsl_response = (
        "QUERY_USER('What is the longest screw length in the inventory?')"
    )
    mock_query_user_llm_answer = (
        "reasoning: the longest screw is 12mm\nuser friendly answer: 12mm"
    )
    mock_user_answer = "ok"
    mock_followup_llm_answer = "ABORT()"
    expected_call_trace: list[tuple[str, Any]] = []

    demo = Demo()

    runtime_context = LLMRuntimeContext(
        tools=[demo.retrieve_screw],
        query_sources=[],
    )

    with patch(
        "fifo_dev_dsl.dia.resolution.resolver.call_airlock_model_server",
        return_value=mock_dsl_response,
    ), patch(
        "fifo_dev_dsl.dia.dsl.elements.query_user.call_airlock_model_server",
        return_value=mock_query_user_llm_answer,
    ):
        resolver = Resolver(runtime_context=runtime_context, prompt=prompt)
        first_outcome = resolver(interaction_reply=None)

    assert first_outcome.result is ResolutionResult.INTERACTION_REQUESTED
    assert first_outcome.interaction is not None
    assert first_outcome.interaction.message == "12mm"

    interaction = Interaction(
        request=first_outcome.interaction,
        answer=InteractionAnswer(mock_user_answer),
    )

    with patch(
        "fifo_dev_dsl.dia.dsl.elements.helper.call_airlock_model_server",
        return_value=mock_followup_llm_answer,
    ):
        final_outcome = resolver(interaction)

    assert final_outcome.result is ResolutionResult.UNCHANGED

    evaluator = Evaluator(runtime_context, resolver.dsl_elements)
    outcome_evalutor = evaluator.evaluate()

    assert outcome_evalutor.status is EvaluationStatus.SUCCESS
    assert demo.call_trace == expected_call_trace


def test_query_gather() -> None:
    """Test automatic resolution using QUERY_GATHER followed by evaluation."""

    prompt = "Give me all the longest screws in the inventory"
    mock_dsl_response = (
        'QUERY_GATHER("Give me all the longest screws in the inventory", '
        '"Longest screw length and count in the inventory")'
    )
    mock_query_gather_llm_answer = (
        "reasoning: the inventory was inspected\nuser friendly answer: "
        "10 screws of 12mm"
    )
    mock_intent_sequencer_answer = "retrieve_screw(count=10, length=12)"
    expected_call_trace = [("retrieve_screw", (10, 12))]

    demo = Demo()

    runtime_context = LLMRuntimeContext(
        tools=[demo.retrieve_screw],
        query_sources=[],
    )

    with patch(
        "fifo_dev_dsl.dia.resolution.resolver.call_airlock_model_server",
        return_value=mock_dsl_response,
    ), patch(
        "fifo_dev_dsl.dia.dsl.elements.query_gather.call_airlock_model_server",
        return_value=mock_query_gather_llm_answer,
    ), patch(
        "fifo_dev_dsl.dia.dsl.elements.helper.call_airlock_model_server",
        return_value=mock_intent_sequencer_answer,
    ):
        resolver = Resolver(runtime_context=runtime_context, prompt=prompt)
        outcome_resolver = resolver(interaction_reply=None)

    assert outcome_resolver is not None
    assert outcome_resolver.result is ResolutionResult.UNCHANGED

    evaluator = Evaluator(runtime_context, resolver.dsl_elements)
    outcome_evalutor = evaluator.evaluate()

    assert outcome_evalutor.status is EvaluationStatus.SUCCESS
    assert demo.call_trace == expected_call_trace
