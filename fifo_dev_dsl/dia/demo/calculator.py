from fifo_dev_common.introspection.tool_decorator import tool_handler
from fifo_dev_dsl.dia.dsl.elements.intent_evaluated_success import IntentEvaluatedSuccess
from fifo_dev_dsl.dia.resolution.resolver import Resolver
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationStatus
from fifo_dev_dsl.dia.runtime.evaluator import Evaluator

class Calculator:
    """
    Demo implementation of basic arithmetic tools for use with the DIA runtime.

    This class defines four operations—add, subtract, multiply, divide—
    each exposed as a tool via @tool_handler for testing and evaluating
    intent sequencing and execution.

    Intended for synthetic data generation, model evaluation, and runtime demos.
    """

    @tool_handler("add")
    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.

        Args:
            a (float):
                first number to add

            b (float):
                second number to add

        Returns:
            float:
                the sum of a and b
        """
        return a + b

    @tool_handler("subtract")
    def subtract(self, a: float, b: float) -> float:
        """
        Subtract two numbers.

        Args:
            a (float):
                the number to subtract from

            b (float):
                the number to subtract

        Returns:
            float:
                the result of a - b
        """
        return a - b

    @tool_handler("multiply")
    def multiply(self, a: float, b: float) -> float:
        """
        Multiply two numbers.

        Args:
            a (float):
                first number

            b (float):
                second number

        Returns:
            float:
                the product of a and b
        """
        return a * b

    @tool_handler("divide")
    def divide(self, a: float, b: float) -> float:
        """
        Divide two numbers.

        Args:
            a (float):
                numerator

            b (float):
                denominator

        Returns:
            float:
                the result of a / b

        Raises:
            ZeroDivisionError: if b is zero
        """
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return a / b

if __name__ == "__main__":

    calculator = Calculator()

    runtime_context = LLMRuntimeContext(
        container_name="phi",
        intent_sequencer_adapter="dia-intent-sequencer-calculator-adapter",
        tools=[
            calculator.add,
            calculator.subtract,
            calculator.divide,
            calculator.multiply
        ],
        query_sources=[
        ]
    )

    print("> ready for command")

    # USER_PROMPT = "-9709 / -255 - 2684"
    # USER_PROMPT = "add 2 and 3 and then multiply the result by 5"
    # USER_PROMPT = "multiply together the numbers 1, 2, 3 and 4"
    USER_PROMPT = "what is 3 plus (2 times 4)?"

    print(f"> {USER_PROMPT}")

    resolver = Resolver(runtime_context, prompt=USER_PROMPT)

    # Step 1: Parse prompt and resolve into a DSL tree
    # This includes sequencing and nesting the intent, and resolving all required slot values.
    resolver.fully_resolve_in_text_mode()
    dsl_elements = resolver.dsl_elements
    dsl_elements.pretty_print_dsl()
    print(dsl_elements.to_dsl_representation())

    # Step 2: Evaluate the tree (in execution mode)
    # Each node is evaluated, recursively if needed, and values are propagated upward.
    evaluator = Evaluator(runtime_context, dsl_elements)
    outcome = evaluator.evaluate()

    # Step 3: (optional) Retry or recover if evaluation fails
    # In this example, all expressions are well-formed and runtime-safe,
    # so we assert success and inspect the final evaluation result.
    assert outcome.status == EvaluationStatus.SUCCESS

    # Display the evaluated dsl tree. It now contains a IntentEvaluatedSuccess node.
    dsl_elements.pretty_print_dsl()
    print(dsl_elements.to_dsl_representation())

    # Access the computed value from the IntentEvaluatedSuccess node and display it
    node = dsl_elements.get_children()[0]
    assert isinstance(node, IntentEvaluatedSuccess)
    print(f"Result = {node.evaluation_outcome.value}")
