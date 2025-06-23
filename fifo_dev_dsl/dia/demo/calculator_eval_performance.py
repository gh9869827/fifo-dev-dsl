from fifo_dev_dsl.dia.demo.calculator import Calculator
from fifo_dev_dsl.dia.demo.calculator_synthetic_data_generator import (
    create_tree,
    evaluate_tree,
    pretty_print_dsl,
    pretty_print_user
)
from fifo_dev_dsl.dia.dsl.elements.intent_evaluated_success import IntentEvaluatedSuccess
from fifo_dev_dsl.dia.resolution.resolver import Resolver
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationStatus
from fifo_dev_dsl.dia.runtime.evaluator import Evaluator

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


def eval_prompt(prompt: str) -> float:
    """
    Parse a natural language math prompt into a DSL representation,
    evaluate it using the configured tool runtime, and return the final result.

    This function assumes the prompt is complete and well-formed.
    It performs both intent sequencing and execution, and asserts
    successful evaluation.

    Args:
        prompt (str):
            Natural language expression to evaluate, e.g. "2 + 3 * 4"

    Returns:
        float:
            Result of evaluating the expression after DSL resolution and execution.

    Raises:
        AssertionError:
            If evaluation fails due to DSL resolution or execution errors.
    """
    # Step 1: Parse prompt and resolve it into a DSL tree.
    # This includes sequencing nested intents and resolving all required arguments.
    resolver = Resolver(runtime_context, prompt=prompt)
    resolver.fully_resolve_in_text_mode()
    dsl_elements = resolver.dsl_elements

    # Step 2: Evaluate the DSL tree in execution mode.
    # Each node is evaluated recursively, and results are propagated upward.
    evaluator = Evaluator(runtime_context, dsl_elements)
    outcome = evaluator.evaluate()

    # Step 3: (optional) Retry or recover if evaluation fails.
    # In this context, we expect all expressions to be well-formed and executable.
    assert outcome.status == EvaluationStatus.SUCCESS

    # Retrieve the final evaluated value from the root node.
    node = dsl_elements.get_children()[0]
    assert isinstance(node, IntentEvaluatedSuccess)

    return node.evaluation_outcome.value


if __name__ == "__main__":
    total, error = 0, 0
    for length_tree in [2, 3, 4, 5, 6]:
        for _ in range(30):
            t = create_tree(length_tree)

            user = pretty_print_user(t)
            dsl_target = pretty_print_dsl(t, {
                "add": {
                    "name": "add",
                    "args": ["a", "b"]
                },
                "subtract": {
                    "name": "subtract",
                    "args": ["a", "b"]
                },
                "multiply": {
                    "name": "multiply",
                    "args": ["a", "b"]
                },
                "divide": {
                    "name": "divide",
                    "args": ["a", "b"]
                }
            })

            try:
                expected = evaluate_tree(t)
            except ZeroDivisionError:
                continue

            try:
                actual = eval_prompt(user)
            except (TypeError, AssertionError, ValueError) as e:
                actual = None

            total += 1
            if actual is not None and abs(actual - expected) < 0.00001:
                print("✅", user, dsl_target)
            else:
                print("❌", user, dsl_target)
                error += 1

    accuracy = (total - error) / total * 100
    print(f"Total: {total} | Errors: {error} | Accuracy: {accuracy:.2f}%")
