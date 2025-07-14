import argparse
from collections import defaultdict
from typing import Iterator, cast, Callable
import re
import operator
from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter
from fifo_dev_dsl.dia.dsl.elements.element_list import ListElement
from fifo_dev_dsl.dia.dsl.elements.intent import Intent
from fifo_dev_dsl.dia.dsl.elements.slot import Slot
from fifo_dev_dsl.dia.dsl.elements.value import Value
from fifo_dev_dsl.dia.dsl.elements.value_return import ReturnValue
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.parser.parser import parse_dsl
from fifo_dev_dsl.dia.demo.calculator import Calculator
from fifo_dev_dsl.dia.demo.calculator_synthetic_data_generator import (
    create_tree,
    evaluate_tree,
    get_system_prompt,
    pretty_print_dsl,
    pretty_print_user,
    random_function_signature
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


def eval_random(delta_flag: bool) -> None:
    """
    Evaluate randomly generated arithmetic expressions using the DIA DSL pipeline.

    For each expression tree of increasing length (from 2 to 6 operations), this function:
      - Generates a random arithmetic expression
      - Converts it to a user-facing natural language prompt
      - Resolves and executes the prompt using the DSL runtime
      - Compares the model output to the expected result

    Outputs ✅/❌ for each prompt and prints per-length and overall accuracy.
    Skips cases with division by zero.

    Args:
        delta_flag (bool):
            If True, failed prompts are logged to `delta.dat`.
            If False, no logging is performed.
    """
    total_global, error_global = 0, 0
    results_by_length: dict[int, list[int]] = {k: [0, 0] for k in range(2, 7)}

    f_delta = open("delta.dat", "w", encoding="utf-8") if delta_flag else None

    try:
        for length_tree in [2, 3, 4, 5, 6]:
            total, error = 0, 0

            for _ in range(200):
                t = create_tree(length_tree)

                user = pretty_print_user(t)
                dsl_target = pretty_print_dsl(t, {
                    "add": {"name": "add", "args": ["a", "b"]},
                    "subtract": {"name": "subtract", "args": ["a", "b"]},
                    "multiply": {"name": "multiply", "args": ["a", "b"]},
                    "divide": {"name": "divide", "args": ["a", "b"]}
                })

                try:
                    expected = evaluate_tree(t)
                except ZeroDivisionError:
                    continue

                try:
                    actual = eval_prompt(user)
                except (ZeroDivisionError, TypeError, AssertionError, ValueError, KeyError):
                    actual = None

                total += 1
                results_by_length[length_tree][0] += 1
                if actual is not None and abs(actual - expected) < 0.00001:
                    outcome = "✅"
                else:
                    outcome = "❌"
                    error += 1
                    results_by_length[length_tree][1] += 1

                    fct_signature = random_function_signature()

                    if f_delta:
                        f_delta.write(
                            f"$\n{get_system_prompt(fct_signature).rstrip()}\n"
                            f"> {pretty_print_user(t).rstrip()}\n"
                            f"< {pretty_print_dsl(t, fct_signature).rstrip()}\n"
                            f"---\n"
                        )

                live_stats: list[str] = []
                for k in range(2, 7):
                    t, e = results_by_length[k]
                    acc = (t - e) / t * 100 if t > 0 else 0.0
                    live_stats.append(f"{acc:5.1f}%")

                print(" ".join(live_stats), outcome, user, dsl_target)

            total_global += total
            error_global += error

        print("\n=== Accuracy per tree length ===")
        for length, (total, error) in results_by_length.items():
            acc = (total - error) / total * 100 if total > 0 else 0.0
            print(f"Length {length}: Total={total} | Errors={error} | Accuracy={acc:.2f}%")

        print("\n=== Overall Accuracy ===")
        accuracy = (total_global - error_global) / total_global * 100 if total_global > 0 else 0.0
        print(f"Total: {total_global} | Errors: {error_global} | Accuracy: {accuracy:.2f}%")
    finally:
        if f_delta:
            f_delta.close()


def eval_test() -> None:
    """
    Evaluate a fixed test set of prompts from the Hugging Face dataset
    `a6188466/dia-intent-sequencer-calculator-dataset`.

    For each test case:
      - Parses the expected DSL string into an executable DSL tree
      - Resolves and executes the input prompt using the DSL runtime
      - Compares the model output to the expected result

    Outputs ✅/❌ per test case and prints a final summary of pass/fail counts and accuracy,
    broken down by expression depth (tree length).
    Skips cases with division by zero.
    """
    adapter_obj = DSLAdapter()
    dataset_dict = adapter_obj.from_hub_to_dataset_wide_dict(
        "a6188466/dia-intent-sequencer-calculator-dataset"
    )
    dataset_test = list(cast(Iterator[dict[str, str]], dataset_dict["test"]))

    stats: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "fail": 0})
    i = 0
    for entry in dataset_test:
        i += 1
        system_prompt = entry["system"]
        input_text = entry["in"]
        expected_dsl_text = entry["out"]

        try:
            expected, tree_length = custom_evaluate_arithmetic_dsl_tree(
                parse_dsl(expected_dsl_text),
                build_op_map_from_prompt(system_prompt)
            )
        except ZeroDivisionError:
            continue

        try:
            actual = eval_prompt(input_text)
        except (TypeError, AssertionError, ValueError, KeyError):
            actual = None

        stats[tree_length]["total"] += 1
        if actual is not None and abs(actual - expected) < 0.00001:
            print("✅", input_text, expected_dsl_text)
        else:
            print("❌", input_text, expected_dsl_text)
            stats[tree_length]["fail"] += 1

    # Print summary
    print("\n=== Accuracy per tree length ===")
    grand_total = grand_errors = 0
    for length in sorted(stats.keys()):
        total = stats[length]["total"]
        fail = stats[length]["fail"]
        success = total - fail
        acc = (success / total) * 100 if total else 0
        print(f"Length {length}: Total={total} | Errors={fail} | Accuracy={acc:.2f}%")
        grand_total += total
        grand_errors += fail

    overall_acc = ((grand_total - grand_errors) / grand_total) * 100 if grand_total else 0
    print("\n=== Overall Accuracy ===")
    print(f"Total: {grand_total} | Errors: {grand_errors} | Accuracy: {overall_acc:.2f}%")


def build_op_map_from_prompt(system_prompt: str) -> dict[str, Callable[[float, float], float]]:
    """
    Extracts a mapping from randomized intent names to arithmetic operators based on descriptions
    in the system prompt.

    Args:
        system_prompt (str):
            The full prompt defining each intent with name and description.

    Returns:
        dict[str, Callable[[float, float], float]]:
            A dictionary mapping intent names (e.g. 'csisz') to Python operator functions
            (e.g. operator.add).

    Raises:
        ValueError:
            If an intent description does not match any known arithmetic operator.
    """
    op_map: dict[str, Callable[[float, float], float]] = {}

    # Map exact keywords to functions
    op_lookup = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }

    # Regex: find lines like "- intent: csisz" followed by "description: Add two numbers."
    pattern = r"- intent:\s+(\w+)\s+description:\s+([^\n]+)"
    matches = re.findall(pattern, system_prompt)

    for name, desc in matches:
        for op_name, func in op_lookup.items():
            if op_name in desc.lower():
                op_map[name] = func
                break
        else:
            raise ValueError(f"Unrecognized operator in description: {desc}")

    return op_map

def custom_evaluate_arithmetic_dsl_tree(
    node: DslBase,
    op_map: dict[str, Callable[[float, float], float]]
) -> tuple[float, int]:
    """
    Evaluate a DIA arithmetic DSL tree and return the result along with the number of leaf values.

    This function manually walks a parsed DIA DSL tree and recursively computes the arithmetic
    result using a mapping of randomized intent names to known binary operations. It is primarily
    used to verify the correctness of model-generated DSLs from datasets with obfuscated function
    and argument names.

    Args:
        node (DslBase):
            The root node of the parsed DSL expression.

        op_map (dict[str, Callable[[float, float], float]]):
            Mapping from randomized intent names (e.g., "krygmc") to standard operators
            (e.g., operator.mul for multiplication).

    Returns:
        tuple[float, int]:
            - The computed result of the expression.
            - The number of Value nodes (i.e., numeric constants) used.

    Raises:
        TypeError:
            If the expression contains unsupported DSL node types.
    """
    if isinstance(node, Value):
        return float(node.value), 1

    if isinstance(node, ReturnValue):
        return custom_evaluate_arithmetic_dsl_tree(node.intent, op_map)

    if isinstance(node, Intent):
        func = op_map[node.name]
        slots = {slot.name: slot for slot in node.get_items()}
        a_node = slots[list(slots.keys())[0]].get_items()[0]
        b_node = slots[list(slots.keys())[1]].get_items()[0]

        a_val, a_count = custom_evaluate_arithmetic_dsl_tree(a_node, op_map)
        b_val, b_count = custom_evaluate_arithmetic_dsl_tree(b_node, op_map)

        return func(a_val, b_val), a_count + b_count

    if isinstance(node, Slot):
        return custom_evaluate_arithmetic_dsl_tree(node.get_items()[0], op_map)

    if isinstance(node, ListElement):
        return custom_evaluate_arithmetic_dsl_tree(node.get_items()[0], op_map)

    raise TypeError(f"Unsupported node type: {type(node).__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--random", action="store_true", help="Evaluate on random examples")
    parser.add_argument(
        "--delta-flag",
        action="store_true",
        help="Log failed random examples to 'delta.dat'. If not set, no file is created."
    )
    args = parser.parse_args()

    if args.random:
        eval_random(args.delta_flag)
    else:
        eval_test()
