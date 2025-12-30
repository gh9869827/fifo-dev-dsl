"""
Test harness for evaluating the accuracy of a date expression DSL model.

This script supports two evaluation modes:

1. It can load a published test set from the Hugging Face Hub and evaluate the model's ability to
   parse each expression and return the correct DSL output.
2. It can test `DATE_FROM_MONTH_WEEKDAY(...)` expressions using template-based
   variations across ordinal, weekday, and month values to evaluate the model's
   generalization across these constructions.

Usage:
    # Using Airlock backend (default):
    python evaluate_mini_date_converter_dsl_model.py \
        --backend-type airlock \
        --container phi \
        --adapter mini-date-converter-dsl-adapter \
        --model Phi4MiniInstruct

    # Using OpenAI-compatible backend:
    python evaluate_mini_date_converter_dsl_model.py \
        --backend-type openai-compatible \
        --base-url http://127.0.0.1:8001/v1 \
        --adapter your-adapter-name

    # For template-based variations test mode:
    python evaluate_mini_date_converter_dsl_model.py \
        --backend-type airlock \
        --container phi \
        --adapter mini-date-converter-dsl-adapter \
        --model Phi4MiniInstruct \
        --template-base 1
"""

from datetime import datetime
from typing import Iterator, cast
import argparse

from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter
from fifo_dev_dsl.common.llm_abstraction import (
    LlmBackend,
    add_backend_cli_arguments,
    create_backend_from_args,
)
from fifo_dev_dsl.domain_specific.mini_date_converter_dsl.core import (
    MiniDateConverterDSL,
    parse_natural_date_expression_with_backend
)

def run_test_dataset(backend: LlmBackend, max_new_tokens: int, temperature: float, reasoning_effort: str | None = None) -> None:
    """
    Run the evaluation on the model test set from the Hugging Face dataset.

    Args:
        backend (LlmBackend):
            LLM backend instance to use for parsing.

        max_new_tokens (int):
            Maximum tokens to generate.

        temperature (float):
            Sampling temperature.

        reasoning_effort (str | None, optional):
            Reasoning effort level for reasoning models. Only applicable when using
            reasoning-capable models. When None, the parameter is not passed to the
            backend. Defaults to None.
    """
    adapter_obj = DSLAdapter()
    dataset_dict = adapter_obj.from_hub_to_dataset_wide_dict(
        "a6188466/mini-date-converter-dsl-dataset"
    )
    dataset_test = list(cast(Iterator[dict[str, str]], dataset_dict["test"]))

    max_in_len = max(len(entry["in"]) for entry in dataset_test)
    max_out_len = max(len(entry["out"]) for entry in dataset_test)

    total = 0
    failures = 0

    for entry in dataset_test:
        total += 1
        now = datetime.now()
        input_text = entry["in"]
        expected_dsl_text = entry["out"]

        padded_in = input_text.ljust(max_in_len)
        padded_out = expected_dsl_text.ljust(max_out_len)

        try:
            actual_dsl, actual_output = parse_natural_date_expression_with_backend(
                input_text,
                now=now,
                backend=backend,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort
            )
            expected_output = MiniDateConverterDSL(now=now).parse(expected_dsl_text)

            if actual_output == expected_output:
                print(f"✅ {padded_in}  →  {padded_out}")
            else:
                failures += 1
                print(f"❌ {padded_in}  →  {padded_out}   (actual: {actual_dsl})")
        except (RuntimeError, ValueError, TypeError) as e:
            failures += 1
            print(f"💥 {padded_in}  →  {padded_out}   (error: {e})")

    print(f"\nSummary: {total - failures}/{total} passed, {failures} failed. "
          f"({((total - failures) / total) * 100:.2f}% success)")

def run_template_base_DATE_FROM_MONTH_WEEKDAY(backend: LlmBackend,
                                              max_new_tokens: int,
                                              temperature: float,
                                              reasoning_effort: str | None = None,
                                              template: int = 1) -> None:
    """
    Tests DATE_FROM_MONTH_WEEKDAY generation using template-based variations
    of natural phrases like "the third Monday of July". This evaluates the model's
    generalization across ordinal, weekday, and month constructions.

    Args:
        backend (LlmBackend):
            LLM backend instance to use for parsing.

        max_new_tokens (int):
            Maximum tokens to generate.

        temperature (float):
            Sampling temperature.

        reasoning_effort (str | None, optional):
            Reasoning effort level for reasoning models. Only applicable when using
            reasoning-capable models. When None, the parameter is not passed to the
            backend. Defaults to None.

        template (int):
            Either 1 for using the template "the {ordinal_str} {day_str} of {month_str}" or
            2 for using "two weeks after the {ordinal_str} {day_str} in {month_str}"

    Raises:
        ValueError: template must be 1 or 2
    """
    if template not in (1, 2):
        raise ValueError("template must be 1 or 2")

    ordinals = [(1, "first"), (2, "second"), (3, "third"), (4, "fourth")]
    days = [(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"), (3, "Thursday"),
            (4, "Friday"), (5, "Saturday"), (6, "Sunday")]
    months = [(1, "January"), (2, "February"), (3, "March"), (4, "April"), (5, "May"),
              (6, "June"), (7, "July"), (8, "August"), (9, "September"), (10, "October"),
              (11, "November"), (12, "December")]

    total = 0
    failures = 0

    for ordinal_idx, ordinal_str in ordinals:
        for day_idx, day_str in days:
            for month_idx, month_str in months:
                total += 1

                if template == 1:
                    text = f"the {ordinal_str} {day_str} of {month_str}"
                    expected_dsl = f"DATE_FROM_MONTH_WEEKDAY({month_idx}, {day_idx}, {ordinal_idx})"
                else:
                    text = f"two weeks after the {ordinal_str} {day_str} in {month_str}"
                    expected_dsl = (
                        f"OFFSET("
                        f"DATE_FROM_MONTH_WEEKDAY({month_idx}, {day_idx}, {ordinal_idx}), "
                        f"2, WEEK)"
                    )

                try:
                    actual_dsl, _ = parse_natural_date_expression_with_backend(
                        text,
                        backend=backend,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort
                    )
                except (RuntimeError, ValueError, TypeError) as e:
                    failures += 1
                    print(f"💥 {text:<40} → {expected_dsl:<40} (error: {e})")
                    continue

                if actual_dsl == expected_dsl:
                    print(f"✅ {text:<40} → {expected_dsl}")
                else:
                    failures += 1
                    print(f"❌ {text:<40} → {expected_dsl:<40} (actual: {actual_dsl})")

    print(f"\nSummary: {total - failures}/{total} passed, {failures} failed. "
          f"({((total - failures) / total) * 100:.2f}% success)")


def main() -> None:
    """
    Runs the evaluation loop over the test dataset, printing per-example results and a final
    summary.

    For available command-line arguments, see add_backend_cli_arguments() in
    fifo_dev_dsl.common.llm_abstraction.

    LLM generation parameters:
        --max-new-tokens:
            Maximum number of tokens to generate. (default: 1024)

        --temperature:
            Sampling temperature (0.0 = greedy). (default: 0.0)

    Evaluation options:
        --template-base:
            If set, evaluates DATE_FROM_MONTH_WEEKDAY expressions using
            template-based variations, focusing on this specific DSL function
            rather than the broader published test set.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate mini date converter DSL model accuracy"
    )

    add_backend_cli_arguments(parser, default_adapter="mini-date-converter-dsl-adapter")

    # LLM generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Reasoning effort level for reasoning models (default: None, not passed to backend)"
    )

    # Evaluation options
    parser.add_argument(
        "--template-base",
        type=int,
        choices=(1, 2),
        help=(
            "Template variation for DATE_FROM_MONTH_WEEKDAY"
            "(1 = base form, 2 = compositional offset form)"
        )
    )

    args = parser.parse_args()
    backend = create_backend_from_args(args, parser)

    # Run evaluation
    if args.template_base:
        run_template_base_DATE_FROM_MONTH_WEEKDAY(
            backend,
            args.max_new_tokens,
            args.temperature,
            args.reasoning_effort,
            template=args.template_base
        )
    else:
        run_test_dataset(backend, args.max_new_tokens, args.temperature, args.reasoning_effort)

if __name__ == "__main__":
    main()
