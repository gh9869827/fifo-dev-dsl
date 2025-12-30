"""
Test harness for evaluating the accuracy of a recurrence expression DSL model.

This script loads a published test set from the Hugging Face Hub and evaluates the model's ability
to parse each recurrence expression and return the correct DSL output.

Usage:
    # Using Airlock backend (default):
    python evaluate_mini_recurrence_converter_dsl_model.py \
        --backend-type airlock \
        --container phi \
        --adapter mini-recurrence-converter-dsl-adapter \
        --model Phi4MiniInstruct

    # Using OpenAI-compatible backend:
    python evaluate_mini_recurrence_converter_dsl_model.py \
        --backend-type openai-compatible \
        --base-url http://127.0.0.1:8001/v1 \
        --adapter your-adapter-name
"""

from typing import Iterator, cast
import argparse

from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter
from fifo_dev_dsl.common.llm_abstraction import (
    LlmBackend,
    add_backend_cli_arguments,
    create_backend_from_args,
)
from fifo_dev_dsl.domain_specific.mini_recurrence_converter_dsl.core import (
    MiniRecurrenceConverterDSL,
    parse_natural_recurrence_expression_with_backend
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
        "a6188466/mini-recurrence-converter-dsl-dataset"
    )
    dataset_test = list(cast(Iterator[dict[str, str]], dataset_dict["test"]))

    max_in_len = max(len(entry["in"]) for entry in dataset_test)
    max_out_len = max(len(entry["out"]) for entry in dataset_test)

    total = 0
    failures = 0

    for entry in dataset_test:
        total += 1

        input_text = entry["in"]
        expected_dsl_text = entry["out"]

        padded_in = input_text.ljust(max_in_len)
        padded_out = expected_dsl_text.ljust(max_out_len)

        try:
            actual_dsl, actual_output = parse_natural_recurrence_expression_with_backend(
                input_text,
                backend=backend,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort
            )
            expected_output = MiniRecurrenceConverterDSL().parse(expected_dsl_text)

            if actual_output.to_dict() == expected_output.to_dict():
                print(f"✅ {padded_in}  →  {padded_out}")
            else:
                failures += 1
                print(f"❌ {padded_in}  →  {padded_out}   (actual: {actual_dsl})")
        except (RuntimeError, ValueError, TypeError) as e:
            failures += 1
            print(f"💥 {padded_in}  →  {padded_out}   (error: {e})")

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
    """
    parser = argparse.ArgumentParser(
        description="Evaluate mini recurrence converter DSL model accuracy"
    )

    add_backend_cli_arguments(parser, default_adapter="mini-recurrence-converter-dsl-adapter")

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

    args = parser.parse_args()
    backend = create_backend_from_args(args, parser)

    run_test_dataset(backend, args.max_new_tokens, args.temperature, args.reasoning_effort)

if __name__ == "__main__":
    main()
