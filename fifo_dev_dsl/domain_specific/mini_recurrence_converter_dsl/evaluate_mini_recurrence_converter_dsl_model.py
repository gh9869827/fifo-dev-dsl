"""
Test harness for evaluating the accuracy of a recurrence expression DSL model.

This script loads a published test set from the Hugging Face Hub and evaluates the model's ability
to parse each recurrence expression and return the correct DSL output.

Usage:
    python evaluate_mini_recurrence_converter_dsl_model.py \
        --container phi                                    \
        --adapter mini-recurrence-converter-dsl
"""

from typing import Iterator, cast
import argparse

from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter
from fifo_dev_dsl.domain_specific.mini_recurrence_converter_dsl.core import (
    MiniRecurrenceConverterDSL,
    parse_natural_recurrence_expression
)

def run_test_dataset(container_name: str, adapter: str) -> None:
    """
    Run the evaluation on the model test set from the Hugging Face dataset.
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
            actual_dsl, actual_output = parse_natural_recurrence_expression(
                input_text, container_name=container_name, adapter=adapter
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

    Arguments:
        --container:
            Name of the Docker container running the Airlock Model Environment where the model is
            loaded. This is used to route DSL parsing queries. (default: "phi")

        --adapter:
            Adapter identifier used by the model to interpret DSL input.
            (default: "mini-recurrence-converter-dsl-adapter")
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container", default="phi",
        help="Model container name to route to"
    )
    parser.add_argument(
        "--adapter", default="mini-recurrence-converter-dsl-adapter",
        help="Adapter name to use for generation"
    )
    args = parser.parse_args()

    run_test_dataset(args.container, args.adapter)

if __name__ == "__main__":
    main()
