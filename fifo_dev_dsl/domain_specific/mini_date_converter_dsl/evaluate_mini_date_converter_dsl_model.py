"""
Test harness for evaluating the accuracy of a date expression DSL model.

This script supports two evaluation modes:

1. It can load a published test set from the Hugging Face Hub and evaluate the model's ability to
   parse each expression and return the correct DSL output.
2. It can exhaustively test `DATE_FROM_MONTH_WEEKDAY(...)` expressions using combinations of
   ordinal, weekday, and month values to verify the model generalizes correctly.

Usage:
    python evaluate_mini_date_converter_dsl_model.py \
        --container phi                              \
        --adapter mini-date-converter-dsl

    # For exhaustive test mode:
    python evaluate_mini_date_converter_dsl_model.py \
        --container phi                              \
        --adapter mini-date-converter-dsl            \
        --exhaustive
"""

from datetime import datetime
from typing import Iterator, cast
import argparse

from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter
from fifo_dev_dsl.domain_specific.mini_date_converter_dsl.core import (
    MiniDateConverterDSL,
    parse_natural_date_expression
)

def run_test_dataset(container_name: str, adapter: str) -> None:
    """
    Run the evaluation on the model test set from the Hugging Face dataset.
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
            actual_dsl, actual_output = parse_natural_date_expression(
                input_text, container_name=container_name, adapter=adapter, now=now
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

def run_exhaustive_DATE_FROM_MONTH_WEEKDAY(container_name: str, adapter: str) -> None:
    """
    Exhaustively tests DATE_FROM_MONTH_WEEKDAY generation from natural phrases like
    "the third Monday of July". This helps verify the model generalizes ordinal +
    weekday + month constructions.
    """
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

                text = f"the {ordinal_str} {day_str} of {month_str}"
                expected_dsl = f"DATE_FROM_MONTH_WEEKDAY({month_idx}, {day_idx}, {ordinal_idx})"

                # alternate call
                # text = f"two weeks after the {ordinal_str} {day_str} in {month_str}"
                # expected_dsl = f"OFFSET(DATE_FROM_MONTH_WEEKDAY({month_idx}, {day_idx}, {ordinal_idx}), 2, WEEK)"

                try:
                    actual_dsl, _ = parse_natural_date_expression(
                        text, container_name=container_name, adapter=adapter
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

    Arguments:
        --container:
            Name of the Docker container running the Airlock Model Environment where the model is
            loaded. This is used to route DSL parsing queries. (default: "phi")

        --adapter:
            Adapter identifier used by the model to interpret DSL input.
            (default: "mini-date-converter-dsl")

        --exhaustive:
            If set, evaluates an exhaustive set of DATE_FROM_MONTH_WEEKDAY expressions instead of
            the published test set.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container", default="phi",
        help="Model container name to route to"
    )
    parser.add_argument(
        "--adapter", default="mini-date-converter-dsl",
        help="Adapter name to use for generation"
    )
    parser.add_argument(
        "--exhaustive", action="store_true",
        help="Run exhaustive DATE_FROM_MONTH_WEEKDAY test suite"
    )
    args = parser.parse_args()

    if args.exhaustive:
        run_exhaustive_DATE_FROM_MONTH_WEEKDAY(args.container, args.adapter)
    else:
        run_test_dataset(args.container, args.adapter)

if __name__ == "__main__":
    main()
