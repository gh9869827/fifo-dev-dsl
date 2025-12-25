"""
Test harness for evaluating the accuracy of a recurrence expression DSL model.

This script loads a published test set from the Hugging Face Hub and evaluates the model's ability
to parse each recurrence expression and return the correct DSL output.

Usage:
    # Using Airlock backend (default):
    python evaluate_mini_recurrence_converter_dsl_model.py \
        --backend-type airlock \
        --container phi \
        --adapter mini-recurrence-converter-dsl-adapter

    # Using OpenAI-compatible backend:
    python evaluate_mini_recurrence_converter_dsl_model.py \
        --backend-type openai-compatible \
        --base-url http://127.0.0.1:8001/v1 \
        --model your-model-name
"""

import sys
from typing import Iterator, cast
import argparse

from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter
from fifo_dev_dsl.common.llm_abstraction import AirlockBackend, OpenAICompatibleBackend, LlmBackend
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

        reasoning_effort (str | None):
            Reasoning effort level for reasoning models. When None, the parameter is
            not passed to the backend.
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

    Arguments:
        --backend-type:
            Type of LLM backend to use. Options: 'airlock', 'openai-compatible'.
            (default: "airlock")

        Airlock backend parameters (used when --backend-type=airlock):
            --container:
                Name of the Docker container running the Airlock Model Environment.
                (default: "phi")

            --adapter:
                Adapter identifier used by the model to interpret DSL input.
                (default: "mini-recurrence-converter-dsl-adapter")

            --host:
                Base URL of the Airlock model server.
                (default: "http://127.0.0.1:8000")

        OpenAI-compatible backend parameters (used when --backend-type=openai-compatible):
            --base-url:
                Base URL for the OpenAI-compatible server, including "/v1".
                (required for openai-compatible backend)

            --model:
                Model name exposed by the server.
                (required for openai-compatible backend)

            --api-key:
                API key for the OpenAI-compatible server.
                (default: "EMPTY")

        LLM generation parameters:
            --max-new-tokens:
                Maximum number of tokens to generate. (default: 1024)

            --temperature:
                Sampling temperature (0.0 = greedy). (default: 0.0)
    """
    parser = argparse.ArgumentParser(
        description="Evaluate mini recurrence converter DSL model accuracy"
    )

    # Backend type selection
    parser.add_argument(
        "--backend-type",
        default="airlock",
        choices=["airlock", "openai-compatible"],
        help="Type of LLM backend to use"
    )

    # Airlock backend parameters
    parser.add_argument(
        "--container",
        default="phi",
        help="Airlock container name (for airlock backend)"
    )
    parser.add_argument(
        "--adapter",
        default="mini-recurrence-converter-dsl-adapter",
        help="Adapter name (for airlock backend)"
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:8000",
        help="Airlock server URL (for airlock backend)"
    )

    # OpenAI-compatible backend parameters
    parser.add_argument(
        "--base-url",
        help="Base URL for OpenAI-compatible server (for openai-compatible backend)"
    )
    parser.add_argument(
        "--model",
        help="Model name (for openai-compatible backend)"
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key (for openai-compatible backend)"
    )

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

    # Create backend based on type
    if args.backend_type == "airlock":
        backend = AirlockBackend(
            container_name=args.container,
            adapter=args.adapter,
            host=args.host
        )
    elif args.backend_type == "openai-compatible":
        if not args.base_url or not args.model:
            parser.error(
                "--base-url and --model are required when using openai-compatible backend"
            )
        backend = OpenAICompatibleBackend(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key
        )
    else:
        parser.error(f"Unknown backend type: {args.backend_type}")
        sys.exit(1)

    run_test_dataset(backend, args.max_new_tokens, args.temperature, args.reasoning_effort)

    # Run evaluation
    run_test_dataset(backend, args.max_new_tokens, args.temperature)

if __name__ == "__main__":
    main()
