from typing import Iterator, cast
import sys
import difflib
import argparse

from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter
from fifo_dev_dsl.common.llm_abstraction import (
    AirlockBackend,
    OpenAICompatibleBackend,
    LlmBackend,
    LlmRequest
)


def dsl_similarity_indicator(str1: str, str2: str) -> str:
    """
    Compare two DSL strings and return a similarity indicator with emoji and score.

    The function uses difflib to compute a similarity ratio between two strings
    and returns:
        🟢 for high similarity (>= 90)
        🟡 for medium similarity (>= 70)
        🔴 for low similarity (< 70)

    The result is returned as: "<emoji> <score>" with the score right-aligned (e.g., "🟡  78")
    
    Args:
        str1 (str):
            First DSL string

        str2 (str):
            Second DSL string

    Returns:
        str:
            An emoji indicator and a 3-digit similarity score
    """
    similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
    score = int(similarity * 100)

    if score >= 95:
        emoji = "🟢"
    elif score >= 75:
        emoji = "🟡"
    else:
        emoji = "🔴"

    return f"{emoji} {score:>3}"


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
                (default: "dia-intent-sequencer-robot-arm-adapter")

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
        description="Evaluate DIA intent sequencer robot arm adapter accuracy"
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
        default="dia-intent-sequencer-robot-arm-adapter",
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

    args = parser.parse_args()
    backend: LlmBackend

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

    adapter_obj = DSLAdapter()
    dataset_dict = adapter_obj.from_hub_to_dataset_wide_dict(
        "a6188466/dia-intent-sequencer-robot-arm-dataset"
    )
    dataset_test = list(cast(Iterator[dict[str, str]], dataset_dict["test"]))

    for entry in dataset_test:

        system_prompt = entry["system"]
        input_text = entry["in"]
        expected_dsl_text = entry["out"]

        try:
            request = LlmRequest(
                system_prompt=system_prompt,
                user_prompt=input_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            model_dsl_text = backend.complete(request)

        except RuntimeError:
            model_dsl_text = ""

        sim = dsl_similarity_indicator(model_dsl_text, expected_dsl_text)
        print(f"{sim} {model_dsl_text} {expected_dsl_text}")


if __name__ == "__main__":
    main()
